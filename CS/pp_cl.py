import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from util import parse_args, set_seed, build_model_tokenizer_config
from preprocess import load_sum_data_by_tag
from models.sum_pp import SumPPModel
from metric import smooth_bleu
from metric.meteor.meteor import Meteor
from metric.rouge.rouge import Rouge



def train(args, train_dataset, eval_dataset, model, local_rank):
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=4)
    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 2

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    best_ppl = 1e6
    patience = 0
    model.zero_grad()

    for idx in range(args.epochs):
        train_sampler.set_epoch(idx)
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_loss = 0
        tr_num = 0
        for step, batch in enumerate(bar):
            source_id, source_mask, target_id, target_mask = [x.to(local_rank) for x in batch]
            model.train()

            loss, _ = model(source_id, source_mask, target_id, target_mask)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % args.save_steps == 0:
                logger.warning(f"epoch {idx} step {step + 1} loss {round(tr_loss / tr_num, 5)}")
                tr_loss = 0
                tr_num = 0

            # backward
            loss.backward()
            # truncate the gradient, used to prevent exploding gradient.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # save model after save_steps.
            if (step + 1) % args.save_steps == 0:
                results = evaluate(args, eval_dataset, model, local_rank)

                if results["eval_ppl"] < best_ppl and dist.get_rank() == 0:
                    patience = 0
                    best_ppl = results["eval_ppl"]
                    checkpoint_prefix = "checkpoint-best-ppl"
                    output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_file = os.path.join(output_dir, f"model-{args.cl_num}.bin")
                    save_content = {
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "new_prompt": results["new_prompt"],
                    }
                    torch.save(save_content, output_file)

                else:
                    patience += 1
                    if patience > args.patience * 5:
                        logger.info('patience greater than {}, early stop!'.format(args.patience))
                        return


def evaluate(args, eval_dataset, model, local_rank):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=4)

    eval_loss = 0
    batch_num = 0
    model.eval()

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        source_id, source_mask, target_id, target_mask = [x.to(local_rank) for x in batch]
        with torch.no_grad():
            loss, new_prompt = model(source_id, source_mask, target_id, target_mask)

        eval_loss += loss.item()
        batch_num += 1

    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)

    result = {
        "eval_ppl": eval_ppl,
        "new_prompt": new_prompt
    }

    logger.info("***** Eval results *****")
    for idx, key in enumerate(sorted(result.keys())):
        if idx > 0:
            break
        else:
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, test_examples, test_dataset, model, tokenizer, local_rank):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=4)

    eval_loss = 0
    batch_num = 0
    pred_ids = []
    model.eval()
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        source_id, source_mask, target_id, target_mask = [x.to(local_rank) for x in batch]
        with torch.no_grad():
            loss, _ = model(source_id, source_mask, target_id, target_mask)
            preds = model.generate_preds(source_id, source_mask)

        eval_loss += loss.item()
        batch_num += 1
        pred_ids.extend(preds)

    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    refs_dict = {}
    preds_dict = {}
    for i in range(len(pred_nls)):
        preds_dict[i] = [pred_nls[i]]
        refs_dict[i] = [test_examples[i].target.strip()]

    score_Meteor, scores_Meteor = Meteor().compute_score(refs_dict, preds_dict)
    score_Rouge, scores_Rouge = Rouge().compute_score(refs_dict, preds_dict)

    eval_accs = []
    predictions = []
    labels = []

    for pred_nl, gold_nl in zip(pred_nls, test_examples):
        eval_accs.append(pred_nl.strip() == gold_nl.target.strip())
        predictions.append(str(gold_nl.idx) + '\t' + pred_nl)
        labels.append(str(gold_nl.idx) + '\t' + gold_nl.target.strip() + '\n')

    (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, labels)
    eval_bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)

    result = {
        "eval_ppl": eval_ppl,
        "eval_bleu": eval_bleu,
        "meteor": round(score_Meteor * 100, 2),
        "rouge-l": round(score_Rouge * 100, 2),
        "em": np.mean(eval_accs),
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))


def main(args):
    local_rank = args.local_rank
    world_size = torch.cuda.device_count()
    print(f"Start running basic DDP example on rank {local_rank}.")

    dist.init_process_group(backend="nccl")
    # dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    gpu_list = args.available_gpu.split(',')
    args.gpu_list = [int(x) for x in gpu_list]
    args.n_gpu = len(args.gpu_list)

    set_seed(args)

    model, tokenizer, config = build_model_tokenizer_config(args)

    task_nums = args.task_list.split(',')

    if args.do_train:
        if args.task_type == "sum":
            for num, task in enumerate(task_nums):
                logger.info(f"-----training {task} %s-----")
                args.cl_num = num
                train_examples, train_dataset = load_sum_data_by_tag(args, tokenizer, "train", tag=task)
                eval_examples, eval_dataset = load_sum_data_by_tag(args, tokenizer, "eval", tag=task)
                logger.info(f"train_size for {task}: {str(len(train_dataset))}")

                if num == 0:
                    mymodel = SumPPModel(model, args).to(local_rank)

                else:
                    checkpoint_prefix = f'checkpoint-best-ppl/model-{args.cl_num - 1}.bin'
                    output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
                    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
                    checkpoint = torch.load(output_dir, map_location=map_location)
                    previous_prompt = checkpoint["new_prompt"]
                    mymodel = SumPPModel(model, args, previous_prompt).to(local_rank)
                    mymodel.load_state_dict(checkpoint['model_state_dict'])

                for name, param in mymodel.named_parameters():
                    if param.requires_grad == True:
                        print(name)

                train(args, train_dataset, eval_dataset, mymodel, local_rank)

        else:
            assert False, "Invalid Task Type."

    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-ppl/model-{args.check_order}.bin'
        output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(output_dir, map_location=map_location)
        previous_prompt = checkpoint["new_prompt"]
        mymodel = SumPPModel(model, args, previous_prompt).to(local_rank)
        mymodel.load_state_dict(checkpoint['model_state_dict'])

        if args.task_type == "sum":
            for num, task in enumerate(range(args.check_order + 1)):
                logger.info(f"-----testing {task} %s-----")
                args.cl_num = num
                test_tag = task_nums[task]
                test_examples, test_dataset = load_sum_data_by_tag(args, tokenizer, "test", tag=test_tag)
                logger.info(f"test_size for {task}: {str(len(test_dataset))}")
                test(args, test_examples, test_dataset, mymodel, tokenizer, local_rank)

        else:
            assert False, "Invalid Task Type."


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.available_gpu

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.log_name = os.path.join(args.output_dir, "log.log")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    # write to file
    handler = logging.FileHandler(args.log_name)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    main(args)