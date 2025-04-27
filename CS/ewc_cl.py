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
from torch.utils.data import TensorDataset
from util import parse_args, set_seed, build_model_tokenizer_config
from preprocess_replay import load_sum_replay_data_by_tag, convert_examples_to_features
from models.sum_finetune import SumModel
from metric import smooth_bleu
from metric.meteor.meteor import Meteor
from metric.rouge.rouge import Rouge
from models.ewc import EWC, construct_exemplars_ours


def train(args, train_examples, train_origin_examples, train_replay_examples, train_dataset,
                eval_examples, eval_origin_examples, eval_replay_examples, eval_dataset,
                model, tokenizer):
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=4)
    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 2

    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    best_ppl = 1e6
    patience = 0
    model.zero_grad()

    # mode = "train"
    model_to_save = model.module if hasattr(model, 'module') else model

    if args.cl_num == 0:
        for idx in range(args.epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            tr_loss = 0
            tr_num = 0
            for step, batch in enumerate(bar):
                source_id, source_mask, target_id, target_mask = [x.to(args.local_rank) for x in batch]
                model.train()

                out = model(input_ids=source_id, attention_mask=source_mask, labels=target_id)
                loss = out.loss

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
                    results = evaluate(args, eval_examples, eval_origin_examples, eval_replay_examples, eval_dataset, model, tokenizer)

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
                            "scheduler": scheduler.state_dict()
                        }
                        torch.save(save_content, output_file)

                    else:
                        patience += 1
                        if patience > args.patience * 5:
                            logger.info('patience greater than {}, early stop!'.format(args.patience))
                            return

    else:
        for idx in range(args.epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            tr_loss = 0
            tr_num = 0

            if 'ewc' in args.method and args.cl_num > 0:
                replay_examples = train_replay_examples
                replay_features = convert_examples_to_features(replay_examples, tokenizer, args, 'train')
                replay_all_source_ids = replay_features['source_ids']
                replay_all_source_mask = replay_features['source_mask']
                replay_all_target_ids = replay_features['target_ids']
                replay_all_target_mask = replay_features['target_mask']

                replay_data = TensorDataset(replay_all_source_ids, replay_all_source_mask, replay_all_target_ids, replay_all_target_mask)
                replay_sampler = SequentialSampler(replay_data)
                replay_dataloader = DataLoader(replay_data, sampler=replay_sampler, batch_size=1)
                ewc = EWC(model, replay_dataloader, args.device, len(replay_examples), tokenizer)

            for step, batch in enumerate(bar):
                source_id, source_mask, target_id, target_mask = [x.to(args.local_rank) for x in batch]
                model.train()

                out = model(input_ids=source_id, attention_mask=source_mask, labels=target_id)
                loss = out.loss

                if args.cl_num > 0 and 'ewc' in args.method:
                    ewc_loss = ewc.penalty(model)
                    loss = loss + args.ewc_weight * ewc_loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
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
                    results = evaluate(args, eval_examples, eval_origin_examples, eval_replay_examples, eval_dataset, model, tokenizer)

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
                            "scheduler": scheduler.state_dict()
                        }
                        torch.save(save_content, output_file)

                    else:
                        patience += 1
                        if patience > args.patience * 5:
                            logger.info('patience greater than {}, early stop!'.format(args.patience))
                            return

    if args.method in ["ewc", "emr"]:
        construct_exemplars_ours(model_to_save, args, train_origin_examples, eval_origin_examples,
                                 tokenizer, args.device, 'random')


def evaluate(args, eval_examples, eval_origin_examples, eval_replay_examples, eval_dataset, model, tokenizer):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=4)

    eval_loss = 0
    batch_num = 0
    model.eval()

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        source_id, source_mask, target_id, target_mask = [x.to(args.local_rank) for x in batch]
        with torch.no_grad():
            out = model(input_ids=source_id, attention_mask=source_mask, labels=target_id)
            loss = out.loss

        eval_loss += loss.item()
        batch_num += 1

    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)

    result = {
        "eval_ppl": eval_ppl
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, test_examples, test_origin_examples, test_replay_examples, test_dataset, model, tokenizer):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=4)

    eval_loss = 0
    batch_num = 0
    pred_ids = []
    model.eval()
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        source_id, source_mask, target_id, target_mask = [x.to(args.local_rank) for x in batch]
        with torch.no_grad():
            # loss = model(source_id, source_mask, target_id, target_mask)
            # preds = model.generate_preds(source_id, source_mask)
            out = model(input_ids=source_id, attention_mask=source_mask, labels=target_id)
            loss = out.loss
            generated_preds = model.generate(input_ids=source_id, attention_mask=source_mask, num_beams=args.beam_size,
                                             max_length=args.max_output_tokens)
            preds = list(generated_preds.cpu().numpy())

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

    mymodel, tokenizer, config = build_model_tokenizer_config(args)
    mymodel = mymodel.to(local_rank)
    # mymodel = SumModel(model, args).to(local_rank)

    task_nums = args.task_list.split(',')

    if args.do_train:
        if args.task_type == "sum":
            for num, task in enumerate(task_nums):
                task_id = int(task[-1])
                print(task_id)

                logger.info(f"-----training {task} %s-----")
                args.cl_num = task_id

                if task_id != 0:
                    train_examples, train_origin_examples, train_replay_examples, train_dataset = load_sum_replay_data_by_tag(args, tokenizer, "train", tag=task)
                    eval_examples, eval_origin_examples, eval_replay_examples, eval_dataset = load_sum_replay_data_by_tag(args, tokenizer, "eval", tag=task)
                    logger.info(f"train_size for {task}: {str(len(train_dataset))}")

                    checkpoint_prefix = f'checkpoint-best-ppl/model-{args.cl_num - 1}.bin'
                    output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
                    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
                    checkpoint = torch.load(output_dir, map_location=map_location)
                    mymodel.load_state_dict(checkpoint['model_state_dict'])

                else:
                    train_examples, train_origin_examples, train_dataset = load_sum_replay_data_by_tag(args, tokenizer, "train", tag=task)
                    eval_examples, eval_origin_examples, eval_dataset = load_sum_replay_data_by_tag(args, tokenizer, "eval", tag=task)
                    logger.info(f"train_size for {task}: {str(len(train_dataset))}")
                    train_replay_examples = None
                    eval_replay_examples = None

                train(args, train_examples, train_origin_examples, train_replay_examples, train_dataset,
                      eval_examples, eval_origin_examples, eval_replay_examples, eval_dataset,
                      mymodel, tokenizer)

        else:
            assert False, "Invalid Task Type."

    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-ppl/model-{args.check_order}.bin'
        output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(output_dir, map_location=map_location)
        mymodel.load_state_dict(checkpoint['model_state_dict'])

        if args.task_type == "sum":
            for num, task in enumerate(range(args.check_order + 1)):
                task_id = task

                logger.info(f"-----testing {task} %s-----")
                args.cl_num = task_id
                test_tag = task_nums[task]

                if task_id != 0:
                    test_examples, test_origin_examples, test_replay_examples, test_dataset = load_sum_replay_data_by_tag(args, tokenizer, "test", tag=test_tag)
                    logger.info(f"test_size for {task}: {str(len(test_dataset))}")
                else:
                    test_examples, test_origin_examples, test_dataset = load_sum_replay_data_by_tag(args, tokenizer, "test", tag=test_tag)
                    logger.info(f"test_size for {task}: {str(len(test_dataset))}")
                    test_replay_examples = None

                test(args, test_examples, test_origin_examples, test_replay_examples, test_dataset,
                     mymodel, tokenizer)

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