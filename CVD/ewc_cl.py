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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from torch.nn.parallel import DistributedDataParallel as DDP

from util import parse_args, set_seed, build_model_tokenizer_config
from preprocess_replay import load_svd_replay_data_by_tag, TextDataset
from models.ewc import EWC, Model, calculate_coefficient, construct_exemplars_ours


def train(args, train_dataset, eval_dataset, model, tokenizer, local_rank):
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=4)
    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 2

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    best_f1 = 0.0
    patience = 0
    tr_loss = 0.0
    global_step = 0
    model.zero_grad()

    if args.cl_num > 0 and 'ewc' in args.method:
        adaptive_weight = calculate_coefficient(train_dataset.origin_data, train_dataset.replay_examples)
    else:
        adaptive_weight = 1
    logger.info("  Adaptive weight = %f", adaptive_weight)

    model_to_save = model.module if hasattr(model, 'module') else model

    if args.cl_num == 0:
        for idx in range(args.epochs):
            train_sampler.set_epoch(idx)
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            tr_loss = 0
            tr_num = 0
            for step, batch in enumerate(bar):
                input_id, input_mask, label = [x.to(local_rank) for x in batch]
                model.train()
                loss, prob = model(input_id, label)

                # if args.n_gpu > 1:
                #     loss = loss.mean()
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

                if (step + 1) % args.save_steps == 0:
                    results = evaluate(args, eval_dataset, model, tokenizer, local_rank)

                    if results["eval_f1"] >= best_f1 and dist.get_rank() == 0:
                        patience = 0
                        best_f1 = results["eval_f1"]
                        checkpoint_prefix = "checkpoint-best-f1"
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
            if 'ewc' in args.method and args.cl_num > 0:
                replay_examples = TextDataset(args, tokenizer, args.train_examplar_path)
                replay_sampler = SequentialSampler(replay_examples)
                replay_dataloader = DataLoader(replay_examples, sampler=replay_sampler, batch_size=1)
                ewc = EWC(model, replay_dataloader, args.device, len(replay_examples))
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            tr_num = 0
            train_loss = 0
            for step, batch in enumerate(bar):
                # inputs = batch[0].to(args.device)
                # labels = batch[1].to(args.device)
                input_id, input_mask, label = [x.to(local_rank) for x in batch]
                model.train()
                loss, prob = model(input_id, label)
                # loss, logits = model(inputs, labels)
                if args.cl_num > 0 and 'ewc' in args.method:
                    ewc_loss = ewc.penalty(model)
                    loss = loss + args.ewc_weight * adaptive_weight * ewc_loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # report loss
                tr_loss += loss.item()
                tr_num += 1
                if (step + 1) % args.save_steps == 0:
                    logger.warning(f"epoch {idx} step {step + 1} loss {round(tr_loss / tr_num, 5)}")
                    tr_loss = 0
                    tr_num = 0

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # tr_loss += loss.item()
                # tr_num += 1
                # train_loss += loss.item()
                # if avg_loss == 0:
                #     avg_loss = tr_loss
                # avg_loss = round(train_loss / tr_num, 5)
                # bar.set_description("epoch {} loss {}".format(idx, avg_loss))

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                if (step + 1) % args.save_steps == 0:
                    results = evaluate(args, eval_dataset, model, tokenizer, local_rank)

                    if results["eval_f1"] >= best_f1 and dist.get_rank() == 0:
                        patience = 0
                        best_f1 = results["eval_f1"]
                        checkpoint_prefix = "checkpoint-best-f1"
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
        train_replay_examples = train_dataset.origin_data
        eval_replay_examples = eval_dataset.origin_data
        construct_exemplars_ours(model_to_save, args, train_replay_examples, eval_replay_examples,
                                 tokenizer, args.device, 'random')




def evaluate(args, eval_dataset, mymodel, tokenizer, local_rank):
    # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=4)

    pred_label = []
    true_label = []
    mymodel.eval()
    for batch in eval_dataloader:
        input_id, input_mask, label = [x.to(local_rank) for x in batch]
        with torch.no_grad():
            loss, prob = mymodel(input_id, label)
            pred_tmp = torch.argmax(prob, dim=1)
            pred_label.append(pred_tmp.cpu().numpy())
            # pred_prob.append(prob.cpu().numpy())
            true_label.append(label.cpu().numpy())

    pred_label = np.concatenate(pred_label, 0)
    true_label = np.concatenate(true_label, 0)
    # best_threshold = args.threshold
    #
    # pred_label = [0 if x < best_threshold else 1 for x in pred_prob]

    precision = precision_score(true_label, pred_label, average="binary")
    recall = recall_score(true_label, pred_label, average="binary")
    f1 = f1_score(true_label, pred_label, average="binary")
    acc = accuracy_score(true_label, pred_label)

    result = {
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_acc": acc,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, test_dataset, mymodel, tokenizer, local_rank):
    # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=4)

    pred_label = []
    true_label = []
    mymodel.eval()
    for batch in test_dataloader:
        input_id, input_mask, label = [x.to(local_rank) for x in batch]
        with torch.no_grad():
            loss, prob = mymodel(input_id, label)
            pred_tmp = torch.argmax(prob, dim=1)
            pred_label.append(pred_tmp.cpu().numpy())
            # pred_prob.append(prob.cpu().numpy())
            true_label.append(label.cpu().numpy())

    pred_label = np.concatenate(pred_label, 0)
    true_label = np.concatenate(true_label, 0)
    # best_threshold = args.threshold
    #
    # pred_label = [0 if x < best_threshold else 1 for x in pred_prob]

    precision = precision_score(true_label, pred_label, average="binary")
    recall = recall_score(true_label, pred_label, average="binary")
    f1 = f1_score(true_label, pred_label, average="binary")
    acc = accuracy_score(true_label, pred_label)

    result = {
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_acc": acc,
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    # return result



def main(args):
    local_rank = args.local_rank
    world_size = torch.cuda.device_count()
    print(f"Start running basic DDP example on rank {local_rank}.")

    # setup(local_rank, world_size)

    dist.init_process_group(backend="nccl")
    # dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda", args.local_rank)
    args.device = device
    gpu_list = args.available_gpu.split(',')
    args.gpu_list = [int(x) for x in gpu_list]
    args.n_gpu = len(args.gpu_list)

    set_seed(args)

    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()

    model, tokenizer, config = build_model_tokenizer_config(args)
    mymodel = Model(model, args).to(local_rank)

    # if args.local_rank == 0:
    #     torch.distributed.barrier()

    task_nums = args.task_list.split(',')

    if args.do_train:
        if args.task_type == "svd":
            for num, task in enumerate(task_nums):
                task_id = int(task)

                logger.info(f"-----training {task_id} %s-----")
                args.cl_num = task_id
                train_dataset = load_svd_replay_data_by_tag(args, tokenizer, "train", tag=task_id)
                eval_dataset = load_svd_replay_data_by_tag(args, tokenizer, "eval", tag=task_id)
                logger.info(f"train_size for {task_id}: {str(len(train_dataset))}")

                if task_id != 0:
                    checkpoint_prefix = f'checkpoint-best-f1/model-{args.cl_num - 1}.bin'
                    output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
                    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
                    checkpoint = torch.load(output_dir, map_location=map_location)
                    mymodel.load_state_dict(checkpoint['model_state_dict'])

                train(args, train_dataset, eval_dataset, mymodel, tokenizer, local_rank)

        else:
            assert False, "Invalid Task Type."

    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-f1/model-{args.check_order}.bin'
        output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(output_dir, map_location=map_location)
        mymodel.load_state_dict(checkpoint['model_state_dict'])

        if args.task_type == "svd":
            for num, task in enumerate(range(args.check_order + 1)):
                task_id = int(task)

                logger.info(f"-----testing {task_id} %s-----")
                args.cl_num = task_id
                test_dataset = load_svd_replay_data_by_tag(args, tokenizer, "test", tag=task_id)
                logger.info(f"test_size for {task_id}: {str(len(test_dataset))}")
                test(args, test_dataset, mymodel, tokenizer, local_rank)

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