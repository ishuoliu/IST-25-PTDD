import argparse
import random
import numpy as np
import torch
import os
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_type", type=str, default="svd")
    # parser.add_argument("--lang", type=str, default="ruby")
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_all_param", action='store_true')

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="beam size for beam search")
    parser.add_argument("--warmup_steps", type=int, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=5,
                        help='patience for early stop')
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--pretrained_model", type=str, default="codebert")
    parser.add_argument("--method", type=str, default="finetune")
    parser.add_argument("--available_gpu", type=str, default="3")
    parser.add_argument("--max_input_tokens", type=int, default=64)
    # parser.add_argument("--max_output_tokens", type=int, default=32)

    parser.add_argument("--prompt_token_num", type=int, default=10)
    parser.add_argument("--mlp_type", type=str, default="mlp1")
    parser.add_argument("--mlp_bottleneck_size", type=int, default=800)
    parser.add_argument("--data_num", type=int, default=-1, help="DATA_NUM == -1 means all data")
    parser.add_argument("--sampled_num", type=int, default=-1)

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument('--local_rank', type=int, default=-1)

    parser.add_argument("--task_list", type=str, default="0,1,2,3,4")
    parser.add_argument("--check_order", type=int, default=4)

    parser.add_argument("--ewc_weight", type=int, default=1, help="The weight of EWC penalty.")
    parser.add_argument("--train_examplar_path", type=str, default="", help="Path of training examplars.")
    parser.add_argument("--eval_examplar_path", type=str, default="", help="Path of valid examplars.")
    parser.add_argument("--train_replay_size", type=int, default=100, help="The size of replayed training examplars.")
    parser.add_argument("--eval_replay_size", type=int, default=25, help="The size of replayed valid examplars.")
    parser.add_argument("--k", default=5, type=int, help="Hyperparameter")
    parser.add_argument("--mu", default=5, type=int, help="Hyperparameter")
    parser.add_argument("--lam", default=0, type=float, help="Hyperparameter")
    parser.add_argument("--ewc_mode", default="whole", type=str, help="whole, only_prompt, only_backbone")

    args = parser.parse_args()
    return args


def build_model_tokenizer_config(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.method in ["prefix", "prompt", "lora", "ewc", "emr", "finetune", "pp_old", "prompt_ewc", "pp_new"]:
        model_classes = {
            "codebert": (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig, "microsoft/codebert-base"),
            "graphcodebert": (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig, "../pretrained/graphcodebert"),
            "unixcoder": (RobertaModel, RobertaTokenizer, RobertaConfig, "../pretrained/unixcoder")
        }

    elif args.method in []:
        model_classes = {
            "codebert": (RobertaModel, RobertaTokenizer, RobertaConfig, "microsoft/codebert-base"),
            "graphcodebert": (RobertaModel, RobertaTokenizer, RobertaConfig, "../pretrained/graphcodebert"),
            "unixcoder": (RobertaModel, RobertaTokenizer, RobertaConfig, "../pretrained/unixcoder")
        }

    else:
        assert False, "Invalid Method Name!"

    model_class, tokenizer_class, config_class, huggingface_name = model_classes[args.pretrained_model]
    args.huggingface_name = huggingface_name

    config = config_class.from_pretrained(huggingface_name)

    if args.pretrained_model in ["unixcoder"]:
        config.is_decoder = True
    tokenizer = tokenizer_class.from_pretrained(huggingface_name)

    model = model_class.from_pretrained(huggingface_name, config=config)

    return model, tokenizer, config