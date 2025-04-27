import json
import torch
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 input_mask,
                 label,
                 origin_source1,
                 origin_source2,
                 origin_target

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label = label
        self.origin_source1 = origin_source1
        self.origin_source2 = origin_source2
        self.origin_target = origin_target


def convert_examples_to_features(js, tokenizer, args):
    code1 = ' '.join(js['code1'].split())
    code1_tokens = tokenizer.tokenize(code1)
    code2 = ' '.join(js['code2'].split())
    code2_tokens = tokenizer.tokenize(code2)
    label = int(js['label'])

    code1_tokens = code1_tokens[:args.max_input_tokens - 2]
    code1_tokens = [tokenizer.cls_token] + code1_tokens + [tokenizer.sep_token]
    code2_tokens = code2_tokens[:args.max_input_tokens - 2]
    code2_tokens = [tokenizer.cls_token] + code2_tokens + [tokenizer.sep_token]

    code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = args.max_input_tokens - len(code1_ids)
    code1_ids += [tokenizer.pad_token_id] * padding_length

    code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = args.max_input_tokens - len(code2_ids)
    code2_ids += [tokenizer.pad_token_id] * padding_length

    source_tokens = code1_tokens + code2_tokens
    source_ids = code1_ids + code2_ids

    source_ids = torch.tensor(source_ids, dtype=torch.long)
    source_mask = source_ids.ne(tokenizer.pad_token_id)
    return InputFeatures(source_tokens, source_ids, source_mask, label, js['code1'], js['code2'], js['label'])


class TextDataset(Dataset):
    def __init__(self, args, tokenizer, filename, extra=None):
        self.examples = []
        self.count = 0
        self.line_num = 0

        with open(filename) as f:
            for line in f:
                self.line_num += 1

        with open(filename) as f:
            for line in tqdm(f, total=self.line_num):
                line = line.strip()
                js = json.loads(line)
                self.examples.append(convert_examples_to_features(js, tokenizer, args))

        self.origin_data = deepcopy(self.examples)

        if extra is not None:
            self.replay_examples = []
            if extra == 'train':
                examplar_path = args.train_examplar_path
            else:
                examplar_path = args.eval_examplar_path

            with open(examplar_path) as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    self.replay_examples.append(convert_examples_to_features(js, tokenizer, args))
                if 'emr' in args.method and extra != "test":
                    self.examples.extend(self.replay_examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (self.examples[i].input_ids,
                self.examples[i].input_mask,
                self.examples[i].label)


def load_clone_replay_data_by_tag(args, tokenizer, mode, tag):
    id = tag

    if mode == "train" or mode == "test":
        filename = f"../datasets/clone/binary/{mode}_{id}.jsonl"
    elif mode == "eval":
        filename = f"../datasets/clone/binary/dev_{id}.jsonl"
    else:
        assert False, "Invalid Mode."

    if int(id) > 0:
        data = TextDataset(args, tokenizer, filename, mode)
    else:
        data = TextDataset(args, tokenizer, filename)

    return data