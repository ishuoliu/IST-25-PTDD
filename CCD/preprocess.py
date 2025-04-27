import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 input_mask,
                 label,
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label = label


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
    # code1_ids = torch.tensor(code1_ids, dtype=torch.long)

    code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = args.max_input_tokens - len(code2_ids)
    code2_ids += [tokenizer.pad_token_id] * padding_length
    # code2_ids = torch.tensor(code2_ids, dtype=torch.long)

    # code1_mask = code1_ids.ne(tokenizer.pad_token_id)
    # code2_mask = code2_ids.ne(tokenizer.pad_token_id)

    source_tokens = code1_tokens + code2_tokens
    source_ids = code1_ids + code2_ids

    source_ids = torch.tensor(source_ids, dtype=torch.long)
    source_mask = source_ids.ne(tokenizer.pad_token_id)

    return InputFeatures(source_tokens, source_ids, source_mask, label)


class TextDataset(Dataset):
    def __init__(self, args, tokenizer, filename):
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
                self.count += self.examples[-1].label

        if "train" in filename:
            self.count = self.count
            self.count = (len(self.examples) - self.count) / (self.count)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (self.examples[i].input_ids,
                self.examples[i].input_mask,
                self.examples[i].label)


def load_clone_data_by_tag(args, tokenizer, mode, tag):
    id = tag

    if mode == "train" or mode == "test":
        filename = f"../datasets/clone/binary/{mode}_{id}.jsonl"
    elif mode == "eval":
        filename = f"../datasets/clone/binary/dev_{id}.jsonl"
    else:
        assert False, "Invalid Mode."

    data = TextDataset(args, tokenizer, filename)

    return data

