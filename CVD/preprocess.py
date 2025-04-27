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
    label = int(js['vul'])

    code = ' '.join(js['func_before'].split())
    code_tokens = tokenizer.tokenize(code)[:args.max_input_tokens - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.max_input_tokens - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

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


def load_svd_data_by_tag(args, tokenizer, mode, tag):
    id = tag

    if mode == "train" or mode == "test":
        filename = f"../datasets/svd/defect/{mode}_{id}.jsonl"
    elif mode == "eval":
        filename = f"../datasets/svd/defect/dev_{id}.jsonl"
    else:
        assert False, "Invalid Mode."

    data = TextDataset(args, tokenizer, filename)

    return data

