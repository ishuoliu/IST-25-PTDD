import json
import torch
import multiprocessing
from tqdm import tqdm
import random
from torch.utils.data import TensorDataset
# from openprompt.data_utils import InputExample


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 manual_feature=None,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.manual_feature = manual_feature
        self.url = url
        self.task = task
        self.sub_task = sub_task


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


def read_summarize_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if "idx" not in js:
                js["idx"] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def convert_examples_to_features(item):
    example_index, example, args, tokenizer, mode = item

    source_str = example.source
    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_input_tokens, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1

    target_str = example.target
    target_str = target_str.replace('</s>', '<unk>')
    target_ids = tokenizer.encode(target_str, max_length=args.max_output_tokens, padding='max_length', truncation=True)
    assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )


def load_sum_data_by_tag(args, tokenizer, mode, tag):
    id = tag[-1]
    lang = tag[: -1]

    if mode == "train" or mode == "test":
        filename = f"../datasets/sum/CodeSearchNet/{lang}/cl/{mode}_{id}.jsonl"
    elif mode == "eval":
        filename = f"../datasets/sum/CodeSearchNet/{lang}/cl/dev_{id}.jsonl"
    else:
        assert False, "Invalid Mode."
    examples = read_summarize_examples(filename, args.data_num)

    if args.sampled_num > 0 and mode == "train":
        examples = random.sample(examples, args.sampled_num)

    tuple_examples = [(idx, example, args, tokenizer, mode) for idx, example in enumerate(examples)]

    all_source_ids = []
    all_target_ids = []

    for example in tqdm(tuple_examples, total=len(tuple_examples)):
        output_feature = convert_examples_to_features(example)
        all_source_ids.append(output_feature.source_ids)
        all_target_ids.append(output_feature.target_ids)

    all_source_ids = torch.tensor(all_source_ids, dtype=torch.long)
    source_mask = all_source_ids.ne(tokenizer.pad_token_id)
    all_target_ids = torch.tensor(all_target_ids, dtype=torch.long)
    target_mask = all_target_ids.ne(tokenizer.pad_token_id)

    data = TensorDataset(all_source_ids, source_mask, all_target_ids, target_mask)
    return examples, data