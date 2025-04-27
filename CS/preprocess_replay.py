import json
import torch
import multiprocessing
from tqdm import tqdm
import random
from copy import deepcopy
from torch.utils.data import TensorDataset
# from openprompt.data_utils import InputExample


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 original_source,
                 original_target
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.origin_source = original_source
        self.origin_target = original_target
        self.code = ''


class InputFeatures(object):
    """A single training/test features for an example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


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
                    original_source=js['code_tokens'],
                    original_target=js['docstring_tokens'],
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def convert_examples_to_features(examples, tokenizer, args, mode):
    codes = []
    target_nl = []

    for example_id, example in tqdm(enumerate(examples), total=len(examples)):
        codes.append(example.source.replace('</s>', '<unk>'))

        if mode == "test":
            target_nl.append("None")
        else:
            target_nl.append(example.target.replace('</s>', '<unk>'))

    encoded_codes = tokenizer(
        codes, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_input_tokens, return_tensors='pt')

    encoded_targets = tokenizer(
        target_nl, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_output_tokens, return_tensors='pt')

    return {'source_ids': encoded_codes['input_ids'], 'target_ids': encoded_targets['input_ids'],
            'source_mask': encoded_codes['attention_mask'], 'target_mask': encoded_targets['attention_mask']}


def load_sum_replay_data_by_tag(args, tokenizer, mode, tag):
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

    origin_examples = deepcopy(examples)

    if int(id) > 0:
        train_replay_examples = read_summarize_examples(args.train_examplar_path, args)
        if 'emr' in args.method and mode != 'test':
            examples.extend(train_replay_examples)

        example_features = convert_examples_to_features(examples, tokenizer, args, mode)
        all_source_ids = example_features['source_ids']
        all_source_mask = example_features['source_mask']
        all_target_ids = example_features['target_ids']
        all_target_mask = example_features['target_mask']

        data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
        return examples, origin_examples, train_replay_examples, data

    else:
        example_features = convert_examples_to_features(examples, tokenizer, args, mode)
        all_source_ids = example_features['source_ids']
        all_source_mask = example_features['source_mask']
        all_target_ids = example_features['target_ids']
        all_target_mask = example_features['target_mask']

        data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)

        return examples, origin_examples, data

