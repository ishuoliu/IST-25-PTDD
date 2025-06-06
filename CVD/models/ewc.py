import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss
import copy
import sys
import math
import numpy as np
import torch
import argparse
import random
import jsonlines
import heapq
import pandas as pd
from tqdm import tqdm, trange
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class TextDataset(Dataset):
    def __init__(self, data):
        self.examples = data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)


def calculate_coefficient(new_data, replay_data):
    vectorizer = CountVectorizer(min_df=15, ngram_range=(1, 1))
    transformer = TfidfTransformer()
    train_corpus = ['label_' + str(i.origin_target) + ' ' + i.origin_source for i in new_data + replay_data]
    train_tfidf = transformer.fit_transform(vectorizer.fit_transform(train_corpus)).toarray().tolist()
    train_tfidf = np.array(train_tfidf)

    replay_feature = train_tfidf[:len(new_data)].mean(axis=0)
    new_feature = train_tfidf[len(new_data):].mean(axis=0)

    a = replay_feature.dot(new_feature)
    b = np.linalg.norm(replay_feature)
    c = np.linalg.norm(new_feature)

    return a / (b * c)


def construct_exemplars_ours(model, args, train_exemplar, eval_exemplar, tokenizer, device, mode):
    task_id = args.cl_num
    train_replay_size = args.train_replay_size
    train_replay_path = args.train_examplar_path
    eval_replay_size = args.eval_replay_size
    eval_replay_path = args.eval_examplar_path

    if task_id > 0:
        old_train_replay_size = train_replay_size // (task_id + 1)
        old_eval_replay_size = eval_replay_size // (task_id + 1)
        new_train_replay_size = train_replay_size - task_id * old_train_replay_size
        new_eval_replay_size = eval_replay_size - task_id * old_eval_replay_size
    else:
        old_train_replay_size = 0
        old_eval_replay_size = 0
        new_train_replay_size = train_replay_size
        new_eval_replay_size = eval_replay_size

    old_replay_train_data = {}
    old_replay_valid_data = {}
    train_exemplars = []
    eval_exemplars = []
    if task_id > 0:
        with jsonlines.open(train_replay_path) as reader:
            # print(reader)
            for obj in reader:
                # print(obj)
                if obj['task_id'] not in old_replay_train_data:
                    old_replay_train_data[obj['task_id']] = [obj]
                else:
                    old_replay_train_data[obj['task_id']].append(obj)
        for key in old_replay_train_data:
            old_replay_train_data[key].sort(key=lambda x: x["score"], reverse=True)
        with jsonlines.open(eval_replay_path) as reader:
            for obj in reader:
                if obj['task_id'] not in old_replay_valid_data:
                    old_replay_valid_data[obj['task_id']] = [obj]
                else:
                    old_replay_valid_data[obj['task_id']].append(obj)
        for key in old_replay_valid_data:
            old_replay_valid_data[key].sort(key=lambda x: x["score"], reverse=True)

    if mode == 'random':
        # train
        random.shuffle(train_exemplar)
        new_train_exemplars = []
        for idx in range(new_train_replay_size):
            new_train_exemplars.append(
                {'func_before': train_exemplar[idx].origin_source, 'vul': train_exemplar[idx].origin_target, \
                 'task_id': str(task_id), 'score': 0})
        for idx in old_replay_train_data:
            train_exemplars.extend(old_replay_train_data[idx][:old_train_replay_size])
        train_exemplars.extend(new_train_exemplars)

        # eval
        random.shuffle(eval_exemplar)
        new_eval_exemplars = []
        for idx in range(new_eval_replay_size):
            new_eval_exemplars.append(
                {'func_before': eval_exemplar[idx].origin_source, 'vul': eval_exemplar[idx].origin_target, \
                 'task_id': str(task_id), 'score': 0})
        for idx in old_replay_valid_data:
            eval_exemplars.extend(old_replay_valid_data[idx][:old_eval_replay_size])
        eval_exemplars.extend(new_eval_exemplars)
    elif mode == 'ours':
        # train
        train_exemplar_class = {}
        eval_exemplar_class = {}
        train_sum = len(train_exemplar)
        eval_sum = len(eval_exemplar)
        new_train_replay_size_copy = new_train_replay_size
        new_eval_replay_size_copy = new_eval_replay_size
        new_eval_exemplars = []
        new_train_exemplars = []
        for i in train_exemplar:
            if i.origin_target not in train_exemplar_class:
                train_exemplar_class[i.origin_target] = [i]
            else:
                train_exemplar_class[i.origin_target].append(i)
        for i in eval_exemplar:
            if i.origin_target not in eval_exemplar_class:
                eval_exemplar_class[i.origin_target] = [i]
            else:
                eval_exemplar_class[i.origin_target].append(i)
        # print(train_exemplar_class.keys())
        # print(eval_exemplar_class.keys())

        for clas in train_exemplar_class:
            # train
            train_exemplar = train_exemplar_class[clas]
            new_train_replay_size = math.ceil(new_train_replay_size_copy * len(train_exemplar) // train_sum)

            vectorizer = CountVectorizer(min_df=15, ngram_range=(1, 1))
            transformer = TfidfTransformer()
            train_corpus = [i.origin_source for i in train_exemplar]
            train_tfidf = transformer.fit_transform(vectorizer.fit_transform(train_corpus)).toarray().tolist()
            cluster_number = args.k
            clf = KMeans(n_clusters=cluster_number, init='k-means++')
            train_label = clf.fit_predict(train_tfidf)

            train_dataset = TextDataset(train_exemplar)
            train_sampler = SequentialSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32, num_workers=8,
                                          pin_memory=True)
            model.eval()
            score = []
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            for step, batch in enumerate(bar):
                inputs = batch[0].to(args.device)
                label = batch[1].to(args.device)
                with torch.no_grad():
                    lm_loss, logit = model(inputs, label, relay=True)
                    score.extend(list(lm_loss.cpu().numpy()))
            score_max = max(score)
            score = [(score_max - i) for i in score]
            score_sum = sum(score)
            score = [i / score_sum for i in score]
            cluster_replay_number = []
            for i in range(cluster_number):
                cluster_replay_number.append(
                    math.ceil(train_label.tolist().count(i) * new_train_replay_size // len(train_label)))
            class_score = {}
            class_pos = {}
            for idx in range(len(train_label)):
                i = train_label[idx]
                j = score[idx]
                if i not in class_score:
                    class_score[i] = [j]
                    class_pos[i] = [idx]
                else:
                    class_score[i].append(j)
                    class_pos[i].append(idx)
            train_topk = []
            for i in range(cluster_number):
                class_topk = heapq.nlargest(min(args.mu * cluster_replay_number[i], len(class_score[i])),
                                            range(len(class_score[i])), class_score[i].__getitem__)
                class_topk = np.random.choice(class_topk, cluster_replay_number[i], replace=False)
                train_topk.extend([class_pos[i][j] for j in class_topk])
            final_score = score

            for idx in train_topk:
                new_train_exemplars.append(
                    {'func_before': train_exemplar[idx].origin_source, 'vul': train_exemplar[idx].origin_target, \
                     'task_id': str(task_id), 'score': final_score[idx]})

            # eval
            eval_exemplar = eval_exemplar_class[clas]
            new_eval_replay_size = math.ceil(new_eval_replay_size_copy * len(eval_exemplar) // eval_sum)

            vectorizer = CountVectorizer(min_df=10, ngram_range=(1, 1))
            transformer = TfidfTransformer()
            eval_corpus = [i.origin_source for i in eval_exemplar]
            eval_tfidf = transformer.fit_transform(vectorizer.fit_transform(eval_corpus)).toarray().tolist()
            clf = KMeans(n_clusters=cluster_number, init='k-means++')
            eval_label = clf.fit_predict(eval_tfidf)

            eval_dataset = TextDataset(eval_exemplar)
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=32, num_workers=8,
                                         pin_memory=True)
            model.eval()
            score = []
            for batch in eval_dataloader:
                inputs = batch[0].to(args.device)
                label = batch[1].to(args.device)
                with torch.no_grad():
                    lm_loss, logit = model(inputs, label, relay=True)
                    score.extend(list(lm_loss.cpu().numpy()))
            score_max = max(score)
            score = [(score_max - i) for i in score]
            score_sum = sum(score)
            score = [i / score_sum for i in score]
            cluster_replay_number = []
            for i in range(cluster_number):
                cluster_replay_number.append(
                    math.ceil(eval_label.tolist().count(i) * new_eval_replay_size // len(eval_label)))
            class_score = {}
            class_pos = {}
            for idx in range(len(eval_label)):
                i = eval_label[idx]
                j = score[idx]
                if i not in class_score:
                    class_score[i] = [j]
                    class_pos[i] = [idx]
                else:
                    class_score[i].append(j)
                    class_pos[i].append(idx)
            eval_topk = []
            for i in range(cluster_number):
                class_topk = heapq.nlargest(min(args.mu * cluster_replay_number[i], len(class_score[i])),
                                            range(len(class_score[i])), class_score[i].__getitem__)
                class_topk = np.random.choice(class_topk, cluster_replay_number[i], replace=False)
                eval_topk.extend([class_pos[i][j] for j in class_topk])
            final_score = score

            for idx in eval_topk:
                new_eval_exemplars.append(
                    {'func_before': eval_exemplar[idx].origin_source, 'vul': eval_exemplar[idx].origin_target, \
                     'task_id': str(task_id), 'score': final_score[idx]})

        for idx in old_replay_train_data:
            train_exemplars.extend(old_replay_train_data[idx][:old_train_replay_size])
        train_exemplars.extend(new_train_exemplars)
        for idx in old_replay_valid_data:
            eval_exemplars.extend(old_replay_valid_data[idx][:old_eval_replay_size])
        eval_exemplars.extend(new_eval_exemplars)

    with jsonlines.open(train_replay_path, mode='w') as f:
        f.write_all(train_exemplars)
    with jsonlines.open(eval_replay_path, mode='w') as f:
        f.write_all(eval_exemplars)


class Model(nn.Module):
    def __init__(self, basemodel, args):
        super(Model, self).__init__()
        self.basemodel = basemodel
        self.args = args
        self.weight = 1

    def forward(self, input_ids=None, labels=None, relay=None):
        # outputs = self.basemodel(input_ids, attention_mask=input_ids.ne(1))
        # prob = torch.nn.functional.softmax(outputs[0])
        #
        # print(prob)
        #
        # if relay is True:
        #     cross_entropy_loss = nn.CrossEntropyLoss()
        #     loss = cross_entropy_loss(outputs[0], labels)
        #     return loss, prob
        # if labels is not None:
        #     cross_entropy_loss = nn.CrossEntropyLoss()
        #     loss = cross_entropy_loss(outputs[0], labels)
        #     return loss, prob
        # else:
        #     return prob


        outputs = self.basemodel(input_ids, attention_mask=input_ids.ne(1))[0]
        logits = outputs
        # prob = torch.sigmoid(logits)
        prob = torch.nn.functional.softmax(logits)
        # print(prob)

        if relay is True:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * (1-labels) + torch.log((1 - prob)[:, 0] + 1e-10) * labels
            return -loss, prob
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * (1-labels) + torch.log((1 - prob)[:, 0] + 1e-10) * labels
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


class EWC(object):
    def __init__(self, model, dataset, device, length):

        self.model = model
        # the data we use to compute fisher information of ewc (old_exemplars)
        self.dataset = dataset
        self.device = device
        self.length = length

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}  # previous parameters
        self._precision_matrices = self._diag_fisher()  # approximated diagnal fisher information matrix

        for n, p in copy.deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):

        self.model.train()
        precision_matrices = {}
        for n, p in copy.deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        for step, batch in enumerate(self.dataset):
            self.model.zero_grad()

            inputs = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            self.model.train()
            loss, logits = self.model(inputs, labels)

            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                precision_matrices[n].data += p.grad.data ** 2 / self.length

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        self.model.zero_grad()
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()

        return loss
