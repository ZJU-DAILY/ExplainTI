import itertools
import logging
import os

import numpy as np
import json

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

lm_path = {
    'bert-base-uncased': 'bert-base-uncased',
    'bert-base-cased': '.bert-base-cased',
    'bert-large-cased': 'bert-large-cased',
    'bert-base-multilingual-uncased': 'bert-base-multilingual-uncased',
    'roberta-base': 'roberta-base',
    'bert-large-uncased': 'bert-large-uncased',
    'roberta-large': 'roberta-large'
}


class TypeDataset(Dataset):
    def __init__(self, path, mode, lm='bert-base-uncased', column_length=64, multi_columns=True, window_size=1):
        self.path = path
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(lm_path[lm])
        self.column_length = column_length
        self.multi_columns = multi_columns
        self.lm = lm
        self.window_size = window_size
        self.windows = []
        self.sentences = []
        self.original_sentences = []
        self.labels = []
        self.original_labels = []
        self.type_vocab = {}
        self.num_type = 0
        if 'WikiTable' in self.path:
            self.read_file()
        else:
            self.read_file_git()
        if self.multi_columns:
            graph_path = os.path.join(self.path, self.mode + '.graph_col_type.npy')
            self.column_graph = np.load(graph_path)
            self.column_graph = self.column_graph.tolist()
            logging.info(f"neighbors size: {np.shape(self.column_graph)[1]}")
        self.len = len(self.sentences)

    def read_file(self):
        type_vocab_path = os.path.join(self.path, 'type_vocab.txt')
        with open(type_vocab_path) as type_file:
            for line in type_file.readlines():
                type_id, type_name = line.strip().split('\t')
                self.type_vocab[type_name] = int(type_id)
        self.num_type = len(self.type_vocab)

        table_path = os.path.join(self.path, self.mode + '.table_col_type.json')
        with open(table_path) as table_file:
            table = json.load(table_file)
            for t in tqdm(table):
                title = t[1].strip()
                columns = t[6]
                labels = t[7]
                for i in range(len(labels)):
                    original_labels = []
                    for j in range(len(labels[i])):
                        original_labels.append(labels[i][j])
                        labels[i][j] = self.type_vocab[labels[i][j]]
                    self.labels.append(labels[i])
                    self.original_labels.append(original_labels)

                for i, column in enumerate(columns):
                    header = t[5][i].strip()
                    if self.multi_columns:
                        sentence = 'Title ' + title + ' Header ' + header + ' Cell '
                    else:
                        sentence = 'Header ' + header + ' Cell '
                    for cell in column:
                        sentence = sentence + cell[1][1].strip() + ' '
                    self.original_sentences.append(sentence)
                    sentence = self.tokenizer(sentence, truncation=True, max_length=self.column_length)
                    self.sentences.append(sentence['input_ids'])
                    real_len = len(sentence["input_ids"])
                    # generate windows
                    st = 1
                    ed = real_len - 2
                    idx = None
                    for j in range(self.window_size):
                        # if ed-st+1 < window_size
                        if (ed - st + 1) < self.window_size:
                            tmp = torch.unsqueeze(
                                torch.arange(min(st + j, real_len - 2), min(st + j, real_len - 2) + 1), dim=1)
                        else:
                            tmp = torch.unsqueeze(torch.arange(st + j, ed + 2 - self.window_size + j), dim=1)
                        if j == 0:
                            idx = tmp
                        else:
                            idx = torch.cat((idx, tmp), dim=1)
                    self.windows.append(idx.numpy().tolist())

    def read_file_git(self):
        type_vocab_path = os.path.join(self.path, 'type_vocab.txt')
        with open(type_vocab_path) as type_file:
            for line in type_file.readlines():
                type_id, type_name = line.strip().split('\t')
                self.type_vocab[type_name] = int(type_id)
        self.num_type = len(self.type_vocab)

        table_path = os.path.join(self.path, self.mode + '.table_col_type.json')
        with open(table_path) as table_file:
            table = json.load(table_file)
            for t in tqdm(table):
                title = t['title']
                columns = t['columns']
                labels = t['labels']
                for i in range(len(labels)):
                    self.labels.append(self.type_vocab[labels[i]])
                    self.original_labels.append(labels[i])

                for i, column in enumerate(columns):
                    header = t['headers'][i].strip()
                    if self.multi_columns:
                        sentence = 'Title ' + title + ' Header ' + header + ' Cell '
                    else:
                        sentence = 'Header ' + header + ' Cell '
                    for cell in column:
                        sentence = sentence + str(cell).strip() + ' '
                    self.original_sentences.append(sentence)
                    sentence = self.tokenizer(sentence, truncation=True, max_length=self.column_length)
                    self.sentences.append(sentence['input_ids'])
                    real_len = len(sentence["input_ids"])
                    # generate windows
                    st = 1
                    ed = real_len - 2
                    idx = None
                    for j in range(self.window_size):
                        # if ed-st+1 < window_size
                        if (ed - st + 1) < self.window_size:
                            tmp = torch.unsqueeze(
                                torch.arange(min(st + j, real_len - 2), min(st + j, real_len - 2) + 1), dim=1)
                        else:
                            tmp = torch.unsqueeze(torch.arange(st + j, ed + 2 - self.window_size + j), dim=1)
                        if j == 0:
                            idx = tmp
                        else:
                            idx = torch.cat((idx, tmp), dim=1)
                    self.windows.append(idx.numpy().tolist())

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]
        length = len(sentence)
        original_sentence = self.original_sentences[index]
        original_label = self.original_labels[index]
        windows = self.windows[index]
        if self.multi_columns:
            neighbors = self.column_graph[index]
        else:
            neighbors = None
        return sentence, label, length, original_sentence, original_label, windows, neighbors

    @staticmethod
    def pad(batch):
        """
        Pads to the longest sample.
        """
        f = lambda x: [sample[x] for sample in batch]

        lengths = f(2)
        max_len = np.array(lengths).max()

        sentences = f(0)
        labels = f(1)
        original_sentence = f(3)
        original_label = f(4)
        windows = f(5)
        neighbors = f(6)

        for sentence in sentences:
            sentence += [0] * (max_len - len(sentence))

        # pad windows in dimension 1
        for win in windows:
            win += [[0] * len(win[0]) for i in range(max_len - 1 - len(win[0]) - len(win))]
        return sentences, labels, original_sentence, original_label, windows, neighbors


class RelationDataset(Dataset):
    def __init__(self, path, mode, lm='bert-base-uncased', column_length=64, multi_columns=True, window_size=1):
        self.path = path
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(lm_path[lm])
        self.column_length = column_length
        self.multi_columns = multi_columns
        self.lm = lm
        self.window_size = window_size
        self.windows = []
        self.sentence_pairs = []
        self.original_sentence_pairs = []
        self.labels = []
        self.original_labels = []
        self.relation_vocab = {}
        self.num_relation = 0
        self.read_file()
        if self.multi_columns:
            graph_path = os.path.join(self.path, self.mode + '.graph_rel_extraction.npy')
            self.column_graph = np.load(graph_path)
            self.column_graph = self.column_graph.tolist()
            logging.info(f"neighbors size: {np.shape(self.column_graph)[1]}")
        self.len = len(self.sentence_pairs)

    def read_file(self):
        relation_vocab_path = os.path.join(self.path, 'relation_vocab.txt')
        with open(relation_vocab_path) as relation_file:
            for line in relation_file.readlines():
                relation_id, relation_name = line.strip().split('\t')
                self.relation_vocab[relation_name] = int(relation_id)
        self.num_relation = len(self.relation_vocab)

        table_path = os.path.join(self.path, self.mode + '.table_rel_extraction.json')
        with open(table_path) as table_file:
            table = json.load(table_file)
            for t in tqdm(table):
                columns = t[6]
                labels = t[7]
                for i in range(len(labels)):
                    original_labels = []
                    for j in range(len(labels[i])):
                        original_labels.append(labels[i][j])
                        labels[i][j] = self.relation_vocab[labels[i][j]]
                    self.original_labels.append(original_labels)
                    self.labels.append(labels[i])

                left_sentence = ''
                for i, column in enumerate(columns):
                    if i == 0:
                        if self.multi_columns:
                            left_sentence = 'Title ' + t[1].strip() + ' Header ' + t[5][i].strip() + ' Cell '
                        else:
                            left_sentence = 'Header ' + t[5][i].strip() + ' Cell '
                        for cell in column:
                            left_sentence = left_sentence + cell[1][1].strip() + ' '
                    else:
                        if self.multi_columns:
                            right_sentence = 'Title ' + t[1].strip() + ' Header ' + t[5][i].strip() + ' Cell '
                        else:
                            right_sentence = 'Header ' + t[5][i].strip() + ' Cell '
                        for cell in column:
                            right_sentence = right_sentence + cell[1][1].strip() + ' '

                        self.original_sentence_pairs.append(f"{left_sentence} [SEP] {right_sentence}")
                        sentence_pair = self.tokenizer(left_sentence, right_sentence, truncation=True,
                                                       max_length=self.column_length)
                        self.sentence_pairs.append(sentence_pair['input_ids'])
                        # generate windows
                        real_len = len(sentence_pair["input_ids"])
                        a_real_sen = self.tokenizer(left_sentence, truncation=False)
                        a_real_len = len(a_real_sen["input_ids"]) - 2
                        b_real_sen = self.tokenizer(right_sentence, truncation=False)
                        b_real_len = len(b_real_sen["input_ids"]) - 2
                        a_st = 1
                        b_ed = real_len - 2

                        if "roberta" in self.lm:
                            half = self.column_length // 2 - 2
                            if a_real_len >= half and b_real_len >= half:
                                a_ed = half
                            elif a_real_len >= half > b_real_len and (a_real_len + b_real_len) >= (half + half):
                                a_ed = self.column_length - 1 - b_real_len - 3
                            elif a_real_len >= half > b_real_len and (a_real_len + b_real_len) < (half + half):
                                a_ed = a_real_len
                            elif a_real_len < half <= b_real_len and (a_real_len + b_real_len) >= (half + half):
                                a_ed = a_real_len
                            elif a_real_len < half <= b_real_len and (a_real_len + b_real_len) < (half + half):
                                a_ed = a_real_len
                            elif a_real_len < half and b_real_len < half:
                                a_ed = a_real_len
                            b_st = a_ed + 3

                        elif "bert" in self.lm:
                            big_half = self.column_length // 2 - 1
                            small_half = self.column_length // 2 - 2
                            if big_half <= a_real_len <= b_real_len and b_real_len >= big_half:
                                a_ed = small_half
                            elif a_real_len >= big_half and big_half <= b_real_len < a_real_len:
                                a_ed = big_half
                            elif a_real_len == small_half and b_real_len >= big_half:
                                a_ed = small_half
                            elif a_real_len >= big_half and b_real_len == small_half:
                                a_ed = big_half
                            elif a_real_len == small_half and b_real_len == small_half:
                                a_ed = small_half
                            elif a_real_len == small_half and b_real_len < small_half:
                                a_ed = small_half
                            elif a_real_len < small_half and b_real_len == small_half:
                                a_ed = a_real_len
                            elif a_real_len < small_half and b_real_len >= big_half and (a_real_len + b_real_len) >= (
                                    small_half + big_half):
                                a_ed = a_real_len
                            elif a_real_len < small_half and b_real_len >= big_half and (a_real_len + b_real_len) < (
                                    small_half + big_half):
                                a_ed = a_real_len
                            elif a_real_len >= big_half and b_real_len < small_half and (a_real_len + b_real_len) >= (
                                    small_half + big_half):
                                a_ed = self.column_length - 1 - b_real_len - 2
                            elif a_real_len >= big_half and b_real_len < small_half and (a_real_len + b_real_len) < (
                                    small_half + big_half):
                                a_ed = a_real_len
                            elif a_real_len < small_half and b_real_len < small_half:
                                a_ed = a_real_len
                            b_st = a_ed + 2
                        a_idx = None
                        b_idx = None
                        for j in range(self.window_size):
                            if (a_ed - a_st + 1) < self.window_size:
                                a_tmp = torch.unsqueeze(
                                    torch.arange(min(a_st + j, a_real_len), min(a_st + j, a_real_len) + 1), dim=1)
                            else:
                                a_tmp = torch.unsqueeze(torch.arange(a_st + j, a_ed + 2 - self.window_size + j), dim=1)
                            if (b_ed - b_st + 1) < self.window_size:
                                b_tmp = torch.unsqueeze(
                                    torch.arange(min(b_st + j, b_real_len), min(b_st + j, b_real_len) + 1), dim=1)
                            else:
                                b_tmp = torch.unsqueeze(torch.arange(b_st + j, b_ed + 2 - self.window_size + j), dim=1)
                            if j == 0:
                                a_idx = a_tmp
                                b_idx = b_tmp
                            else:
                                a_idx = torch.cat((a_idx, a_tmp), dim=1)
                                b_idx = torch.cat((b_idx, b_tmp), dim=1)
                        a_idx = a_idx.numpy().tolist()
                        b_idx = b_idx.numpy().tolist()
                        idx = torch.tensor(list(map(lambda x: x[0] + x[1], itertools.product(a_idx, b_idx))))
                        self.windows.append(idx.numpy().tolist())

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sentence_pair = self.sentence_pairs[index]
        label = self.labels[index]
        length = len(sentence_pair)
        original_sentence_pair = self.original_sentence_pairs[index]
        original_labels = self.original_labels[index]
        window = self.windows[index]
        if self.multi_columns:
            neighbors = self.column_graph[index]
        else:
            neighbors = None
        return sentence_pair, label, length, original_sentence_pair, original_labels, window, neighbors

    @staticmethod
    def pad(batch):
        """
        Pads to the longest sample.
        """
        f = lambda x: [sample[x] for sample in batch]

        lengths = f(2)
        max_len = np.array(lengths).max()

        sentence_pairs = f(0)
        labels = f(1)
        original_sentence_pair = f(3)
        original_label = f(4)
        window = f(5)
        neighbors = f(6)

        for sentence_pair in sentence_pairs:
            sentence_pair += [0] * (max_len - len(sentence_pair))

        # pad windows in dimension 1
        left_len = int(max_len / 2 - 1)
        right_len = int(max_len / 2 - 2)
        window_size = int(len(window[0][0]) / 2)
        pair_max_len = (left_len + 1 - window_size) * (right_len + 1 - window_size)
        for win in window:
            win += [[0] * len(win[0]) for i in range(pair_max_len - len(win))]
        return sentence_pairs, labels, original_sentence_pair, original_label, window, neighbors


class LocalTypeDataset(Dataset):
    def __init__(self,path:str,lm="roberta-base",column_length=64):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_path[lm])
        self.path=path
        self.column_length=column_length
        self.sentences=[]
        self.labels=[]
        self.type_vocab=dict()
        type_vocab_path = os.path.join(self.path, 'type_vocab.txt')
        with open(type_vocab_path) as type_file:
            for line in type_file.readlines():
                type_id, type_name = line.strip().split('\t')
                self.type_vocab[type_name] = int(type_id)
        self.num_type = len(self.type_vocab)

    def update_samples():
        pass



    def read_file(self):
        table_path = os.path.join(self.path, self.mode + '.table_col_type.json')
        with open(table_path) as table_file:
            table = json.load(table_file)
            for t in tqdm(table):
                title = t[1].strip()
                columns = t[6]
                labels = t[7]
                for i in range(len(labels)):
                    original_labels = []
                    for j in range(len(labels[i])):
                        original_labels.append(labels[i][j])
                        labels[i][j] = self.type_vocab[labels[i][j]]
                    self.labels.append(labels[i])
                    self.original_labels.append(original_labels)

                for i, column in enumerate(columns):
                    header = t[5][i].strip()
                    if self.multi_columns:
                        sentence = 'Title ' + title + ' Header ' + header + ' Cell '
                    else:
                        sentence = 'Header ' + header + ' Cell '
                    for cell in column:
                        sentence = sentence + cell[1][1].strip() + ' '
                    self.original_sentences.append(sentence)
                    sentence = self.tokenizer(sentence, truncation=True, max_length=self.column_length)
                    self.sentences.append(sentence['input_ids'])
                    real_len = len(sentence["input_ids"])
                    # generate windows
                    st = 1
                    ed = real_len - 2
                    idx = None
                    for j in range(self.window_size):
                        # if ed-st+1 < window_size
                        if (ed - st + 1) < self.window_size:
                            tmp = torch.unsqueeze(
                                torch.arange(min(st + j, real_len - 2), min(st + j, real_len - 2) + 1), dim=1)
                        else:
                            tmp = torch.unsqueeze(torch.arange(st + j, ed + 2 - self.window_size + j), dim=1)
                        if j == 0:
                            idx = tmp
                        else:
                            idx = torch.cat((idx, tmp), dim=1)
                    self.windows.append(idx.numpy().tolist())

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]
        length = len(sentence)
        original_sentence = self.original_sentences[index]
        original_label = self.original_labels[index]
        windows = self.windows[index]
        if self.multi_columns:
            neighbors = self.column_graph[index]
        else:
            neighbors = None
        return sentence, label, length, original_sentence, original_label, windows, neighbors

    @staticmethod
    def pad(batch):
        """
        Pads to the longest sample.
        """
        f = lambda x: [sample[x] for sample in batch]

        lengths = f(2)
        max_len = np.array(lengths).max()

        sentences = f(0)
        labels = f(1)
        original_sentence = f(3)
        original_label = f(4)
        windows = f(5)
        neighbors = f(6)

        for sentence in sentences:
            sentence += [0] * (max_len - len(sentence))

        # pad windows in dimension 1
        for win in windows:
            win += [[0] * len(win[0]) for i in range(max_len - 1 - len(win[0]) - len(win))]
        return sentences, labels, original_sentence, original_label, windows, neighbors


class RelationDataset(Dataset):
    def __init__(self, path, mode, lm='bert-base-uncased', column_length=64, multi_columns=True, window_size=1):
        self.path = path
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(lm_path[lm])
        self.column_length = column_length
        self.multi_columns = multi_columns
        self.lm = lm
        self.window_size = window_size
        self.windows = []
        self.sentence_pairs = []
        self.original_sentence_pairs = []
        self.labels = []
        self.original_labels = []
        self.relation_vocab = {}
        self.num_relation = 0
        self.read_file()
        if self.multi_columns:
            graph_path = os.path.join(self.path, self.mode + '.graph_rel_extraction.npy')
            self.column_graph = np.load(graph_path)
            self.column_graph = self.column_graph.tolist()
            logging.info(f"neighbors size: {np.shape(self.column_graph)[1]}")
        self.len = len(self.sentence_pairs)

    def read_file(self):
        relation_vocab_path = os.path.join(self.path, 'relation_vocab.txt')
        with open(relation_vocab_path) as relation_file:
            for line in relation_file.readlines():
                relation_id, relation_name = line.strip().split('\t')
                self.relation_vocab[relation_name] = int(relation_id)
        self.num_relation = len(self.relation_vocab)

        table_path = os.path.join(self.path, self.mode + '.table_rel_extraction.json')
        with open(table_path) as table_file:
            table = json.load(table_file)
            for t in tqdm(table):
                columns = t[6]
                labels = t[7]
                for i in range(len(labels)):
                    original_labels = []
                    for j in range(len(labels[i])):
                        original_labels.append(labels[i][j])
                        labels[i][j] = self.relation_vocab[labels[i][j]]
                    self.original_labels.append(original_labels)
                    self.labels.append(labels[i])

                left_sentence = ''
                for i, column in enumerate(columns):
                    if i == 0:
                        if self.multi_columns:
                            left_sentence = 'Title ' + t[1].strip() + ' Header ' + t[5][i].strip() + ' Cell '
                        else:
                            left_sentence = 'Header ' + t[5][i].strip() + ' Cell '
                        for cell in column:
                            left_sentence = left_sentence + cell[1][1].strip() + ' '
                    else:
                        if self.multi_columns:
                            right_sentence = 'Title ' + t[1].strip() + ' Header ' + t[5][i].strip() + ' Cell '
                        else:
                            right_sentence = 'Header ' + t[5][i].strip() + ' Cell '
                        for cell in column:
                            right_sentence = right_sentence + cell[1][1].strip() + ' '

                        self.original_sentence_pairs.append(f"{left_sentence} [SEP] {right_sentence}")
                        sentence_pair = self.tokenizer(left_sentence, right_sentence, truncation=True,
                                                       max_length=self.column_length)
                        self.sentence_pairs.append(sentence_pair['input_ids'])
                        # generate windows
                        real_len = len(sentence_pair["input_ids"])
                        a_real_sen = self.tokenizer(left_sentence, truncation=False)
                        a_real_len = len(a_real_sen["input_ids"]) - 2
                        b_real_sen = self.tokenizer(right_sentence, truncation=False)
                        b_real_len = len(b_real_sen["input_ids"]) - 2
                        a_st = 1
                        b_ed = real_len - 2

                        if "roberta" in self.lm:
                            half = self.column_length // 2 - 2
                            if a_real_len >= half and b_real_len >= half:
                                a_ed = half
                            elif a_real_len >= half > b_real_len and (a_real_len + b_real_len) >= (half + half):
                                a_ed = self.column_length - 1 - b_real_len - 3
                            elif a_real_len >= half > b_real_len and (a_real_len + b_real_len) < (half + half):
                                a_ed = a_real_len
                            elif a_real_len < half <= b_real_len and (a_real_len + b_real_len) >= (half + half):
                                a_ed = a_real_len
                            elif a_real_len < half <= b_real_len and (a_real_len + b_real_len) < (half + half):
                                a_ed = a_real_len
                            elif a_real_len < half and b_real_len < half:
                                a_ed = a_real_len
                            b_st = a_ed + 3

                        elif "bert" in self.lm:
                            big_half = self.column_length // 2 - 1
                            small_half = self.column_length // 2 - 2
                            if big_half <= a_real_len <= b_real_len and b_real_len >= big_half:
                                a_ed = small_half
                            elif a_real_len >= big_half and big_half <= b_real_len < a_real_len:
                                a_ed = big_half
                            elif a_real_len == small_half and b_real_len >= big_half:
                                a_ed = small_half
                            elif a_real_len >= big_half and b_real_len == small_half:
                                a_ed = big_half
                            elif a_real_len == small_half and b_real_len == small_half:
                                a_ed = small_half
                            elif a_real_len == small_half and b_real_len < small_half:
                                a_ed = small_half
                            elif a_real_len < small_half and b_real_len == small_half:
                                a_ed = a_real_len
                            elif a_real_len < small_half and b_real_len >= big_half and (a_real_len + b_real_len) >= (
                                    small_half + big_half):
                                a_ed = a_real_len
                            elif a_real_len < small_half and b_real_len >= big_half and (a_real_len + b_real_len) < (
                                    small_half + big_half):
                                a_ed = a_real_len
                            elif a_real_len >= big_half and b_real_len < small_half and (a_real_len + b_real_len) >= (
                                    small_half + big_half):
                                a_ed = self.column_length - 1 - b_real_len - 2
                            elif a_real_len >= big_half and b_real_len < small_half and (a_real_len + b_real_len) < (
                                    small_half + big_half):
                                a_ed = a_real_len
                            elif a_real_len < small_half and b_real_len < small_half:
                                a_ed = a_real_len
                            b_st = a_ed + 2
                        a_idx = None
                        b_idx = None
                        for j in range(self.window_size):
                            if (a_ed - a_st + 1) < self.window_size:
                                a_tmp = torch.unsqueeze(
                                    torch.arange(min(a_st + j, a_real_len), min(a_st + j, a_real_len) + 1), dim=1)
                            else:
                                a_tmp = torch.unsqueeze(torch.arange(a_st + j, a_ed + 2 - self.window_size + j), dim=1)
                            if (b_ed - b_st + 1) < self.window_size:
                                b_tmp = torch.unsqueeze(
                                    torch.arange(min(b_st + j, b_real_len), min(b_st + j, b_real_len) + 1), dim=1)
                            else:
                                b_tmp = torch.unsqueeze(torch.arange(b_st + j, b_ed + 2 - self.window_size + j), dim=1)
                            if j == 0:
                                a_idx = a_tmp
                                b_idx = b_tmp
                            else:
                                a_idx = torch.cat((a_idx, a_tmp), dim=1)
                                b_idx = torch.cat((b_idx, b_tmp), dim=1)
                        a_idx = a_idx.numpy().tolist()
                        b_idx = b_idx.numpy().tolist()
                        idx = torch.tensor(list(map(lambda x: x[0] + x[1], itertools.product(a_idx, b_idx))))
                        self.windows.append(idx.numpy().tolist())

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sentence_pair = self.sentence_pairs[index]
        label = self.labels[index]
        length = len(sentence_pair)
        original_sentence_pair = self.original_sentence_pairs[index]
        original_labels = self.original_labels[index]
        window = self.windows[index]
        if self.multi_columns:
            neighbors = self.column_graph[index]
        else:
            neighbors = None
        return sentence_pair, label, length, original_sentence_pair, original_labels, window, neighbors

    @staticmethod
    def pad(batch):
        """
        Pads to the longest sample.
        """
        f = lambda x: [sample[x] for sample in batch]

        lengths = f(2)
        max_len = np.array(lengths).max()

        sentence_pairs = f(0)
        labels = f(1)
        original_sentence_pair = f(3)
        original_label = f(4)
        window = f(5)
        neighbors = f(6)

        for sentence_pair in sentence_pairs:
            sentence_pair += [0] * (max_len - len(sentence_pair))

        # pad windows in dimension 1
        left_len = int(max_len / 2 - 1)
        right_len = int(max_len / 2 - 2)
        window_size = int(len(window[0][0]) / 2)
        pair_max_len = (left_len + 1 - window_size) * (right_len + 1 - window_size)
        for win in window:
            win += [[0] * len(win[0]) for i in range(pair_max_len - len(win))]
        return sentence_pairs, labels, original_sentence_pair, original_label, window, neighbors

