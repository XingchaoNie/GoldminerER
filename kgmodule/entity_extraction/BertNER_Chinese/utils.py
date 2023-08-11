#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: Aaron_Lou
# @Date  : 2021-02-22
# @WeChat  : FreeRotate
# @Contact : 1196390870@qq.com
import math
import time
import torch
import numpy as np
import pandas as pd
from datetime import timedelta

PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'

class WordLabel(object):
    def __init__(self, word: str, label: str):
        self.word = word
        self.label = label
    def __str__(self):
        return str(self.__dict__)
    def __repr__(self):
        return str(self.__dict__)


class DataLoader(object):
    def __init__(self, dataset: list, batch_size=1, shuffle=False, collate_fn=None):
        self.dataset = dataset
        # print(self.datasets)  [[{'word': '上', 'label': 'O'}, {'word': '午', 'label': 'O'}, {'word': '1', 'label': 'B-dev'},..]...[]]
        # print(len(self.datasets))    474 12 16
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            # permutation() 随机排列一个序列，返回一个排列的序列。
            idxs = np.random.permutation(n)
        else:
            idxs = range(n)

        batch = []
        for idx in idxs:
            batch.append(self.dataset[idx])   # [[{'word': '上', 'label': 'O'}, {'word': '午', 'label': 'O'},...]..[]]
            # print(self.batch_size)    10
            if len(batch) == self.batch_size:
                if self.collate_fn:
                    yield self.collate_fn(batch)
                else:
                    yield batch
                batch = []

        if len(batch) > 0:
            if self.collate_fn:
                yield self.collate_fn(batch)
            else:
                # print(batch)
                yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

# 返回嵌套列表，内层嵌套是每行的字符字典
def load_dataset(file_path):
    file = pd.read_csv(file_path, encoding='utf-8')
    sentence, labels = [], []
    sentence_labels = []
    dataset = []
    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    for word, label in zip(file['sentence'], file['labels']):
        # 遇到分割符sep就将该句子的字符字典加入dataset列表
        if word == 'sep' and label == 'sep':
            for w, l in zip(sentence, labels):
                # print(WordLabel(w,l))  # {'word': '上', 'label': 'O'}
                sentence_labels.append(WordLabel(w, l))
            dataset.append(sentence_labels)
            sentence_labels = []
            sentence, labels = [], []
        else:
            sentence.append(word)
            labels.append(label)
    # print(datasets)  [[{'word': '上', 'label': 'O'},{..}]..[]]
    return dataset

def build_dataset(config):
    # load_dataset 返回嵌套列表，内层嵌套是每行的字符字典
    # 如[[{'word': '上', 'label': 'O'},{..}]..[]]
    trainSL = load_dataset(config.train_path)
    devSL = load_dataset(config.dev_path)
    testSL = load_dataset(config.test_path)

    return trainSL, devSL, testSL

def build_data_loader(config, trainSL, devSL, testSL):
    train_data_loader = DataLoader(trainSL, batch_size=config.batch_size, shuffle=False)
    dev_data_loader = DataLoader(devSL, batch_size=config.batch_size, shuffle=False)
    test_data_loader = DataLoader(testSL, batch_size=config.batch_size, shuffle=False)
    return train_data_loader, dev_data_loader, test_data_loader

def bert_word2id(sentence_list, config):
    #接收一个batch的数据
    ids_list = []
    lens_list = []
    segment_list = []
    mask_list = []
    tokenizer = config.tokenizer
    max_ids_lens = 0
    max_lens_lens = 0

    for sentence in sentence_list:
        ids = [101]     #加上[CLS]标识符
        lens = [1]
        for word in sentence:
            if isinstance(word, float):
                if math.isnan(word):
                    word = 'nna'
            id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            for i in id:
                ids.append(i)
            lens.append(len(id))
        if len(ids) > max_ids_lens:
            max_ids_lens = len(ids)
        if len(lens) > max_lens_lens:
            max_lens_lens = len(lens)

        ids_list.append(ids)    #单词后的编码
        lens_list.append(lens)  #原始单词被切分编码后长度

    mask_list += [[1]*len(ids) for ids in ids_list]
    for index, item in enumerate(ids_list):
        pad_size = max_ids_lens - len(item)
        ids_list[index] += [0 for i in range(pad_size)]
        mask_list[index] += [0 for i in range(pad_size)]


    for index, item in enumerate(lens_list):
        pad_size = max_lens_lens - len(item)
        lens_list[index] += [0 for i in range(pad_size)]

    segment_list += [[0] * len(ids) for ids in ids_list]

    return ids_list, segment_list, mask_list, lens_list

def batch_variable(config, batch_data):
    batch_size = len(batch_data)
    max_seq_len = 1 + max(len(insts) for insts in batch_data)
    label_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    sentence_list = []
    for index, sentence in enumerate(batch_data):
        # print(index, sentence)    # 0 [{'word': '上', 'label': 'O'}, {'word': '午', 'label': 'O'}, {'word': '1', 'label': 'B-dev'}....]
        seq_len = len(sentence) + 1
        # class2id类别转标号
        # 每句的每字类别
        label_ids[index, 1:seq_len] = torch.tensor([config.class2id[item.label] for item in sentence])
        # print(label_ids[index, 1:seq_len])   tensor([0, 0, 1, 2, 2, 2, 0, 0, 0,....])
        label_mask[index, :seq_len].fill_(1)
        # print(label_mask[index, :seq_len]) # tensor([True, True, True, True...])
        sentence_list.append([item.word for item in sentence])

    # 两层列表,外层有batch_size个句子
    # print(sentence_list)   [['上', '午', '1', '#', '风', '机', '维', ....]..[]]
    ids_list, segment_list, mask_list, lens_list = bert_word2id(sentence_list, config)
    id_list = torch.LongTensor(ids_list)
    mask_list = torch.LongTensor(mask_list)  # 有字符就1,没有就0
    segment_list = torch.LongTensor(segment_list)   # 全0
    lens_list = torch.LongTensor(lens_list)
    # print(id_list)
    bert_inputs = [id_list.to(config.device), mask_list.to(config.device), segment_list.to(config.device),  lens_list.to(config.device)]

    return bert_inputs, label_ids.to(config.device), label_mask.to(config.device)



def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))