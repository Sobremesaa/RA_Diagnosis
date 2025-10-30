#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/1/7 23:33
# @File    : data_processor.py
# @Software: PyCharm
# @description: 数据处理，包括label_map处理，dataset建立

import os
import json
import pickle

import torch
from torch.utils.data import Dataset

def get_label_map():
    # 标签字典保存路径
    label_map_path = '../mydata/label_map.json'
    # 第一次运行需要遍历训练集获取到标签字典，并存储成json文件保存，第二次运行即可直接载入json文件
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r', encoding='utf-8') as fp:
            label_map = json.load(fp)
    else:
        print("error:标签文件不存在！")

    # {0: 'B-address', 1: 'I-address', 2: 'B-book', 3: 'I-book'...}
    label_map_inv = {v: k for k, v in label_map.items()}
    return label_map, label_map_inv

def get_vocab(data_path=''):
    # 词表保存路径
    vocab_path = '../mydata/vocab.pkl'
    # 第一次运行需要遍历训练集获取到标签字典，并存储成json文件保存，第二次运行即可直接载入json文件
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as fp:
            vocab = pickle.load(fp)
    else:
        line_vacab = []
        # 加载数据集
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                print('line-----',line)
                rows = line.split(' ')
                if len(rows) != 1:
                    line_vacab.append(rows[0])
        # 建立词表字典，提前加入'PAD'和'UNK'
        # 'PAD'：在一个batch中不同长度的序列用该字符补齐
        # 'UNK'：当验证集或测试集出现词表以外的词时，用该字符代替
        vocab = {'PAD': 0, 'UNK': 1}
        # 遍历数据集，不重复取出所有字符，并记录索引
        for data in line_vacab:  # 获取实体标签，如'name'，'compan
            # print(data)
            if data not in vocab:
                vocab[data] = len(vocab)
        # vocab：{'PAD': 0, 'UNK': 1, '浙': 2, '商': 3, '银': 4, '行': 5...}
        # 保存成pkl文件
        with open(vocab_path, 'wb') as fp:
            pickle.dump(vocab, fp)
    # print('vocab-------')
    # print(vocab)

    # 翻转字表，预测时输出的序列为索引，方便转换成中文汉字
    # vocab_inv：{0: 'PAD', 1: 'UNK', 2: '浙', 3: '商', 4: '银', 5: '行'...}
    vocab_inv = {v: k for k, v in vocab.items()}
    return vocab, vocab_inv

def data_process(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        # print(lines)
        x, y = [], []
        for line in lines:
            rows = line.split(' ')
            # print('rows----', rows)
            if len(rows) == 1:
                data.append([x,y])
                x = []
                y = []
            else:
                x.append(rows[0])
                y.append(rows[1])
    return data

class Mydataset(Dataset):
    def __init__(self, file_path, vocab, label_map):
        self.file_path = file_path
        # 数据预处理
        self.data = data_process(self.file_path)
        # print("-----data-----")
        # print(self.data)
        self.label_map, self.label_map_inv = label_map
        self.vocab, self.vocab_inv = vocab
        # self.data为中文汉字和英文标签，将其转化为索引形式
        self.examples = []
        for text, label in self.data:
            print("text------",text)
            print("label-----",label)
            t = [self.vocab.get(t, self.vocab['UNK']) for t in text]
            print('t------',t)
            l = [self.label_map[l] for l in label]
            print('l------',l)
            self.examples.append([t, l])
        print("self.examples-----",self.examples[:500])

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.data)

    def collect_fn(self, batch):
        # 取出一个batch中的文本和标签，将其单独放到变量中处理
        # 长度为batch_size，每个序列长度为原始长度
        # print("batch------", batch)
        text = [t for t, l in batch]
        # print("2text------",text)
        label = [l for t, l in batch]
        # 获取一个batch内所有序列的长度，长度为batch_size
        seq_len = [len(i) for i in text]
        # 提取出最大长度用于填充
        max_len = max(seq_len)

        # 填充到最大长度，文本用'PAD'补齐，标签用'O'补齐
        text = [t + [self.vocab['PAD']] * (max_len - len(t)) for t in text]
        label = [l + [self.label_map['O']] * (max_len - len(l)) for l in label]

        # 将其转化成tensor，再输入到模型中，这里的dtype必须是long否则报错
        # text 和 label shape：(batch_size, max_len)
        # seq_len shape：(batch_size,)
        text = torch.tensor(text, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        seq_len = torch.tensor(seq_len, dtype=torch.long)

        return text, label, seq_len
