#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : BERT.py
# @Author: Aaron_Lou
# @Date  : 2021/5/22 20:25
# @WeChat  : FreeRotate
# @Contact : 1196390870@qq.com
import os

import torch
import torch.nn as nn
from torchcrf import CRF
import warnings
# from TorchCRF import CRF
# from models.transformers.models.bert import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer

warnings.filterwarnings('ignore')

class Config(object):
    """
    配置参数
    """
    def __init__(self, dataset):
        self.model_name = 'BERT'  # 模型名称

        self.train_path = dataset + 'train.csv'  # 训练集
        self.dev_path = dataset + 'dev.csv'  # 验证集
        self.test_path = dataset + 'test.csv'  # 测试集

        # print(dataset + 'class.txt')
        # TODO 抽取PKG图谱中实体和关系，需要更改这里
        # self.class_list = ['O', 'B-approach', 'I-approach', 'B-condition', 'I-condition', 'B-criteria', 'I-criteria', 'B-definition', 'I-definition', 'B-feature', 'I-feature', 'B-location', 'I-location', 'B-measure', 'I-measure', 'B-part', 'I-part', 'B-procedure', 'I-procedure', 'B-reference', 'I-reference', 'B-tolerance', 'I-tolerance', 'B-tool', 'I-tool']
        self.class_list = ['O', 'B-RCLSB', 'I-RCLSB', 'B-RCLSBName', 'I-RCLSBName', 'B-RCLSBNo', 'I-RCLSBNo', 'B-RCLSBCate',
         'I-RCLSBCate', 'B-RCLSBType', 'I-RCLSBType', 'B-RCLSBJrfs', 'I-RCLSBJrfs', 'B-RCLGZ', 'I-RCLGZ', 'B-RCLGZNo',
         'I-RCLGZNo', 'B-RCLGZName', 'I-RCLGZName', 'B-RCLGZDWG', 'I-RCLGZDWG', 'B-TL', 'I-TL', 'B-TLNo', 'I-TLNo',
         'B-TLName', 'I-TLName', 'B-LQJZ', 'I-LQJZ', 'B-LQJZNo', 'I-LQJZNo', 'B-LQJZName', 'I-LQJZName', 'B-RCLCS',
         'I-RCLCS', 'B-RCLCSBfjg', 'I-RCLCSBfjg', 'B-RCLCSBffs', 'I-RCLCSBffs', 'B-RCLCSZll', 'I-RCLCSZll',
         'B-RCLCSYxhd', 'I-RCLCSYxhd', 'B-RCLCSBwwd', 'I-RCLCSBwwd', 'B-RCLCSSwsj', 'I-RCLCSSwsj', 'B-RCLCSBwsj',
         'I-RCLCSBwsj', 'B-RCLCSZysj', 'I-RCLCSZysj', 'B-RCLCSLqjz', 'I-RCLCSLqjz', 'B-RCLCSLqsj', 'I-RCLCSLqsj',
         'B-RCLCSJzwdbhfw', 'I-RCLCSJzwdbhfw']
        # self.class_list = ['O', 'B-RCLSB', 'I-RCLSB', 'B-RCLSBName', 'I-RCLSBName', 'B-RCLSBNo', 'I-RCLSBNo', 'B-RCLSBCate', 'I-RCLSBCate', 'B-RCLSBType', 'I-RCLSBType', 'B-RCLSBJrfs', 'I-RCLSBJrfs', 'B-RCLGZ', 'I-RCLGZ', 'B-RCLGZNo', 'I-RCLGZNo', 'B-RCLGZName', 'I-RCLGZName', 'B-RCLGZDWG', 'I-RCLGZDWG', 'B-TL', 'I-TL', 'B-TLNo', 'I-TLNo', 'B-TLName', 'I-TLName', 'B-LQJZ', 'I-LQJZ', 'B-LQJZNo', 'I-LQJZNo', 'B-LQJZName', 'I-LQJZName', 'B-GJ', 'I-GJ', 'B-GJDWG', 'I-GJDWG', 'B-GJName', 'I-GJName', 'B-GJMaterial', 'I-GJMaterial', 'B-GJShape', 'I-GJShape', 'B-GJJgyl', 'I-GJJgyl', 'B-RCLCS', 'I-RCLCS', 'B-RCLCSBfjg', 'I-RCLCSBfjg', 'B-RCLCSBffs', 'I-RCLCSBffs', 'B-RCLCSZll', 'I-RCLCSZll', 'B-RCLCSYxhd', 'I-RCLCSYxhd', 'B-RCLCSBwwd', 'I-RCLCSBwwd', 'B-RCLCSSwsj', 'I-RCLCSSwsj', 'B-RCLCSBwsj', 'I-RCLCSBwsj', 'B-RCLCSZysj', 'I-RCLCSZysj', 'B-RCLCSLqjz', 'I-RCLCSLqjz', 'B-RCLCSLqsj', 'I-RCLCSLqsj', 'B-RCLCSJzwdbhfw', 'I-RCLCSJzwdbhfw', 'B-ZLXX', 'I-ZLXX', 'B-ZLXXYd', 'I-ZLXXYd', 'B-ZLXXQfqd', 'I-ZLXXQfqd', 'B-ZLXXKlqd', 'I-ZLXXKlqd', 'B-ZLXXDmssl', 'I-ZLXXDmssl', 'B-ZLXXDmscl', 'I-ZLXXDmscl', 'B-ZLXXCjg', 'I-ZLXXCjg']
        # self.class_list = [x.strip() for x in open(dataset + 'class.txt').readlines()]  # 类别
        self.id2class = dict(enumerate(self.class_list))  # 标号转类别

        self.class2id = {j: i for i, j in self.id2class.items()}  # 类别转标号
        # print(self.class2id)   # {'O': 0, 'B-dev': 1, 'I-dev': 2, 'B-warn': 3, 'I-warn': 4, 'B-maintain': 5, 'I-maintain': 6, 'B-bug': 7, 'I-bug': 8, 'B-solve_way': 9, 'I-solve_way': 10, 'B-bug_reason': 11, 'I-bug_reason': 12}

        self.num_classes = len(self.class_list)  # 类别数量
        # print(self.num_classes)   # 13
        self.pad_size = 128

        self.device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else 'cpu')

        # TODO: 确保在六所上部署没问题后进行调参
        self.num_epochs = 1  # 轮次数  20  !!!!!
        self.batch_size = 4  # batch_size，一次传入64个pad_size   12
        self.learning_rate = 1e-5  # 学习率
        # bert_path：单独进行NER训练的时候用相对路径，对新来文件进行预测的时候用绝对路径
        self.bert_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), r'pretrain_models', r'bert_pytorch_model')
        print(self.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path, do_lower_case=False)  # bert切词器

        self.bert_layers = 4

        self.grad_clip = 5.0
        self.lr_decay_factor = 0.95
        self.min_lr = 1e-6

        self.save_path = dataset + 'saved_model/' + self.model_name + \
                         '_Epoch' + str(self.num_epochs) + \
                         '_Batch' + str(self.batch_size) + \
                         '_Lr' + str(self.learning_rate) + '.pth'  # 保存模型

        self.evaluate_path = dataset + 'evaluate/' + self.model_name + \
                             '_Epoch' + str(self.num_epochs) + \
                             '_Batch' + str(self.batch_size) + \
                             '_Lr' + str(self.learning_rate) + '.csv'  # 验证效果

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.num_classes = config.num_classes
        self.bert = BertModel.from_pretrained(config.bert_path, output_hidden_states=True)

        self.bert_layers = self.bert.config.num_hidden_layers
        self.nb_layers = config.bert_layers if config.bert_layers < self.bert_layers else self.bert_layers
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.tag_crf = CRF(num_tags=config.num_classes, batch_first=True)

        self.classifier = nn.Linear(self.hidden_size, config.num_classes)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, bert_inputs, label_ids, label_mask, crf=False):
        ids_lens = bert_inputs[3]
        batch_size, seq_len = ids_lens.shape
        mask = ids_lens.gt(0)

        input_ids = bert_inputs[0]
        attention_mask = bert_inputs[1].type_as(mask)
        token_type_ids = bert_inputs[2]

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_enc_out, _, all_enc_outs = outputs[0], outputs[1], outputs[2]
        top_enc_outs = all_enc_outs[-self.nb_layers:]
        bert_out = sum(top_enc_outs) / len(top_enc_outs)    # 取最后n层平均值输出

        bert_mask = attention_mask.type_as(mask)

        bert_chunks = bert_out[bert_mask].split(ids_lens[mask].tolist())
        bert_out = torch.stack(tuple([bc.mean(0) for bc in bert_chunks]))
        bert_embed = bert_out.new_zeros(batch_size, seq_len, self.hidden_size)
        # 将bert_embed中mask对应1的位置替换成bert_out，0的位置不变
        bert_embed = bert_embed.masked_scatter_(mask.unsqueeze(dim=-1), bert_out)
        bert_embed = self.dropout(bert_embed)

        label_predict = self.classifier(bert_embed)

        if crf:
            lld = self.tag_crf(label_predict , label_ids, label_mask)
            label_loss = lld.neg()
        else:
            active_logits = label_predict.view(-1, self.num_classes)
            active_labels = torch.where(label_mask.view(-1), label_ids.view(-1), self.loss_fct.ignore_index)
            label_loss = self.loss_fct(active_logits, active_labels)

        loss = label_loss
        output = label_predict.data.argmax(dim=-1)
        return loss, output