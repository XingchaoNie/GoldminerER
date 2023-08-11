#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: Aaron_Lou
# @Date  : 2021-02-25
# @WeChat  : FreeRotate
# @Contact : 1196390870@qq.com
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from seqeval.metrics import accuracy_score, classification_report, f1_score
from transformers import AdamW
from utils import batch_variable, get_time_dif
from tqdm import tqdm


def calc_train_acc(pred_labels, gold_labels, mask=None):
    nb_right = ((pred_labels == gold_labels) * mask).sum().item()
    nb_total = mask.sum().item()
    return nb_right, nb_total

def train(config, model, train_iter, dev_iter, test_iter):
    """
    模型训练方法
    :param config:
    :param model:
    :param train_iter:
    :param dev_iter:
    :return:
    """

    start_time = time.time()
    optimizer = AdamW(params=model.parameters(), lr=config.learning_rate)
    lr_decay = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=config.lr_decay_factor, verbose=True,
                                                    patience=1, min_lr=config.min_lr)
    dev_best_loss = float('inf')
    dev_best_f1 = float('-inf')
    avg_loss = []

    for epoch in range(1, config.num_epochs + 1):
        train_loss = 0.
        train_right, train_total = 0, 0
        for batch_idx, batch_data in enumerate(train_iter):
            model.train()
            bert_inputs, label_ids, label_mask  = batch_variable(config, batch_data)
            loss, predicts = model(bert_inputs, label_ids, label_mask)

            loss_val = loss.data.item()
            avg_loss.append(loss_val)
            train_loss += loss_val

            nb_right, nb_total = calc_train_acc(predicts, label_ids, label_mask)
            train_right += nb_right
            train_total += nb_total

            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=config.grad_clip)
            optimizer.step()
            model.zero_grad()
            if batch_idx % 5 == 0:
                time_dif = get_time_dif(start_time)
                print('Epoch:{}-----Iter:{}-----Time:{}-----train_loss:{:.3f}-----ACC:{:.3f}'
                      .format(epoch, batch_idx + 1, time_dif, np.array(avg_loss).mean(), train_right / train_total))

        dev_loss, dev_acc, dev_f1, dev_report = evaluate(config, model, dev_iter)
        msg = 'Dev Loss:{0:>5.2}, Dev Acc:{1:>6.2%}, Dev F1 :{2:>6.2%}'
        print(msg.format(dev_loss, dev_acc, dev_f1))
        print("Precision, Recall and F1-Score")
        print(dev_report)

        if dev_f1 >= dev_best_f1:
            dev_best_f1 = dev_f1
            torch.save(model.state_dict(), config.save_path)
            improve = '* * * * * * * * * * * * * * Save Model * * * * * * * * * * * * * *'
            print(improve)

        # test_loss, test_acc, test_f1, test_report = evaluate(config, model, test_iter)
        # msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}, Test F1 :{2:>6.2%}'
        # print(msg.format(test_loss, test_acc, test_f1))
        # print("Precision, Recall and F1-Score")
        # print(test_report)

        lr_decay.step(dev_f1)


def evaluate(config, model, one_iter, output_dict=False):
    """

    :param config:
    :param model:
    :param one_iter:
    :return:
    """
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(one_iter)):
            # print(batch_idx, batch_data)
            bert_inputs, label_ids, label_mask  = batch_variable(config, batch_data)
            loss, predicts = model(bert_inputs, label_ids, label_mask)

            loss_total = loss_total + loss

            for i, sen_mask in enumerate(label_mask):
                for j, word_mask in enumerate(sen_mask):
                    if word_mask.item() == False:
                        predicts[i][j] = 0
            labels_list = []
            for index_i, ids in enumerate(label_ids):
                labels_list.append([config.id2class[id.cpu().item()] for index_j, id in enumerate(ids)])
            predicts_list = []
            for index_i, pres in enumerate(predicts):
                    predicts_list.append([config.id2class[pre.cpu().item()]  for index_j, pre in enumerate(pres)])

            labels_all += labels_list
            predict_all += predicts_list

    acc = accuracy_score(labels_all, predict_all)
    f1 = f1_score(labels_all, predict_all)
    report = classification_report(labels_all, predict_all, digits=3, output_dict=output_dict)
    return loss_total / len(one_iter), acc, f1,  report