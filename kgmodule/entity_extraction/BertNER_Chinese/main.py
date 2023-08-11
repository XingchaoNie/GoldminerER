#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py

import os
import time

import numpy as np
import predict
import torch
import train
import power.entity_extraction.BertNER_Chinese.test as test
import utils
from models.BERT import Config, Model

os.environ['CUDA_VISIBLE_DEVICE'] = '1'

if __name__ == '__main__':
    dataset = './data/Abstract/'
    config = Config(dataset)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True   # 保证每次运行结果一样

    start_time = time.time()
    print('加载数据集')
    trainSL, devSL, testSL = utils.build_dataset(config)
    # print(len(trainSL))   474
    # print(len(devSL))   12
    # print(len(testSL))  16
    train_iter, dev_iter, test_iter = utils.build_data_loader(config, trainSL, devSL, testSL)
    time_dif = utils.get_time_dif(start_time)
    print("模型开始之前，准备数据时间：", time_dif)

    #模型训练，评估与测试
    model = Model(config).to(config.device)
    # print("dddddd")
    train.train(config, model, train_iter, dev_iter, test_iter)  # 训练模型
    test.test(config, model, dev_iter)
    # print("dsadasdsadasd")
    predict.predict(config, model, test_iter)
    # print("dasdasdasdasdasdasdasd")
    predict.predict(config, model, dev_iter)
    # print("dddddddddddddddddddddddddddd")