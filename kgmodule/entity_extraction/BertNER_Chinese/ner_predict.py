'''
    不训练直接进行ner
    即对新文本直接进行预测
'''

import data_pre
import os
import numpy as np
import predict
import torch
import utils
from models.BERT import Config, Model


def ner_pred(in_filename):
    # 先把纯文本转化成csv文件
    data_pre.data_gen(in_filename)

    # 虽然提取了训练和验证的数据，但是预测过程只使用了test.csv
    os.environ['CUDA_VISIBLE_DEVICE'] = '1'
    dataset = './data/Abstract/'
    config = Config(dataset)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True  # 保证每次运行结果一样

    print('加载预测数据集')
    trainSL, devSL, testSL = utils.build_dataset(config)
    # print(len(trainSL))   474
    # print(len(devSL))   12
    # print(len(testSL))  16
    train_iter, dev_iter, test_iter = utils.build_data_loader(config, trainSL, devSL, testSL)

    # 模型训练，评估与测试
    model = Model(config).to(config.device)
    predict.predict(config, model, test_iter)


if __name__ == '__main__':
    filename = 'data/Abstract/new.txt'
    ner_pred(filename)
