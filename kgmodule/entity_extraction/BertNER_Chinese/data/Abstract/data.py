# 构建实体识别所用训练数据，拆分训练集 验证集 测试集
import os
import re
import json
import random
import pandas as pd


def data():
    random.seed(12345)

    # data = pd.read_csv(r'./data_all.csv')
    data = pd.read_csv(r'../../datasets/data_all.csv')
    # data = pd.read_csv(r'D:\Re\code\power\datasets\data_all.csv')
    # data = pd.read_csv(r'../../../../datasets/data_all.csv')

    data = data.values[:, 1:]
    sentence = []
    labels = []
    # 循环读取内层列表
    cnt = 0
    for i in data:
        cnt += 1
        # 每一句话有一个sep，一句话字符大于500，那么会在500的时候断开，也就是说限制每句话最长500字
        flag = 0
        # print(i[0])
        # print(type(i[0]))
        # break
        # i = i.reshape(2,-1)
        if isinstance(i[0], float):
            continue
            # print(i[0])
            # print(cnt)
            # break
        for j in list(i[0]):
            flag += 1
            if (flag > 500):
                sentence.append('sep')
                flag = 0
            sentence.append(j)
        sentence.append('sep')
        # print(sentence)
        flag = 0
        for j in i[1].split(' '):
            flag += 1
            if (flag > 500):
                labels.append('sep')
                flag = 0

            labels.append(j)
        labels.append('sep')

    # print(sentence)
    # print(labels)

    sen_len = len(sentence)
    lab_len = len(labels)
    seq1 = sen_len // 25 * 18  # 训练和验证的分界数
    seq2 = sen_len // 25 * 22  # 验证和测试的分界数
    print(seq1, seq2)
    train_sentence = sentence[:seq1]
    dev_sentence = sentence[seq1:seq2]
    test_sentence = sentence[seq2:]

    train_labels = labels[:seq1]
    dev_labels = labels[seq1:seq2]
    test_labels = labels[seq2:]

    # 训练集
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'sentence': train_sentence, 'labels': train_labels})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("data/Abstract/train.csv", index=False, sep=',')

    # 验证集
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'sentence': dev_sentence, 'labels': dev_labels})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("data/Abstract/dev.csv", index=False, sep=',')

    # 测试集
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'sentence': test_sentence, 'labels': test_labels})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("data/Abstract/test.csv", index=False, sep=',')

if __name__ == '__main__':
    data()
