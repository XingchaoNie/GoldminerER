#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: Aaron_Lou
# @Date  : 2021-04-14
# @WeChat  : FreeRotate
# @Contact : 1196390870@qq.com
import torch
import pandas as pd
from train import evaluate

def test(config, model, one_iter):
    model.load_state_dict(torch.load(config.save_path), False)
    model.eval()
    loss, acc, f1, report = evaluate(config, model, one_iter, output_dict=True)
    msg = 'Loss:{0:>5.2}, Acc:{1:>6.2%}, Dev F1 :{2:>6.2%}'
    print(msg.format(loss, acc, f1))
    print("Precision, Recall and F1-Score")
    file = pd.DataFrame(report)
    file = file.T
    print(file)
    file.to_csv(config.evaluate_path, mode='a')