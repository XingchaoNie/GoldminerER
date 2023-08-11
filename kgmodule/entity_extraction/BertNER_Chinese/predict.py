#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : predict.py
# @Author: Aaron_Lou
# @Date  : 2021/5/23 19:46
# @WeChat  : FreeRotate
# @Contact : 1196390870@qq.com
import torch
import os
from utils import batch_variable


def predict(config, model, one_iter, output_dict=True):
    # 返回抽取到的实体列表组成的字典
    # {entity1: [list1], entity2: [list2]}
    """

    :param config:
    :param model:
    :param one_iter:
    :return:
    """
    model.load_state_dict(torch.load(config.save_path), False)
    model.eval()
    # TODO: 张博确认好实体关系类型之后在这里改
    # predict_all = {
    #     "approach": [],
    #     "condition": [],
    #     "criteria": [],
    #     "definition": [],
    #     "feature": [],
    #     "location": [],
    #     "measure": [],
    #     "part": [],
    #     "procedure": [],
    #     "reference": [],
    #     "tolerance": [],
    #     "tool": []
    # }
    predict_all = {'RCLSB': [], 'RCLSBName': [], 'RCLSBNo': [], 'RCLSBCate': [], 'RCLSBType': [], 'RCLSBJrfs': [], 'RCLGZ': [],
     'RCLGZNo': [], 'RCLGZName': [], 'RCLGZDWG': [], 'TL': [], 'TLNo': [], 'TLName': [], 'LQJZ': [], 'LQJZNo': [],
     'LQJZName': [], 'RCLCS': [], 'RCLCSBfjg': [], 'RCLCSBffs': [], 'RCLCSZll': [], 'RCLCSYxhd': [], 'RCLCSBwwd': [],
     'RCLCSSwsj': [], 'RCLCSBwsj': [], 'RCLCSZysj': [], 'RCLCSLqjz': [], 'RCLCSLqsj': [], 'RCLCSJzwdbhfw': []}
    with torch.no_grad():
        # TODO: 张博确认好实体关系类型之后在这里改
        # 遍历batch_size个大小的预测数据
        # approach = []
        # condition = []
        # criteria = []
        # definition = []
        # feature = []
        # location = []
        # measure = []
        # part = []
        # procedure = []
        # reference = []
        # tolerance = []
        # tool = []

        RCLSB = []
        RCLSBName = []
        RCLSBNo = []
        RCLSBCate = []
        RCLSBType = []
        RCLSBJrfs = []
        RCLGZ = []
        RCLGZNo = []
        RCLGZName = []
        RCLGZDWG = []
        TL = []
        TLNo = []
        TLName = []
        LQJZ = []
        LQJZNo = []
        LQJZName = []
        RCLCS = []
        RCLCSBfjg = []
        RCLCSBffs = []
        RCLCSZll = []
        RCLCSYxhd = []
        RCLCSBwwd = []
        RCLCSSwsj = []
        RCLCSBwsj = []
        RCLCSZysj = []
        RCLCSLqjz = []
        RCLCSLqsj = []
        RCLCSJzwdbhfw = []

        for batch_idx, batch_data in enumerate(one_iter):
            bert_inputs, label_ids, label_mask = batch_variable(config, batch_data)
            # model return loss, output
            _, predicts = model(bert_inputs, label_ids, label_mask)
            print(predicts)
            # TODO:张博确认好实体关系类型之后在这里改
            # dict = {0: 'O', 1: 'B-approach', 2: 'I-approach', 3: 'B-condition', 4: 'I-condition',
            #         5: 'B-criteria', 6: 'I-criteria', 7: 'B-definition', 8: 'I-definition', 9: 'B-feature',
            #         10: 'I-feature', 11: 'B-location', 12: 'I-location', 13: 'B-measure', 14: 'I-measure', 15: 'B-part',
            #         16: 'I-part', 17: 'B-procedure', 18: 'I-procedure', 19: 'B-reference', 20: 'I-reference',
            #         21: 'B-tolerance', 22: 'I-tolerance', 23: 'B-tool', 24: 'I-tool'}
            dict = {0: 'O', 1: 'B-RCLSB', 2: 'I-RCLSB', 3: 'B-RCLSBName', 4: 'I-RCLSBName', 5: 'B-RCLSBNo', 6: 'I-RCLSBNo', 7: 'B-RCLSBCate', 8: 'I-RCLSBCate', 9: 'B-RCLSBType', 10: 'I-RCLSBType', 11: 'B-RCLSBJrfs', 12: 'I-RCLSBJrfs', 13: 'B-RCLGZ', 14: 'I-RCLGZ', 15: 'B-RCLGZNo', 16: 'I-RCLGZNo', 17: 'B-RCLGZName', 18: 'I-RCLGZName', 19: 'B-RCLGZDWG', 20: 'I-RCLGZDWG', 21: 'B-TL', 22: 'I-TL', 23: 'B-TLNo', 24: 'I-TLNo', 25: 'B-TLName', 26: 'I-TLName', 27: 'B-LQJZ', 28: 'I-LQJZ', 29: 'B-LQJZNo', 30: 'I-LQJZNo', 31: 'B-LQJZName', 32: 'I-LQJZName', 33: 'B-RCLCS', 34: 'I-RCLCS', 35: 'B-RCLCSBfjg', 36: 'I-RCLCSBfjg', 37: 'B-RCLCSBffs', 38: 'I-RCLCSBffs', 39: 'B-RCLCSZll', 40: 'I-RCLCSZll', 41: 'B-RCLCSYxhd', 42: 'I-RCLCSYxhd', 43: 'B-RCLCSBwwd', 44: 'I-RCLCSBwwd', 45: 'B-RCLCSSwsj', 46: 'I-RCLCSSwsj', 47: 'B-RCLCSBwsj', 48: 'I-RCLCSBwsj', 49: 'B-RCLCSZysj', 50: 'I-RCLCSZysj', 51: 'B-RCLCSLqjz', 52: 'I-RCLCSLqjz', 53: 'B-RCLCSLqsj', 54: 'I-RCLCSLqsj', 55: 'B-RCLCSJzwdbhfw', 56: 'I-RCLCSJzwdbhfw'}

            # 遍历tensor，去掉第一列
            predicts = predicts[:, torch.arange(predicts.size(1)) != 0]
            dim0, dim1 = predicts.shape
            print(f"shape：{dim0}, {dim1}")  # 4 100

            # 抽取所有类型的实体
            for i in range(dim0):
                for j in range(len(batch_data[i])):
                    # TODO:张博确认好实体关系类型之后在这里改，整个if下
                    if predicts[i][j].item() != 0:
                        if dict[predicts[i][j].item()] == 'B-RCLSB':
                            rclsb = ''
                            rclsb = rclsb + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLSB':
                                rclsb = rclsb + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclsb) > 1:
                                RCLSB.append(rclsb)

                        elif dict[predicts[i][j].item()] == 'B-RCLSBName':
                            rclsbname = ''
                            rclsbname = rclsbname + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLSBName':
                                rclsbname = rclsbname + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclsbname) > 1:
                                RCLSBName.append(rclsbname)

                        elif dict[predicts[i][j].item()] == 'B-RCLSBNo':
                            rclsbno = ''
                            rclsbno = rclsbno + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLSBNo':
                                rclsbno = rclsbno + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclsbno) > 1:
                                RCLSBNo.append(rclsbno)

                        elif dict[predicts[i][j].item()] == 'B-RCLSBCate':
                            rclsbcate = ''
                            rclsbcate = rclsbcate + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLSBCate':
                                rclsbcate = rclsbcate + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclsbcate) > 1:
                                RCLSBCate.append(rclsbcate)

                        elif dict[predicts[i][j].item()] == 'B-RCLSBType':
                            rclsbtype = ''
                            rclsbtype = rclsbtype + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLSBType':
                                rclsbtype = rclsbtype + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclsbtype) > 1:
                                RCLSBType.append(rclsbtype)

                        elif dict[predicts[i][j].item()] == 'B-RCLSBJrfs':
                            rclsbjrfs = ''
                            rclsbjrfs = rclsbjrfs + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLSBJrfs':
                                rclsbjrfs = rclsbjrfs + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclsbjrfs) > 1:
                                RCLSBJrfs.append(rclsbjrfs)

                        elif dict[predicts[i][j].item()] == 'B-RCLGZ':
                            rclgz = ''
                            rclgz = rclgz + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLGZ':
                                rclgz = rclgz + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclgz) > 1:
                                RCLGZ.append(rclgz)

                        elif dict[predicts[i][j].item()] == 'B-RCLGZNo':
                            rclgzno = ''
                            rclgzno = rclgzno + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLGZNo':
                                rclgzno = rclgzno + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclgzno) > 1:
                                RCLGZNo.append(rclgzno)

                        elif dict[predicts[i][j].item()] == 'B-RCLGZName':
                            rclgzname = ''
                            rclgzname = rclgzname + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLGZName':
                                rclgzname = rclgzname + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclgzname) > 1:
                                RCLGZName.append(rclgzname)

                        elif dict[predicts[i][j].item()] == 'B-RCLGZDWG':
                            rclgzdwg = ''
                            rclgzdwg = rclgzdwg + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLGZDWG':
                                rclgzdwg = rclgzdwg + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclgzdwg) > 1:
                                RCLGZDWG.append(rclgzdwg)

                        elif dict[predicts[i][j].item()] == 'B-TL':
                            tl = ''
                            tl = tl + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-TL':
                                tl = tl + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(tl) > 1:
                                TL.append(tl)

                        elif dict[predicts[i][j].item()] == 'B-TLNo':
                            tlno = ''
                            tlno = tlno + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-TLNo':
                                tlno = tlno + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(tlno) > 1:
                                TLNo.append(tlno)

                        elif dict[predicts[i][j].item()] == 'B-TLName':
                            tlname = ''
                            tlname = tlname + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-TLName':
                                tlname = tlname + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(tlname) > 1:
                                TLName.append(tlname)

                        elif dict[predicts[i][j].item()] == 'B-LQJZ':
                            lqjz = ''
                            lqjz = lqjz + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-LQJZ':
                                lqjz = lqjz + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(lqjz) > 1:
                                LQJZ.append(lqjz)

                        elif dict[predicts[i][j].item()] == 'B-LQJZNo':
                            lqjzno = ''
                            lqjzno = lqjzno + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-LQJZNo':
                                lqjzno = lqjzno + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(lqjzno) > 1:
                                LQJZNo.append(lqjzno)

                        elif dict[predicts[i][j].item()] == 'B-LQJZName':
                            lqjzname = ''
                            lqjzname = lqjzname + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-LQJZName':
                                lqjzname = lqjzname + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(lqjzname) > 1:
                                LQJZName.append(lqjzname)

                        elif dict[predicts[i][j].item()] == 'B-RCLCS':
                            rclcs = ''
                            rclcs = rclcs + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLCS':
                                rclcs = rclcs + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclcs) > 1:
                                RCLCS.append(rclcs)

                        elif dict[predicts[i][j].item()] == 'B-RCLCSBfjg':
                            rclcsbfjg = ''
                            rclcsbfjg = rclcsbfjg + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLCSBfjg':
                                rclcsbfjg = rclcsbfjg + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclcsbfjg) > 1:
                                RCLCSBfjg.append(rclcsbfjg)

                        elif dict[predicts[i][j].item()] == 'B-RCLCSBffs':
                            rclcsbffs = ''
                            rclcsbffs = rclcsbffs + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLCSBffs':
                                rclcsbffs = rclcsbffs + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclcsbffs) > 1:
                                RCLCSBffs.append(rclcsbffs)

                        elif dict[predicts[i][j].item()] == 'B-RCLCSZll':
                            rclcszll = ''
                            rclcszll = rclcszll + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLCSZll':
                                rclcszll = rclcszll + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclcszll) > 1:
                                RCLCSZll.append(rclcszll)

                        elif dict[predicts[i][j].item()] == 'B-RCLCSYxhd':
                            rclcsyxhd = ''
                            rclcsyxhd = rclcsyxhd + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLCSYxhd':
                                rclcsyxhd = rclcsyxhd + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclcsyxhd) > 1:
                                RCLCSYxhd.append(rclcsyxhd)

                        elif dict[predicts[i][j].item()] == 'B-RCLCSBwwd':
                            rclcsbwwd = ''
                            rclcsbwwd = rclcsbwwd + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLCSBwwd':
                                rclcsbwwd = rclcsbwwd + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclcsbwwd) > 1:
                                RCLCSBwwd.append(rclcsbwwd)

                        elif dict[predicts[i][j].item()] == 'B-RCLCSSwsj':
                            rclcsswsj = ''
                            rclcsswsj = rclcsswsj + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLCSSwsj':
                                rclcsswsj = rclcsswsj + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclcsswsj) > 1:
                                RCLCSSwsj.append(rclcsswsj)

                        elif dict[predicts[i][j].item()] == 'B-RCLCSBwsj':
                            rclcsbwsj = ''
                            rclcsbwsj = rclcsbwsj + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLCSBwsj':
                                rclcsbwsj = rclcsbwsj + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclcsbwsj) > 1:
                                RCLCSBwsj.append(rclcsbwsj)

                        elif dict[predicts[i][j].item()] == 'B-RCLCSZysj':
                            rclcszysj = ''
                            rclcszysj = rclcszysj + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLCSZysj':
                                rclcszysj = rclcszysj + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclcszysj) > 1:
                                RCLCSZysj.append(rclcszysj)

                        elif dict[predicts[i][j].item()] == 'B-RCLCSLqjz':
                            rclcslqjz = ''
                            rclcslqjz = rclcslqjz + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLCSLqjz':
                                rclcslqjz = rclcslqjz + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclcslqjz) > 1:
                                RCLCSLqjz.append(rclcslqjz)

                        elif dict[predicts[i][j].item()] == 'B-RCLCSLqsj':
                            rclcslqsj = ''
                            rclcslqsj = rclcslqsj + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLCSLqsj':
                                rclcslqsj = rclcslqsj + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclcslqsj) > 1:
                                RCLCSLqsj.append(rclcslqsj)

                        elif dict[predicts[i][j].item()] == 'B-RCLCSJzwdbhfw':
                            rclcsjzwdbh = ''
                            rclcsjzwdbh = rclcsjzwdbh + batch_data[i][j].word
                            if j < dim1 - 1:
                                j += 1
                            while dict[predicts[i][j].item()] == 'I-RCLCSJzwdbhfw':
                                rclcsjzwdbh = rclcsjzwdbh + batch_data[i][j].word
                                if j < dim1 - 1:
                                    j += 1
                                else:
                                    break
                            if len(rclcsjzwdbh) > 1:
                                RCLCSJzwdbhfw.append(rclcsjzwdbh)

                        # 将每句的实体按空格分隔替换进datasets/data_final.csv中（用于知识图谱的关系建立）

            # TODO: 张博确认好实体关系类型之后在这里改
            # print("approach:" + str(approach))
            # print("condition:" + str(condition))
            # print("criteria:" + str(criteria))
            # print("definition:" + str(definition))
            # print("feature:" + str(feature))
            # print("location:" + str(location))
            # print("measure" + str(measure))
            # print("part:" + str(part))
            # print("procedure:" + str(procedure))
            # print("reference:" + str(reference))
            # print("tolerance:" + str(tolerance))
            # print("tool:" + str(tool))
            print("RCLSB" + str(RCLSB))
            print("RCLSBName" + str(RCLSBName))
            print("RCLSBNo" + str(RCLSBNo))
            print("RCLSBCate" + str(RCLSBCate))
            print("RCLSBType" + str(RCLSBType))
            print("RCLSBJrfs" + str(RCLSBJrfs))
            print("RCLGZ" + str(RCLGZ))
            print("RCLGZNo" + str(RCLGZNo))
            print("RCLGZName" + str(RCLGZName))
            print("RCLGZDWG" + str(RCLGZDWG))
            print("TL" + str(TL))
            print("TLNo" + str(TLNo))
            print("TLName" + str(TLName))
            print("LQJZ" + str(LQJZ))
            print("LQJZNo" + str(LQJZNo))
            print("LQJZName" + str(LQJZName))
            print("RCLCS" + str(RCLCS))
            print("RCLCSBfjg" + str(RCLCSBfjg))
            print("RCLCSBffs" + str(RCLCSBffs))
            print("RCLCSZll" + str(RCLCSZll))
            print("RCLCSYxhd" + str(RCLCSYxhd))
            print("RCLCSBwwd" + str(RCLCSBwwd))
            print("RCLCSSwsj" + str(RCLCSSwsj))
            print("RCLCSBwsj" + str(RCLCSBwsj))
            print("RCLCSZysj" + str(RCLCSZysj))
            print("RCLCSLqjz" + str(RCLCSLqjz))
            print("RCLCSLqsj" + str(RCLCSLqsj))
            print("RCLCSJzwdbhfw" + str(RCLCSJzwdbhfw))
            print("-------------------------------------------------------------------------------------")

        # 将识别到的实体存到字典中返回
        # predict_all["approach"] = list(set(approach))
        # predict_all["condition"] = list(set(condition))
        # predict_all["criteria"] = list(set(criteria))
        # predict_all["definition"] = list(set(definition))
        # predict_all["feature"] = list(set(feature))
        # predict_all["location"] = list(set(location))
        # predict_all["measure"] = list(set(measure))
        # predict_all["part"] = list(set(part))
        # predict_all["procedure"] = list(set(procedure))
        # predict_all["reference"] = list(set(reference))
        # predict_all["tolerance"] = list(set(tolerance))
        # predict_all["tool"] = list(set(tool))
        predict_all["RCLSB"] = list(set(RCLSB))
        predict_all["RCLSBName"] = list(set(RCLSBName))
        predict_all["RCLSBNo"] = list(set(RCLSBNo))
        predict_all["RCLSBCate"] = list(set(RCLSBCate))
        predict_all["RCLSBType"] = list(set(RCLSBType))
        predict_all["RCLSBJrfs"] = list(set(RCLSBJrfs))
        predict_all["RCLGZ"] = list(set(RCLGZ))
        predict_all["RCLGZNo"] = list(set(RCLGZNo))
        predict_all["RCLGZName"] = list(set(RCLGZName))
        predict_all["RCLGZDWG"] = list(set(RCLGZDWG))
        predict_all["TL"] = list(set(TL))
        predict_all["TLNo"] = list(set(TLNo))
        predict_all["TLName"] = list(set(TLName))
        predict_all["LQJZ"] = list(set(LQJZ))
        predict_all["LQJZNo"] = list(set(LQJZNo))
        predict_all["LQJZName"] = list(set(LQJZName))
        predict_all["RCLCS"] = list(set(RCLCS))
        predict_all["RCLCSBfjg"] = list(set(RCLCSBfjg))
        predict_all["RCLCSBffs"] = list(set(RCLCSBffs))
        predict_all["RCLCSZll"] = list(set(RCLCSZll))
        predict_all["RCLCSYxhd"] = list(set(RCLCSYxhd))
        predict_all["RCLCSBwwd"] = list(set(RCLCSBwwd))
        predict_all["RCLCSSwsj"] = list(set(RCLCSSwsj))
        predict_all["RCLCSBwsj"] = list(set(RCLCSBwsj))
        predict_all["RCLCSZysj"] = list(set(RCLCSZysj))
        predict_all["RCLCSLqjz"] = list(set(RCLCSLqjz))
        predict_all["RCLCSLqsj"] = list(set(RCLCSLqsj))
        predict_all["RCLCSJzwdbhfw"] = list(set(RCLCSJzwdbhfw))


        # 新识别到的实体回到字典之中
        father = os.path.dirname
        pre_path = os.path.join(father(father(father(os.path.abspath(__file__)))), r'data', r'dict')
        # print(pre_path)
        # TODO:张博确认好实体关系类型之后在这里改
        # save2dict(list(set(approach)), os.path.join(pre_path, r'approach_dict.txt'))
        # save2dict(list(set(condition)), os.path.join(pre_path, r'condition_dict.txt'))
        # save2dict(list(set(criteria)), os.path.join(pre_path, r'criteria_dict.txt'))
        # save2dict(list(set(definition)), os.path.join(pre_path, r'definition_dict.txt'))
        # save2dict(list(set(feature)), os.path.join(pre_path, r'feature_dict.txt'))
        # save2dict(list(set(location)), os.path.join(pre_path, r'location_dict.txt'))
        # save2dict(list(set(measure)), os.path.join(pre_path, r'measure_dict.txt'))
        # save2dict(list(set(part)), os.path.join(pre_path, r'part_dict.txt'))
        # save2dict(list(set(procedure)), os.path.join(pre_path, r'procedure_dict.txt'))
        # save2dict(list(set(reference)), os.path.join(pre_path, r'reference_dict.txt'))
        # save2dict(list(set(tolerance)), os.path.join(pre_path, r'tolerance_dict.txt'))
        # save2dict(list(set(tool)), os.path.join(pre_path, r'tool_dict.txt'))
        save2dict(list(set(RCLSB)), os.path.join(pre_path, r'RCLSB_dict.txt'))
        save2dict(list(set(RCLSBName)), os.path.join(pre_path, r'RCLSBName_dict.txt'))
        save2dict(list(set(RCLSBNo)), os.path.join(pre_path, r'RCLSBNo_dict.txt'))
        save2dict(list(set(RCLSBCate)), os.path.join(pre_path, r'RCLSBCate_dict.txt'))
        save2dict(list(set(RCLSBType)), os.path.join(pre_path, r'RCLSBType_dict.txt'))
        save2dict(list(set(RCLSBJrfs)), os.path.join(pre_path, r'RCLSBJrfs_dict.txt'))
        save2dict(list(set(RCLGZ)), os.path.join(pre_path, r'RCLGZ_dict.txt'))
        save2dict(list(set(RCLGZNo)), os.path.join(pre_path, r'RCLGZNo_dict.txt'))
        save2dict(list(set(RCLGZName)), os.path.join(pre_path, r'RCLGZName_dict.txt'))
        save2dict(list(set(RCLGZDWG)), os.path.join(pre_path, r'RCLGZDWG_dict.txt'))
        save2dict(list(set(TL)), os.path.join(pre_path, r'TL_dict.txt'))
        save2dict(list(set(TLNo)), os.path.join(pre_path, r'TLNo_dict.txt'))
        save2dict(list(set(TLName)), os.path.join(pre_path, r'TLName_dict.txt'))
        save2dict(list(set(LQJZ)), os.path.join(pre_path, r'LQJZ_dict.txt'))
        save2dict(list(set(LQJZNo)), os.path.join(pre_path, r'LQJZNo_dict.txt'))
        save2dict(list(set(LQJZName)), os.path.join(pre_path, r'LQJZName_dict.txt'))
        save2dict(list(set(RCLCS)), os.path.join(pre_path, r'RCLCS_dict.txt'))
        save2dict(list(set(RCLCSBfjg)), os.path.join(pre_path, r'RCLCSBfjg_dict.txt'))
        save2dict(list(set(RCLCSBffs)), os.path.join(pre_path, r'RCLCSBffs_dict.txt'))
        save2dict(list(set(RCLCSZll)), os.path.join(pre_path, r'RCLCSZll_dict.txt'))
        save2dict(list(set(RCLCSYxhd)), os.path.join(pre_path, r'RCLCSYxhd_dict.txt'))
        save2dict(list(set(RCLCSBwwd)), os.path.join(pre_path, r'RCLCSBwwd_dict.txt'))
        save2dict(list(set(RCLCSSwsj)), os.path.join(pre_path, r'RCLCSSwsj_dict.txt'))
        save2dict(list(set(RCLCSBwsj)), os.path.join(pre_path, r'RCLCSBwsj_dict.txt'))
        save2dict(list(set(RCLCSZysj)), os.path.join(pre_path, r'RCLCSZysj_dict.txt'))
        save2dict(list(set(RCLCSLqjz)), os.path.join(pre_path, r'RCLCSLqjz_dict.txt'))
        save2dict(list(set(RCLCSLqsj)), os.path.join(pre_path, r'RCLCSLqsj_dict.txt'))
        save2dict(list(set(RCLCSJzwdbhfw)), os.path.join(pre_path, r'RCLCSJzwdbhfw_dict.txt'))


    return predict_all


def save2dict(entities, filename):
    entities = list(set(entities))
    # 这里选择w是为了只记录每次训练之后的预测结果，后续这个文件一是用来加入到word_dict.txt中，
    # 丰富自动化分需要的词典，提高下一次训练的准确性。另一个是作为预测的结果，后续要使用

    with open(filename, 'w', encoding='utf-8') as f:
        for e in entities:
            f.write(e)
            f.write('\n')
    f.close()


if __name__ == '__main__':
    predict()
