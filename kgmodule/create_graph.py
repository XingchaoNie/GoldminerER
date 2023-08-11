'''
    实现对输入进来的文本，先使用ner_pred完成对其中实体的抽取，存入各个dict文件中暂存
    然后用demo_predict完成对其中关系的抽取。
    最后用两者结果构建图谱

'''

from py2neo import NodeMatcher, Graph
import json
import os

# import relation_extraction.demo_predict
from power.entity_extraction.BertNER_Chinese.utils import WordLabel, DataLoader
import torch
import numpy as np
import itertools
import power.predict as predict
from power.entity_extraction.BertNER_Chinese.models.BERT import Config, Model
from power.relation_extraction.relation_extraction import re_predict

from power.relation_extraction.relation_extraction.hparams import hparams
from db_connect import OperationMysql

# TODO 抽取PKG图谱中实体和关系，需要更改这里
r_type_dict = {
    "热处理参数:具有属性": "rechulicanshu:jysx",
    "热处理设备:具有属性": "rechulishebei:jysx",
    "unknow": "unknow",
    "热处理工装:具有属性": "rechuligongzhuang:jysx",
    "涂料:具有属性": "tuliao:jysx",
    "冷却介质:具有属性": "lengquejiezhi:jysx",
    "要求": "yaoqiu",
    "淬火": "cuihuo",
    "回火": "huihuo",
    "退火": "tuihuo",
    "固溶": "gurong",
    "时效": "shixiao",
    "采用热处理设备": "caiyongrechulishebei",
    "采用热处理工装": "caiyongrechuligongzhuang",
    "采用涂料": "caiyongtuliao",
    "采用冷却介质": "caiyonglengquejiezhi"
}

def sens2dataloader(sens):
    lab = 'O' * len(sens)
    datas = []
    datasets = []
    for ch, tag in zip(sens, lab):
        # print(WordLabel(ch, tag))
        datas.append(WordLabel(ch, tag))
    datasets.append(datas)
    dataloader = DataLoader(datasets, batch_size=2, shuffle=False)
    return dataloader


def create_node(graph, entity):
    graph.run('merge (n: Knowledge{Name: "' + str(entity) + '"}) return n')


def create_relation(graph, h_node, t_node, r):
    graph.run(
        'match (n:Knowledge), (m:Knowledge) where n.Name = "' + str(h_node) + '" and m.Name = "' + str(
            t_node) + '" merge (n)-[r:' + str(
            r) + '{Name: "' + str(r) + '"}]->(m) return r')


def get_pred_res_s(sens="机床导轨副的磨损与工作的连续性、负荷特性、工作条件、导轨的材质和结构等有关。"):
    '''
    抽取句子sens中的实体和关系，将结果存到中间表中。
    Args:
        sens:

    Returns:
        os.path.dirname(os.path.abspath(__file__)) +
    '''
    data_iter = sens2dataloader(sens)
    dataset = os.path.dirname(os.path.abspath(__file__)) + '/entity_extraction/BertNER_Chinese/data/Abstract/'
    config = Config(dataset)

    np.random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True  # 保证每次运行结果一样

    print('识别实体')
    print('加载模型')

    model = Model(config).to(config.device)
    entities = predict.predict(config, model, data_iter)
    print('实体预测完毕')
    print(entities)

    all_rdfs, all_types = [], []
    # 关系抽取部分
    # 针对特定实体之间进行抽取
    print("预测实体间的关系")
    rdfs1, types1 = extract_relation_between_entities_s(sens, entities, "part", "tool")
    rdfs2, types2 = extract_relation_between_entities_s(sens, entities, "part", "criteria")

    all_rdfs = all_rdfs + rdfs1 + rdfs2
    all_types = all_types + types1 + types2
    return all_rdfs, all_types


def extract_relation_between_entities_s(sens, entities, h_type, t_type):
    """
    查找给定两个实体类型之间的关系
    Args:
        sens:
        entities:
        h_type:
        t_type:
    Returns:返回查找到的三元组rdfs和对应的类型types

    """

    # 首先组合头尾实体对
    head_tail = []
    for h_node in entities[h_type]:
        for t_node in entities[t_type]:
            if h_node != t_node:
                head_tail.append([h_node, t_node])

    # 预测关系并存入数据库
    rdfs = []
    types = []

    for h_t in head_tail[:10]:
        head_node = h_t[0]
        tail_node = h_t[1]
        relation = re_predict.predict(hparams, sens, head_node, tail_node)
        t_rdf = {
            "h_node": head_node,
            "t_node": tail_node,
            "relation": relation
        }
        rdfs.append(t_rdf)
        # print(head_node, tail_node)
        # 每识别一个就将其保存成三元组
        t_type = {
            'h_type': h_type,
            't_type': t_type,
            'r_type': r_type_dict[relation]
        }
        types.append(t_type)

    return rdfs, types


def extract_relation_between_entities(SysNo, sens, entities, h_type, t_type):
    """
    查找给定两个实体类型之间的关系
    Args:
        SysNo:
        sens:
        entities:
        h_type:
        t_type:
    Returns:返回查找到的三元组rdfs和对应的类型types

    """

    # 首先组合头尾实体对
    head_tail = []
    for h_node in entities[h_type]:
        for t_node in entities[t_type]:
            if h_node != t_node:
                head_tail.append([h_node, t_node])

    # 预测关系并存入数据库
    rdfs = []
    types = []

    for h_t in head_tail[:10]:
        head_node = h_t[0]
        tail_node = h_t[1]
        relation = re_predict.predict(hparams, sens, head_node, tail_node)
        t_rdf = {
            "h_node": head_node,
            "t_node": tail_node,
            "relation": relation
        }
        rdfs.append(t_rdf)
        # print(head_node, tail_node)
        # 每识别一个就将其保存成三元组
        t_type = {
            'h_type': h_type,
            't_type': t_type,
            'r_type': r_type_dict[relation]
        }
        types.append(t_type)

    # 存入数据库, 这里的数据库中的表需要改成真实中间表
    op_sql = OperationMysql()
    for t_s_rdf, t_s_type in zip(rdfs, types):
        print(t_s_rdf, t_s_type)
        sql = """
            INSERT into alm.talm_fusion_data_info(SEQUENCE, RDFS, TYPES) VALUES ('%s', '%s', '%s')
            """ % (SysNo, json.dumps(t_s_rdf), json.dumps(t_s_type))
        op_sql.cur.execute(sql)
        op_sql.conn.commit()
    op_sql.cur.close()
    op_sql.conn.close()

    return rdfs, types


def get_pred_res_1(SysNo, sens="机床导轨副的磨损与工作的连续性、负荷特性、工作条件、导轨的材质和结构等有关。"):
    '''
    抽取句子sens中的实体和关系，将结果存到中间表中。
    Args:
        sens:

    Returns:
        os.path.dirname(os.path.abspath(__file__)) +
    '''
    data_iter = sens2dataloader(sens)
    dataset = os.path.dirname(os.path.abspath(__file__)) + '/entity_extraction/BertNER_Chinese/data/Abstract/'
    config = Config(dataset)

    np.random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True  # 保证每次运行结果一样

    print('识别实体')
    print('加载模型')

    model = Model(config).to(config.device)
    entities = predict.predict(config, model, data_iter)
    print('实体预测完毕')
    print(entities)

    all_rdfs, all_types = [], []
    # 关系抽取部分
    # 针对特定实体之间进行抽取
    print("预测实体间的关系")
    rdfs, types = extract_relation_between_entities(SysNo, sens, entities, "part", "tool")

    all_rdfs = all_rdfs + rdfs
    all_types = all_types + types
    return all_rdfs, all_types


def extract_relation_between_entities_2(SysNo, sens, entities, h_type, t_type, r_type):
    """
    查找给定两个实体类型之间特定关系是否存在
    Args:
        SysNo:
        sens:
        entities:
        h_type:
        t_type:
    Returns:返回查找到的三元组rdfs和对应的类型types

    """

    # 首先组合头尾实体对
    head_tail = []
    for h_node in entities[h_type]:
        for t_node in entities[t_type]:
            if h_node != t_node:
                head_tail.append([h_node, t_node])

    # 预测关系并存入数据库
    rdfs = []
    types = []

    for h_t in head_tail[:10]:
        head_node = h_t[0]
        tail_node = h_t[1]
        relation = re_predict.predict(hparams, sens, head_node, tail_node)
        if r_type == relation:
            t_rdf = {
                "h_node": head_node,
                "t_node": tail_node,
                "relation": relation
            }
            rdfs.append(t_rdf)
            # print(head_node, tail_node)
            # 每识别一个就将其保存成三元组
            t_type = {
                'h_type': h_type,
                't_type': t_type,
                'r_type': r_type_dict[relation]
            }
            types.append(t_type)
            break

    return rdfs, types



def get_pred_res_2(sens="机床导轨副的磨损与工作的连续性、负荷特性、工作条件、导轨的材质和结构等有关。"):
    '''
    抽取句子sens中的实体和关系，将结果存到中间表中。
    Args:
        sens:

    Returns:
        os.path.dirname(os.path.abspath(__file__)) +
    '''
    data_iter = sens2dataloader(sens)
    dataset = os.path.dirname(os.path.abspath(__file__)) + '/entity_extraction/BertNER_Chinese/data/Abstract/'
    config = Config(dataset)

    np.random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True  # 保证每次运行结果一样

    print('识别实体')
    print('加载模型')

    model = Model(config).to(config.device)
    entities_relations = predict.predict(config, model, data_iter)
    print('实体关系预测完毕')
    print(entities_relations)

    return entities_relations

def create_entity_rclcs(rdfs_rclcs, types_rclcs):
    """
    创建热处理参数实体的rdfs和types
    Args:
        rdfs_rclcs:
        types_rclcs:

    Returns:

    """

if __name__ == '__main__':

    sens = '由于在加工过程中产生了切削力、切削热和摩擦，它们将引起工艺系统的受力变形、受热变形和磨损，这些都会影响在调整时所获得的工件与刀具之间的相对位置，造成各种加工误差。这类在加工过程中产生的原始误差称为工艺系统的动误差。'
    # get_pred_res_1(123, sens)
    # get_pred_res_s(sens)
    print(get_pred_res_2(sens))

