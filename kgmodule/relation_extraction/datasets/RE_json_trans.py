"""
doccano 标注数据转换成模型输入格式
"""

import json
from tqdm import tqdm
import random


def find_pos(target, text):
    s_pos = text.find(target)
    e_pos = s_pos + len(target)
    return s_pos, e_pos


def construct_node(node):
    pos = [node['s_idx'], node['e_idx']]
    name = node['name']
    res_n = {}
    res_n['name'] = name
    res_n['pos'] = pos
    return res_n


#
def trans_data(filename):
    with open(filename, 'r', encoding='utf-8') as f1:
        json_str = f1.read()
        # print(json_str)
        data_json = json.loads(json_str)
        sentences = {}
        entities = {}
        relations = []

        # 文章拆成句子 这里注意all.json最后一句话没有换行符，所以在all.json手动加上。
        t_text = ''
        for idx, ch in enumerate(data_json['text']):
            if ch != '\n':
                t_text = t_text + ch
            else:
                sentences[idx] = t_text  # 这个idx是每句话结束时的换行符的位置
                t_text = ''
        print(len(sentences[20]))
        # 处理实体
        all_entities = data_json['entities']
        for entity in tqdm(all_entities):
            t_entity = {}
            name = data_json['text'][entity['start_offset']:entity['end_offset']]
            pos = entity['end_offset']
            while pos < len(data_json['text']) and data_json['text'][pos] != '\n':
                pos = pos + 1
            id = entity['id']
            t_entity['name'] = name
            t_entity['id'] = id
            t_entity['text'] = sentences[pos]
            s_idx, e_idx = 0, 0
            s_idx, e_idx = find_pos(t_entity['name'], t_entity['text'])  # 查找实体名称在句子中的位置，即新位置
            t_entity['s_idx'] = s_idx
            t_entity['e_idx'] = e_idx
            entities[t_entity['id']] = t_entity  # 实体的id作为key

        # 处理关系
        all_relations = data_json['relations']
        for relation in tqdm(all_relations):
            t_relation = {}
            h_node = construct_node(entities[relation['from_id']])
            t_node = construct_node(entities[relation['to_id']])
            r_name = relation['type']
            r_text = entities[relation['from_id']]['text']
            t_relation['h'] = h_node
            t_relation['t'] = t_node
            t_relation['relation'] = r_name
            t_relation['text'] = r_text
            relations.append(t_relation)
            # relations.append(',')

        # 导出文件保存
        out_filename1 = 'train_2.jsonl'
        out_filename2 = 'val_2.jsonl'
        num_a = len(relations)
        indices = list(range(len(relations)))
        random.shuffle(indices)
        print(indices)
        with open(out_filename1, 'w', encoding='utf-8') as ft:
            with open(out_filename2, 'w', encoding='utf-8') as fv:
                for id, idx in enumerate(indices, 0):
                    res = json.dumps(relations[idx], ensure_ascii=False)
                    if id < 7 * num_a / 10:
                        ft.write(res)
                        ft.write('\n')
                    else:
                        fv.write(res)
                        fv.write('\n')
            fv.close()
        ft.close()
    f1.close()


if __name__ == '__main__':
    filename = 'all.json'
    trans_data(filename)
