import json
import tqdm
input_file = 'datasets/train.jsonl'
relation_output_file = 'datasets/relation.txt'
f_output = open(relation_output_file, 'w', encoding='utf-8')
relation_set = set()
with open(input_file, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        line = line.strip()
        item = json.loads(line)
        relation_set.add(item['relation'])
for relation_item in relation_set:
    f_output.write(relation_item)
    f_output.write('\n')
f_output.close()
