import csv
from tqdm import tqdm


def tex2csv():
    out = open('../../datasets/data_all.csv', 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(out, dialect='excel')

    with open('../../data/data_process/labled/labeled0.txt', 'r', encoding='utf-8') as f1:
        lines = f1.readlines()
        csv_writer.writerow(['', 'text', 'flag'])
        text_list = []
        label_list = []
        index = 0
        cur_str = ''
        cur_lable = ''

        for line in tqdm(lines):
            # print(line)
            if line is '\n':
                cur_str = cur_str.strip()
                cur_lable = cur_lable.strip()
                text_list.append(index)
                text_list.append(cur_str)
                text_list.append(cur_lable)
                csv_writer.writerow(text_list)
                cur_str = ''
                cur_lable = ''
                text_list = []
                label_list = []
                index = index + 1
            else:
                cur_str = cur_str + line[0] # 当前文本
                cur_lable = cur_lable + line[2:-1] + ' ' # 当前lable
    f1.close()
    out.close()

if __name__ == '__main__':
    tex2csv()