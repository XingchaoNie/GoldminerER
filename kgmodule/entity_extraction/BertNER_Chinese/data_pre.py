'''

把纯文本转换成预测所用的csv

'''

import pandas as pd


def data_gen(in_filename='data/Abstract/new.txt'):
    out_filename = 'data/Abstract/test.csv'
    data = pd.read_csv(in_filename,header=None)

    data = data.values[:,:]
    sentence = []
    lables = []
    print(type(data))
    for datum in data:
        for j in datum:
            flag = 0
            for i in j:
                flag += 1
                if(flag > 500):
                    sentence.append('sep')
                    lables.append('sep')
                    flag = 0
                sentence.append(i)
                lables.append('O')
            sentence.append('sep')
            lables.append('sep')


    dataframe = pd.DataFrame({'sentence': sentence, 'labels': lables})
    dataframe.to_csv(out_filename, index=False,sep=',')


if __name__ == '__main__':
    data_gen()


