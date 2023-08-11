"""
NER模型自动训练
"""
import os
import warnings

warnings.filterwarnings('ignore')


def ner_cir(n=1):
    os.system('{} {}'.format('python', '../../data/data_process/file2unlabel.py'))
    for i in range(n):
        # os.system('{} {}'.format('python', '../../data/data_process/word2txt.py'))
        # 基于外部词典进行标注
        os.system('{} {}'.format('python', '../../data/data_process/NERdataset_gen.py'))
        # txt转换成csv
        os.system('{} {}'.format('python', '../../data/data_process/labeled2csv.py'))
        # 训练测试验证文件划分
        os.system('{} {}'.format('python', 'data/Abstract/data.py'))
        # 训练和将识别的实体写回字典
        os.system('{} {}'.format('python', 'main.py'))
        if i > 9:
            # 训练了10个epoch之后，我们认为这时候的模型有了一定的准确度，于是可以将分字典加到总字典中
            os.system('{} {}'.format('python', '../../data/dict/dict_transfer.py'))




if __name__ == '__main__':
    ner_cir(1)          # 可以更改这里整体的迭代训练轮次，默认是1
