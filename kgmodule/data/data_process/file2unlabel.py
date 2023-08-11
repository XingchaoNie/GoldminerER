"""
    将下载好并转码的txt文件整合成训练文件unlabel0.txt并放在模型下
"""

import os


def train_file_gen():
    father = os.path.dirname
    filepath = os.path.join(father(father(father(father(os.path.abspath(__file__))))), r'files', r'txt')
    tar_filepath = os.path.join(father(os.path.abspath(__file__)), r'unlabel', r'unlabel0.txt')
    print(filepath)
    print(tar_filepath)
    with open(tar_filepath, 'a', encoding='utf-8') as f_in:
        for filename in os.listdir(filepath):
            file_path = os.path.join(filepath, filename)
            print(file_path)
            with open(file_path, 'r', encoding="utf-8") as f_out:
                content = f_out.readlines()
                for line in content:
                    if line not in ['\n', '\t', '\r']:
                        f_in.write(line)
            f_out.close()
    f_in.close()


if __name__ == '__main__':
    train_file_gen()