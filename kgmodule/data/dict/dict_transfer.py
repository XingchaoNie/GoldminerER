import os


def dict_trans(filename):
    father = os.path.dirname
    pref_path = father(os.path.abspath(__file__))
    t_filename = filename[:-5]
    filename = filename + '.txt'
    filename = os.path.join(pref_path, filename)
    # rewrite_lines = []
    with open(filename, 'r', encoding='utf-8') as f1:
        lines = f1.readlines()
        lines = list(set(lines))
        # rewrite_lines = lines
        tar_path = os.path.join(father(pref_path), 'data_process', 'word_dict.txt')
        with open(tar_path, 'a', encoding='utf-8') as f2:
            for line in lines:
                line = line.strip()
                line = line + ' ' + t_filename + '\n'
                # print(line)
                f2.write(line)
        f2.close()
    f1.close()

    # # 将字典中重复的部分全部合并
    # with open(filename, 'w', encoding='utf-8') as f3:
    #     for w_line in rewrite_lines:
    #         f3.write(w_line)
    #         f3.write('\n')
    # f3.close()



if __name__ == '__main__':
    dict_trans('approach_dict')
    dict_trans('condition_dict')
    dict_trans('criteria_dict')
    dict_trans('definition_dict')
    dict_trans('feature_dict')
    dict_trans('location_dict')
    dict_trans('measure_dict')
    dict_trans('part_dict')
    dict_trans('procedure_dict')
    dict_trans('reference_dict')
    dict_trans('tolerance_dict')
    dict_trans('tool_dict')

