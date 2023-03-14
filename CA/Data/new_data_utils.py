import os
import json
import random
import logging
from tqdm import tqdm
from random import sample
from collections import Counter

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename='data_process.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

def load_entity(BIO_path):
    """
    获得实体信息和标签信息
    BIO_PATH:信息文件地址
    :return:
    """
    entity_list, entity_type, sentences_seq = [], [], []
    temp_e,temp_t = [],[]
    with open(BIO_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            jsn = json.loads(line.strip())

            for word in jsn:
                sentences_seq.append(word['sentence'])
                for indexs in word['ner']:
                    temp_e.append([indexs['index'][0],indexs['index'][-1]])
                    temp_t.append(indexs['type'])
                entity_list.append(temp_e)
                entity_type.append(temp_t)
                temp_e, temp_t = [], []
    return entity_list, entity_type, sentences_seq


def loda_enumeration_data(sentences_seq):
    """
    获得句子所有枚举长度信息  长度不受限制
    :param word_seq: 文本序列
    :return: 实体的下标集合
    """
    all_entity_list = []
    for sentence in sentences_seq:
        sen_list = []
        L = list(range(len(sentence)))
        for i,first in enumerate(L):
            for j,second in enumerate(L):
                # if j < i or abs(i - j) >= len(L):
                if j < i or abs(i - j) >= 7:
                    continue
                sub_list = [i,j]
                sen_list.append(sub_list)
        all_entity_list.append(sen_list)
    return all_entity_list

def data_boundary_mapping(B_path,E_path,enu_entity_list,istest):
    """
    使用边界信息去标注实体
    :param B_path: 开始边界
    :param E_path: 结束边界
    :param enu_entity_list:所有枚举实体
    :return:
    """
    # 边界标注数据集的处理
    result_list = []
    B_line = open(B_path, "r", encoding="utf-8").read()
    E_line = open(E_path, "r", encoding="utf-8").read()

    # 开头和结束标签列表
    B_label = [[B_word.split("\t")[-1] for B_word in B_sentence.strip().split("\n")] for B_sentence in
               B_line.strip().split("\n" + "\n")]

    E_lable = [[E_word.split("\t")[-1] for E_word in E_sentence.strip().split("\n")] for E_sentence in
               E_line.strip().split("\n" + "\n")]
    #边界匹配过滤实体
    with tqdm(B_label) as loader:
        for indexs, (begin, end, sentence) in enumerate(zip(loader, E_lable, enu_entity_list)):
            entity_list = []
            if not sentence:
                result_list.append(entity_list)
            else:
                for i, lableb in enumerate(begin):
                    if lableb == 'O':
                        continue
                    for j,lablee in enumerate(end):
                        # if "B" == lableb :
                        if "B" == lableb and "B" == lablee:
                            if i>j:
                                continue
                            if [i, j] not in entity_list:
                                entity_list.append([i, j])
                for i, lable in enumerate(zip(begin, end)):
                    if "B" == lable[0]:
                        for entity in sentence:
                            if entity[0] == i:
                                if entity not in entity_list:
                                    entity_list.append(entity)
                            if entity[0] > i:
                                break
                    # if "B" == lable[1]:
                    #     for entity in sentence:
                    #         if entity[1] == i:
                    #             if entity not in entity_list:
                    #                 entity_list.append(entity)
                    #         if entity[1] > i:
                    #             continue
                result_list.append(entity_list)
    return result_list


def process_boundary_data(entity_list, entity_type, sentences_seq, boundary_data_list,final_path):
    """
    标记B/E
    :param entity_list: 正确的实体信息
    :param entity_type: 实体标签
    :param sentences_seq: 句子序列信息
    :param boundary_data_list: 边界识别后的实体
    :param final_path: 处理数据保存路径
    :return:
    """

    if os.path.exists(final_path):
        os.remove(final_path)
    with open(final_path,'w+',encoding='utf-8',errors="ignore") as fs:
        with tqdm(sentences_seq) as loader:
            for i,sentences in enumerate(loader):
                #标记正例列表中的真实实体和候选实体
                for entitys in boundary_data_list[i]:
                    if entitys in entity_list[i]:
                        label = entity_type[i][entity_list[i].index(entitys)]
                    else:
                        label = 'NEG'
                    if entitys[0] > entitys[1]:
                        continue
                    for j, char in enumerate(sentences):
                        if entitys[0] <= j <= entitys[1]:
                            if entitys[0] == j:
                                fs.write("[B]" + "\t" + "O" + "\n")
                            fs.write(char + "\t" + label + "\n")
                            if entitys[1] == j:
                                fs.write("[E]" + "\t" + "O" + "\n")
                        else:
                            fs.write(char + "\t" + "O" + "\n")
                    fs.write("\n")
    fs.close()

def performance(boundary_data_list, entity_list,type):
    """
    boundary_data_list, entity_list
    :param pre_list: boundary_data_list
    :param test_list: entity_list
    :return:
    """
    true_num = 0  # 真正的实体数目
    pre_true = 0  # 枚举结果中认为是实体的数目
    test_true_num = 0  # 真正集中的正例数目

    for test_node in entity_list:
        test_true_num += len(test_node)

    for pre_node in boundary_data_list:
        pre_true += len(pre_node)
    num=[]
    for true_node_child, pre_node_child in zip(entity_list, boundary_data_list):
        for test in true_node_child:
            if test in pre_node_child:
                num.append(test[1]-test[0])
                true_num += 1

    print("-------------------------*", type, "*---------------------------------")
    print("|数据集集中的正例的数目: " + str(test_true_num))
    print("|枚举结果中认为是正例的数目:" + str(pre_true))
    print("|预测为正例的结果中真正的正例数目:" + str(true_num))
    logging.info("数据集中的正例的数目: %s,预测结果中认为是正例的数目:%s,预测为正例的结果中真正的正例数目：%s",
                 str(test_true_num),str(pre_true),str(true_num))
    P = 0 if pre_true == 0 else 100. * true_num / pre_true
    R = 0 if test_true_num == 0 else 100. * true_num / test_true_num
    F = 0 if P + R == 0 else 2 * P * R / (P + R)
    print("|Precision: %.2f" % (P), "%")
    print("|Recall: %.2f" % (R), "%")
    print("|F1: %.2f" % (F), "%")
    logging.info("|Precision: %.2f,|Recall: %.2f,|F1: %.2f",P,R,F)
    logging.info("\n")
    print()

def load_boundray_data(init_path,final_path,B_path,E_path,type,t):

    entity_list, entity_type, sentences_seq = load_entity(init_path)
    # 获得无句长限制的枚举实体
    enu_entity_list = loda_enumeration_data(sentences_seq)
    boundary_data_list = data_boundary_mapping(B_path,E_path,enu_entity_list,False)
    process_boundary_data(entity_list, entity_type, sentences_seq, boundary_data_list,
                 final_path)
    #查看边界识别性能
    performance(boundary_data_list, entity_list,type)


def numerate_boundary_data(type):
    init_paths = {
        "train": "genia/true_data/train.json",
        "dev": "genia/true_data/dev.json",
        "test": "genia/true_data/test.json"
    }
    sava_paths = {
        "train": "genia/train.jsons",
        "dev": "genia/dev.json",
        "test": "genia/test.json",
    }
    B_paths = {
        "train": "genia/boundary_data/train/B.jsons",
        "dev": "genia/boundary_data/dev/B.jsons",
        "test": "genia/boundary_data/test/pred_B.txt",
    }
    E_paths = {
        "train": "genia/boundary_data/train/E.jsons",
        "dev": "genia/boundary_data/dev/E.jsons",
        "test": "genia/boundary_data/test/pred_E.txt",
    }
    for idx, data_path, sava_path, B_path, E_path in zip(range(len(init_paths)),
                                                         init_paths.items(), sava_paths.items(),
                                                         B_paths.items(), E_paths.items()):
        logging.info("%s数据开始加载", data_path[0])
        print("{}数据开始加载".format(data_path[0]))
        load_boundray_data(data_path[1], sava_path[1], B_path[1], E_path[1], data_path[0],type)

if __name__ == '__main__':
    types = ['ACE_chinese','Bio','ACE_Eng','resume']
    type=types[0]
    logging.info("%s数据开始加载", type)
    numerate_boundary_data(type)