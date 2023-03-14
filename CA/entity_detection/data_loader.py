"""
加入了BE标签的位置
"""
import json
import torch
import tqdm as tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
# from gensim.models import KeyedVectors
from transformers import AutoTokenizer
import os
import utils
import tqdm
from tqdm import trange
import requests
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    NEG = 'neg'
    def __init__(self):
        # self.label2id = {self.NEG: 0, "veh": 1, "loc": 2, "wea": 3, "gpe": 4, "per": 5, "org": 6,'fac':7 }
        # self.id2label = {0: self.NEG, 1: "veh", 2: "loc", 3: "wea", 4:"gpe", 5: "per", 6: "org", 7:'fac'}
        self.label2id = {self.NEG: 0}
        self.id2label = {0: self.NEG}
    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        self.add_label(label)
        return self.label2id[label]

    def get_label(self,type):
        if type == 2:
            return list(range(1,len(self.label2id),1))
        else:
            return [1]
    def get_label_name(self):
        return self.label2id

class TESTVocabulary(object):
    NEG = 'neg'
    def __init__(self):

        self.label2id = {self.NEG: 0}
        self.id2label = {0: self.NEG}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label
        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def get_label(self,type):
        if type == 2:
            return list(range(1,len(self.label2id),1))
        else:
            return [1]

    def get_label_name(self):
        return self.label2id

def collate_fn(data):
    bert_inputs, grid_labels,pieces2word, sent_length = map(list, zip(*data))

    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    grid_labels = torch.LongTensor(grid_labels)

    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, grid_labels, pieces2word,  sent_length


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, pieces2word, sent_length):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.pieces2word = pieces2word
        self.sent_length = sent_length

    def __getitem__(self, item):
        t = torch.LongTensor(self.bert_inputs[item]), \
               self.grid_labels[item], \
               torch.LongTensor(self.pieces2word[item]), \
               self.sent_length[item]
        return t

    def __len__(self):
        return len(self.bert_inputs)


def process_bert(data, tokenizer, vocab):

    bert_inputs = []
    grid_labels = []
    pieces2word = []
    sent_length = []

    with trange(len(data['sentence'])) as loader:
        for index,instance,label,locate in zip(loader,data['sentence'],data['label'],data['index']):
            if len(instance) == 0:
                continue
            tokens = [tokenizer.tokenize(word) for word in instance]
            pieces = [piece for pieces in tokens for piece in pieces]
            _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
            _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

            length = len(instance)

            _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)

            if tokenizer is not None:
                start = 0
                for i, pieces in enumerate(tokens):
                    if len(pieces) == 0:
                        continue
                    pieces = list(range(start, start + len(pieces)))
                    _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                    start += len(pieces)

            grid_labels.append(vocab.label_to_id(label))

            sent_length.append(length)
            bert_inputs.append(_bert_inputs)
            pieces2word.append(_pieces2word)

    return bert_inputs, grid_labels, pieces2word, sent_length

def fill_vocab(vocab, dataset):
    data = {
        'sentence':[],
        'label':[],
        'index':[]
    }
    txt,entity_num,count = [],0,0
    indexs = []
    multi_label = ''
    with trange(len(dataset)) as loader:
        for index,line in zip(loader,dataset):
            line = line.strip()
            if len(line) == 0:
                data['sentence'].append(txt)
                data['label'].append(multi_label)
                data['index'].append(indexs)
                count = 0
                indexs = []
                txt = []
                multi_label = ''
            else:
                count+=1
                if line in ["\n",'']:
                    continue
                word, label = line.split("\t")
                txt.append(word)
                if label != "O":
                    vocab.add_label(label)
                    multi_label = label
                    indexs.append(count)

    for num in data['label']:
        if num !='NEG':
            entity_num +=1
    return entity_num,data


def load_data_bert(config,types,result):
    with open('../Data/{}/add_cues_data/train.jsons'.format(config.dataset), 'r', encoding='utf-8') as f:
        train_data = f.readlines()
    # with open('./data/{}/dev.jsons'.format(config.dataset), 'r', encoding='utf-8') as f:
    #     dev_data = f.readlines()
    with open('../Data/{}/add_cues_data/test.jsons'.format(config.dataset), 'r', encoding='utf-8') as f:
        test_data = f.readlines()
    with open('../Data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        true_data = f.readlines()

    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="../cache/")
    tokenizer.add_special_tokens({'additional_special_tokens': ["[B]", "[E]"]})

    vocab = Vocabulary()

    train_ent_num,train_dataset = fill_vocab(vocab, train_data)
    # dev_ent_num ,dev_dataset= fill_vocab(vocab, dev_data)
    test_ent_num ,test_dataset= fill_vocab(vocab, test_data)
    true_ent_num = load_true_data_jsons(true_data)

    table = pt.PrettyTable([config.dataset, 'sentences', 'entities'])
    table.add_row(['train', len(train_dataset['sentence']), train_ent_num])
    # table.add_row(['dev', len(dev_dataset['sentence']), dev_ent_num])
    table.add_row(['test', len(test_dataset['sentence']), test_ent_num])
    config.logger.info("\n{}".format(table))

    config.label_num = len(vocab.label2id)
    config.vocab = vocab

    train_dataset = RelationDataset(*process_bert(train_dataset, tokenizer, vocab))
    # dev_dataset = RelationDataset(*process_bert(dev_dataset, tokenizer, vocab))
    test_dataset = RelationDataset(*process_bert(test_dataset, tokenizer, vocab))
    return [train_dataset, test_dataset],true_ent_num,tokenizer

def load_true_data_jsons(dataset):
    pre_true = 0
    with trange(len(dataset)) as fr:
        for index,line in zip(fr,dataset):
            jsn = json.loads(line.strip())
            for sen in jsn:
                pre_true += len(sen['ner'])
    print("原实体个数：{}".format(pre_true))
    return pre_true