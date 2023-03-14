import argparse
import torch
import os
from transformers import BertConfig, BertTokenizer, BertModel, \
    RobertaConfig, RobertaTokenizer, RobertaModel, AutoModel, AutoConfig, AutoTokenizer,AlbertModel


_GLOBAL_ARGS = None


def get_args_parser():
    parser = argparse.ArgumentParser()
    #实体分类模块
    parser.add_argument('--config', type=str, default='./config/CN-zh-y.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)
    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)
    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")
    parser.add_argument('--seed', type=int,default=47)

    args = parser.parse_args()

    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    return args


def get_args():
    return get_args_parser()

