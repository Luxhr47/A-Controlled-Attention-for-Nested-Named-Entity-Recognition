import argparse

import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange
import config
import data_loader
import utils
from model import Model


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        with tqdm((data_loader)) as loader:
            loader.set_description('train')
            for data_batch in loader:
                data_batch = [data.cuda() for data in data_batch]

                bert_inputs, grid_labels, pieces2word, sent_length = data_batch

                outputs = model(bert_inputs,pieces2word, sent_length)

                loss = self.criterion(outputs, grid_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss_list.append(loss.cpu().item())
                self.scheduler.step()


    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        pred_result = []
        label_result = []

        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                data_batch = [data.cuda() for data in data_batch]

                bert_inputs, grid_labels, pieces2word, sent_length = data_batch

                outputs = model(bert_inputs, pieces2word, sent_length)
                grid_labels = grid_labels

                label_result.append(grid_labels)
                pred_result.append(outputs)

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(),
                                                      pred_result.cpu().numpy(),
                                                      average="micro")

        print("**********************DEV-DATA******************************************")
        print('[* Pi: {:1.4f} \t Ri:{:1.4f} \t Fi:{:1.4f} *]'.format(p, r, f1))
        print('[* Pi: {:1.4f} \t Ri:{:1.4f} \t Fi:{:1.4f} *]'.format(p, r, f1))
        print("************************************************************************")


    def predict(self, data_loader,true_ent_num):
        self.model.eval()

        pred_result = []
        label_result = []

        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                data_batch = [data.cuda() for data in data_batch]

                bert_inputs, grid_labels, pieces2word, sent_length = data_batch
                outputs = model(bert_inputs, pieces2word, sent_length)
                outputs = torch.argmax(outputs, -1)
                grid_labels = grid_labels.contiguous().view(-1)
                outputs = outputs.contiguous().view(-1)

                label_result.append(grid_labels)
                pred_result.append(outputs)

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        true_enetitys_num = true_ent_num
        TP, FN, FP, TN = 0, 0, 0, 0
        for idx, true, pred in zip(range(len(label_result)), label_result.tolist(), pred_result.tolist()):
            if true != 0:  # 实际为真
                if pred == true:
                    TP += 1
                else:
                    FN += 1
            else:  # s实际为假
                if true != pred:
                    FP += 1
                else:
                    TN += 1
        logger.info('TP:{}\t FP:{}\t FN:{}\t TN:{}\t'.format(TP, FP, FN, TN))
        try:
            p = 1.0 * TP / (TP + FP)
            r = 1.0 * TP / true_enetitys_num
            score = 2 * p * r / (p + r) if p + r > 0 else 0
        except:
            p = 0
            r = 0
            score = 0.1

        logger.info("************************************************************************")
        logger.info('[* P: {:1.4f} \t R:{:1.4f} \t F:{:1.4f} *]'.format(p, r, score))
        logger.info("************************************************************************")
        target_name = config.vocab.get_label_name()
        report = metrics.classification_report(label_result.cpu().tolist(), pred_result.cpu().tolist(), digits=4,target_names=target_name)
        logger.info("\n{}".format(report))
        return score

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/genia.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
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

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    config = config.Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # random.seed(config.seed)
    # np.random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # torch.cuda.manual_seed(config.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


    logger.info("Loading Data")
    datasets,true_ent_num,tokenizer = data_loader.load_data_bert(config,"jsons",'multi')

    train_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle=i == 0,
                   drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(datasets[0]) // config.batch_size * config.epochs

    logger.info("Building Model")
    model = Model(config,tokenizer)
    # model.load_state_dict(torch.load("bio.pt"))
    model = model.cuda()

    trainer = Trainer(model)

    best_f1 = 0
    best_test_f1 = 0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)
        # trainer.eval(i, dev_loader)
        test_f1 = trainer.predict(test_loader, true_ent_num)
        if test_f1 > best_f1:
            best_f1 = test_f1
            trainer.save("bio.pt")
    logger.info("Best TEST F1: {:3.4f}".format(best_f1))
    trainer.load("bio.pt")
    trainer.predict("Final", test_loader,true_ent_num, True)
