import argparse
import math
import os
import random
from datetime import time

import numpy as np
import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig
from transformers import BertTokenizer, BertModel
from transformers import AlbertConfig, AlbertTokenizer, AlbertModel
from transformers import AlbertPreTrainedModel

from model import models
from model import sampling
from model import util
from model.entities import Dataset
from model.evaluator import Evaluator
from model.input_reader import JsonInputReader, BaseInputReader
from model.loss import SpERTLoss, Loss, Focal_loss
from tqdm import tqdm
from model.trainer import BaseTrainer

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def seed_torch(seed=128):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SpERTTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.best_f1 = 0
        seed_torch(args.seed)  # 以固定种子初始化

        # byte-pair encoding
        # self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
        #                                                 do_lower_case=args.lowercase,
        #                                                 cache_dir=args.cache_path)

        # ALBERT
        if args.bert_type == 'albert':
            self._tokenizer = AlbertTokenizer(args.tokenizer_path + 'spiece.model')
            config = AlbertConfig.from_pretrained(self.args.model_path)
            self.bert_config = config

        # BERT
        if args.bert_type == 'bert':
            config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
            self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                            do_lower_case=False,
                                                            cache_dir=args.cache_path)
            self.bert_config = config

        # path to export predictions to
        self._predictions_path = os.path.join(self._log_path, 'predictions_%s_epoch_%s.json')

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("MFodel type: %s" % args.model_type)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, args.neg_entity_count,
                                        args.neg_boundary_count, args.max_span_size, self._logger,
                                        args.boundary_represent_mode, args.BD_include_type, self.args.detect_upper,
                                        self.args.detect_entity_token)
        input_reader.read({train_label: train_path, valid_label: valid_path})
        self._log_datasets(input_reader)

        train_dataset = input_reader.get_dataset(train_label)

        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        validation_dataset = input_reader.get_dataset(valid_label)

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # create model
        model_class = models.get_model(self.args.model_type)  # spbert

        # load model
        # config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        # config = BertConfig.from_pretrained(self.args.model_path)

        config = self.bert_config
        util.check_version(config, model_class, self.args.model_path)

        config.spert_version = model_class.VERSION
        # bert_model = BertModel.from_pretrained(self.args.model_path)
        model = model_class.from_pretrained(self.args.model_path,
                                            config=config,
                                            # SpERT model parameters
                                            tokenizer=self._tokenizer,
                                            entity_types=input_reader.entity_type_count,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            boundary_filter_threshold=self.args.boundary_filter_threshold,
                                            boundary_represent_mode=self.args.boundary_represent_mode,
                                            use_sent_ctx=self.args.use_sent_ctx,
                                            detect_boundary=self.args.detect_boundary,
                                            use_size_embedding=self.args.use_size_embedding,
                                            BD_include_type=self.args.BD_include_type,
                                            dataset=train_dataset,
                                            detect_upper=self.args.detect_upper,
                                            detect_entity_token=self.args.detect_entity_token,
                                            bert_type=self.args.bert_type,
                                            max_span_size=self.args.max_span_size
                                            )

        # model = DataParallel(model, device_ids=[0, 1])

        # SpERT is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train SpERT on multiple GPUs, uncomment the following lines
        # # parallelize model
        # if self._device.type != 'cpu':
        #     model = torch.nn.DataParallel(model)

        model.to(self._device)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)

        # optimizer = AdamW(param_groups, weight_decay=args.weight_decay, correct_bias=False)

        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)

        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # create loss function
        bound_criterion = torch.nn.BCELoss(reduction='none')
        # bound_criterion = Focal_loss(alpha=0.25, gamma=2)
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        compute_loss = SpERTLoss(bound_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)

        # eval validation set
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        # train
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch, validation_dataset,
                              input_reader)

            # eval validation sets
            if not args.final_eval or (epoch == args.epochs - 1):
                if (epoch + 1) >=20:
                    self._eval(model, validation_dataset, input_reader, epoch, updates_epoch)

        # save final model
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        self._save_model(self.bert_config, self._save_path, model, self._tokenizer, global_iteration,
                         optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                         include_iteration=False, name='final_model')

        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)
        self._close_summary_writer()

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size, logger=self._logger,
                                        boundary_present_mode=args.boundary_represent_mode,
                                        detect_upper=self.args.detect_upper,
                                        detect_entity_token=self.args.detect_entity_token
                                        )
        input_reader.read({dataset_label: dataset_path})
        eval_dataset = input_reader.get_dataset(dataset_label)
        self._log_datasets(input_reader)

        # create model
        model_class = models.get_model(self.args.model_type)

        # config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        # util.check_version(config, model_class, self.args.model_path)
        # config = AlbertConfig.from_pretrained(self.args.model_path)

        model = model_class.from_pretrained(self.args.model_path,
                                            config=self.bert_config,
                                            # SpERT model parameters
                                            tokenizer=self._tokenizer,
                                            entity_types=input_reader.entity_type_count,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            boundary_filter_threshold=self.args.boundary_filter_threshold,
                                            boundary_represent_mode=self.args.boundary_represent_mode,
                                            use_sent_ctx=self.args.use_sent_ctx,
                                            detect_boundary=self.args.detect_boundary,
                                            use_size_embedding=self.args.use_size_embedding,
                                            BD_include_type=self.args.BD_include_type,
                                            dataset=eval_dataset,
                                            detect_upper=self.args.detect_upper,
                                            detect_entity_token=self.args.detect_entity_token,
                                            bert_type=self.args.bert_type,
                                            max_span_size=self.args.max_span_size
                                            )
        # model.load_state_dict(torch.load(args.model_path + 'model.pkl'))

        model.to(self._device)

        # evaluate
        self._eval(model, eval_dataset, input_reader)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int, validation_dataset: Dataset, input_reader: JsonInputReader):
        self._logger.info("Train epoch: %s" % epoch)

        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True, drop_last=True,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        model.zero_grad()

        iteration = 0
        i = 0
        total = dataset.document_count // self.args.train_batch_size
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            # i += 1
            # if i // 1504:
            #     i = i
            # else:
            #     continue

            sentence_len = batch['encodings'].shape[1]
            entity_num = batch['entity_sizes'].shape[1]
            model.train()
            batch_doc = batch.pop('doc')
            batch = util.to_device(batch, self._device)

            # forward step
            # entity_logits, rel_logits = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
            #                                   entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
            #                                   relations=batch['rels'], rel_masks=batch['rel_masks'])
            entity_logits, boundary_logits, entity_token_logits, upper_logits = \
                model(encodings=batch['encodings'],
                      context_masks=batch['context_masks'],
                      entity_masks=batch['entity_masks'],
                      entity_sizes=batch['entity_sizes'],
                      entity_boundary_masks=batch['entity_boundary_masks'],
                      pos_tags=batch['pos_tags'],
                      word_masks=batch['word_masks'],
                      entity_boundaries_idx=batch['entity_boundaries_idx'],
                      ori_entity_masks=batch['ori_entity_masks'],
                      char_encodings=batch['char_encodings'],
                      char_masks=batch['char_masks'],
                      word_encodings=batch['word_encodings']
                      )

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(entity_logits=entity_logits, boundary_logits=boundary_logits,
                                              entity_token_logits=entity_token_logits,
                                              entity_boundary_types=batch['entity_boundary_types'],
                                              entity_types=batch['entity_types'],
                                              entity_token_types=batch['entity_token_types'],
                                              entity_sample_masks=batch['entity_sample_masks'],
                                              entity_boundary_sample_masks=batch['entity_boundary_sample_masks'],
                                              entity_token_masks=batch['entity_token_masks'],
                                              upper_logits=upper_logits,
                                              has_upper=batch['has_upper'])

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

            # if global_iteration % 200 == 0:
            #     self.train_global_iteration = global_iteration
            #     self.optimizer = optimizer
            #     self._eval(model, validation_dataset, input_reader, epoch, updates_epoch, complete_train=False)

        return iteration

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0, complete_train: bool = True):
        if complete_train:
            self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                              self.args.boundary_filter_threshold, self.args.no_overlapping, self._predictions_path,
                              self._examples_path, self.args.example_count, epoch, dataset.label,
                              self.args.boundary_represent_mode,
                              self.args.BD_include_type,
                              self.args.detect_upper,
                              self.args.detect_entity_token,
                              self.args.detect_boundary
                              )

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)
            total_data = tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch)


            for batch in total_data:
                # move batch to selected device
                batch_doc = batch.pop('doc')
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                entity_clf, boundary_clf, entity_word_clf, upper_clf, entities = \
                    model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                          boundaries_masks=batch['boundaries_masks'],
                          boundaries_sample_masks=batch['boundaries_sample_masks'],
                          docs=batch_doc,
                          dataset=dataset,
                          evaluate=True,
                          pos_tags=batch['pos_tags'],
                          word_masks=batch['word_masks'],
                          entity_boundaries_idx=batch['entity_boundaries_idx'],
                          char_encodings=batch['char_encodings'],
                          char_masks=batch['char_masks'],
                          word_encodings=batch['word_encodings'],
                          entity_token_masks=batch['entity_token_masks'],
                          entity_masks=batch['entity_masks'],
                          entity_sample_masks=batch['entity_sample_masks'],
                          ori_entity_masks=batch['ori_entity_masks'],
                          entity_spans=batch['entity_spans'],
                          )
                # evaluate batch
                evaluator.eval_batch(entity_clf, boundary_clf, entities, entity_word_clf, upper_clf, batch, batch_doc)

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, bound_eval, result_str,bound_str = evaluator.compute_scores()



        self._log_eval(*ner_eval, *bound_eval[:3],
                       epoch, iteration, global_iteration, dataset.label)

        if self.args.store_predictions and not self.args.no_overlapping:
            evaluator.store_predictions()

        if self.args.store_examples:
            evaluator.store_examples()

        f1 = ner_eval[2]
        if dataset.label == 'valid' and f1 > self.best_f1:
            self.best_f1 = f1
            if not os.path.exists(os.getcwd() + '/' + self._save_path + '/best_model'):
                os.makedirs(os.getcwd() + '/' + self._save_path + '/best_model')
            with open(self._save_path + '/best_model/' + dataset.label + '_result.txt', 'w') as f:
                f.write(result_str)

            self._save_model(self.bert_config, self._save_path, model, self._tokenizer, global_iteration,
                             optimizer=None, extra=None,
                             include_iteration=False, name='best_model')

        if dataset.label == 'test':
            with open(self._log_path + '/' + dataset.label + '_result.txt', 'w') as f:
                f.write(result_str)

            with open(self._log_path + '/' + 'B.txt', 'w') as f:
                f.write(bound_str[0])

            with open(self._log_path + '/' + 'E.txt', 'w') as f:
                f.write(bound_str[1])

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        # Bert_model_param_no_decay = []
        # Bert_downstream_param_no_decay = []
        # Bert_model_param_decay = []
        # Bert_downstream_param_decay = []

        # for items, p in param_optimizer:
        #     if "bert" in items:
        #         if not any(nd in items for nd in no_decay):
        #             Bert_model_param_no_decay.append(p)
        #         else:
        #             Bert_model_param_decay.append(p)
        #     else:
        #         if not any(nd in items for nd in no_decay):
        #             Bert_downstream_param_no_decay.append(p)
        #         else:
        #             Bert_downstream_param_decay.append(p)

        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]



        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  boundary_prec: float, boundary_rec: float, boundary_f1: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/boundary_prec', boundary_prec, global_iteration)
        self._log_tensorboard(label, 'eval/boundary_rec', boundary_rec, global_iteration)
        self._log_tensorboard(label, 'eval/boundary_f1', boundary_f1, global_iteration)
        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,
                      boundary_prec, boundary_rec, boundary_f1,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        # self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        # self._logger.info("Relations:")
        # for r in input_reader.relation_types.values():
        #     self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            # self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_nec_prec_micro', 'rel_nec_rec_micro', 'rel_nec_f1_micro',
                                                 'rel_nec_prec_macro', 'rel_nec_rec_macro', 'rel_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})
