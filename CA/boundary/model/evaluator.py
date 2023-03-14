import json
import os
import warnings
from typing import List, Tuple, Dict

import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from transformers import BertTokenizer

from model import util
from model.entities import Document, Dataset, EntityType
from model.input_reader import JsonInputReader
from model.opt import jinja2

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class boundaryType:
    def __init__(self, i, short_name):
        self.short_name = short_name
        self.index = i


class Evaluator:
    def __init__(self, dataset: Dataset, input_reader: JsonInputReader, text_encoder: BertTokenizer,
                 rel_filter_threshold: float, no_overlapping: bool,
                 predictions_path: str, examples_path: str, example_count: int, epoch: int, dataset_label: str,
                 boundary_represent_mode: int, BD_include_type: bool, detect_upper: bool, detect_entity_token: bool,
                 detect_boundary: bool):
        self._detect_upper = detect_upper
        self._detect_entity_token = detect_entity_token
        self._detect_boundary = detect_boundary

        self._text_encoder = text_encoder
        self._input_reader = input_reader
        self._dataset = dataset
        self._boundary_filter_threshold = rel_filter_threshold
        self._no_overlapping = no_overlapping
        self._epoch = epoch
        self._dataset_label = dataset_label

        self._predictions_path = predictions_path

        self._examples_path = examples_path
        self._example_count = example_count

        # entities
        self._gt_entities = []  # ground truth
        self._pred_entities = []  # prediction
        self._gt_entity_words = []
        self._pred_entity_words = []

        # token upper
        self._pred_token_upper = []
        self._gt_token_upper = []

        # boundaries
        self._gt_boundaries = []
        self._pred_boundaries = []
        self._BD_include_type = BD_include_type
        self._boundary_represent_mode = boundary_represent_mode

        self._boundary_types = {'start': boundaryType(1, 'start'), 'end': boundaryType(2, 'end')}
        self._pseudo_boundary_type = boundaryType(1, 'boundary')
        self._entity_token_types = {boundaryType(1, 'pos')}
        self._token_upper_types = {boundaryType(1, 'upper')}

        self._pseudo_entity_type = EntityType('Entity', 1, 'Entity', 'Entity')  # for span only evaluation

        self._convert_gt(self._dataset.documents)

    def eval_batch(self, batch_entity_clf: torch.tensor, batch_boundary_clf: torch.tensor,
                   batch_entities: torch.tensor, batch_entity_word_clf: torch.tensor, batch_upper_clf: torch.tensor,
                   batch: dict, doc: List[Document]):

        batch_size = batch_entity_clf.shape[0]
        boundaries_sample_masks = batch['boundaries_sample_masks'].float()
        entity_token_sample_masks = batch['entity_token_masks'].float()
        if self._detect_boundary:
            boundary_type_num = batch_boundary_clf.shape[2]
            batch_boundary_clf[batch_boundary_clf < self._boundary_filter_threshold] = 0
            batch_boundary_clf = batch_boundary_clf * boundaries_sample_masks.unsqueeze(-1).repeat(1, 1,
                                                                                                   boundary_type_num).float()

        # get maximum activation (index of predicted entity type)
        batch_entity_types = batch_entity_clf.argmax(dim=-1)

        # apply boundary sample mask
        # if not self._BD_include_type:
        #     batch_boundary_clf = batch_boundary_clf.view(batch_size, -1)
        #     batch_boundary_clf *= boundaries_sample_masks

        # apply threshold to boundaries

        # if self._boundary_filter_threshold > 0:
        #     batch_boundary_clf[batch_boundary_clf < self._boundary_filter_threshold] = 0

        if self._detect_entity_token:
            batch_entity_word_clf[batch_entity_word_clf < 0.5] = 0
            batch_entity_word_clf[batch_entity_word_clf >= 0.5] = 1
            batch_entity_word_clf = batch_entity_word_clf.view(batch_size, -1)
            batch_entity_word_clf = (batch_entity_word_clf * entity_token_sample_masks).long()

        if self._detect_upper:
            batch_upper_clf[batch_upper_clf < 0.5] = 0
            batch_upper_clf[batch_upper_clf >= 0.5] = 1
            batch_upper_clf = batch_upper_clf.view(batch_size, -1)
            batch_upper_clf = (batch_upper_clf * entity_token_sample_masks).long()

        for i in range(batch_size):
            # get model predictions for sample

            entity_types = batch_entity_types[i]
            tokens = doc[i].tokens
            tokens_count = len(tokens)
            # get predicted boundary labels and corresponding entity pairs

            sample_pred_boundaries = []
            if self._detect_boundary:
                boundary_clf = batch_boundary_clf[i]
                for j in range(boundary_type_num):
                    a = boundary_clf[:, j]
                    boundaries_nonzero = a.view(-1).nonzero().view(-1)
                    boundary_scores = a[boundaries_nonzero]
                    boundary_peds = boundaries_nonzero
                    if self._boundary_represent_mode == 1:
                        boundary_span = [tokens[b].span for b in boundary_peds]
                    else:
                        boundary_span = []
                        for b in boundary_peds:
                            if b == 0:
                                boundary_span.append(tokens[0].span)
                            elif b == tokens_count:
                                boundary_span.append(tokens[tokens_count - 1].span)
                            else:
                                boundary_span.append(tokens[b - 1:b + 1].span)
                    sample_pred_boundaries += self._convert_pred_boundaries(boundary_peds, boundary_span,
                                                                            boundary_scores, j + 1)

            # get entities that are not classified as 'None'
            valid_entity_indices = entity_types.nonzero().view(-1)
            valid_entity_types = entity_types[valid_entity_indices]
            valid_entity_spans = batch_entities[i][valid_entity_indices]
            valid_entity_scores = torch.gather(batch_entity_clf[i][valid_entity_indices], 1,
                                               valid_entity_types.unsqueeze(1)).view(-1)

            sample_pred_entities = self._convert_pred_entities(valid_entity_types, valid_entity_spans,
                                                               valid_entity_scores)

            if self._detect_entity_token:
                sample_pred_entity_words = batch_entity_word_clf[i].tolist()[
                                           :int(entity_token_sample_masks[i].sum()) + 4]
                self._pred_entity_words.append(sample_pred_entity_words)
            if self._detect_upper:
                sample_pred_upper = batch_entity_word_clf[i].tolist()[:int(entity_token_sample_masks[i].sum()) + 4]
                self._pred_token_upper.append(sample_pred_upper)
            # if self._no_overlapping:
            #     sample_pred_entities, sample_pred_relations = self._remove_overlapping(sample_pred_entities,
            #                                                                            sample_pred_relations)

            self._pred_entities.append(sample_pred_entities)
            self._pred_boundaries.append(sample_pred_boundaries)

    def compute_scores(self):

        result_str = ''
        output_str = "Evaluation\n" + "--- NER(include entity type) ---\n"
        result_str += output_str
        print(output_str)
        gt, pred = self._convert_by_setting(self._gt_entities, self._pred_entities, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=False)
        # if not complete_train:
        #     return ner_eval, None, None

        result_str += self._results_str

        # output_str = "\n--- NER(not include entity type) ---\n"
        # result_str += output_str
        # print(output_str)
        # gt, pred = self._convert_by_setting(self._gt_entities, self._pred_entities)
        # self._score(gt, pred, print_results=True, mode='entity')
        # result_str += self._results_str

        if self._BD_include_type:
            gt, pred = self._convert_boundaries_by_setting(self._gt_boundaries, self._pred_boundaries,
                                                           include_boundary_types=True)
            output_str = "\n--- Boundaries Detection(include boundary type) ---\n"
            result_str += output_str
            print(output_str)
            bound_eval = self._score(gt, pred, print_results=True)
            result_str += self._results_str
        else:
            gt, pred = self._convert_boundaries_by_setting(self._gt_boundaries, self._pred_boundaries)
            output_str = "\n--- Boundaries Detection(not include boundary type) ---\n"
            result_str += output_str
            print(output_str)
            bound_eval = self._score(gt, pred, print_results=True)
            result_str += self._results_str

        if self._detect_entity_token:
            gt, pred = [], []
            for _gt, _pred in zip(self._gt_entity_words, self._pred_entity_words):
                gt += _gt
                pred += _pred
            output_str = "\n--- Entity word Detection ---\n"
            result_str += output_str
            print(output_str)
            self._compute_metrics(gt, pred, self._entity_token_types,
                                  print_results=True)
            result_str += self._results_str

        if self._detect_upper:
            gt, pred = [], []
            for _gt, _pred in zip(self._gt_token_upper, self._pred_token_upper):
                gt += _gt
                pred += _pred
            output_str = "\n--- token upper Detection ---\n"
            result_str += output_str
            print(output_str)
            self._compute_metrics(gt, pred, self._token_upper_types, print_results=True)
            result_str += self._results_str

        start_output_str = ""
        end_output_str = ""
        for boundaries, doc in zip(self._pred_boundaries, self._dataset.documents):
            sentence = doc.tokens
            token_count = len(sentence)
            sentence = [token.phrase for token in sentence]
            pred_begin = ['O'] * token_count
            pred_end = ['O'] * token_count
            for bound in boundaries:
                bt = bound[2].short_name
                if bt == 'start':
                    pred_begin[bound[4]] = 'B'
                else:
                    pred_end[bound[4]-1] = 'B'

            for token, lb in zip(sentence, pred_begin):
                start_output_str += token + '\t' + lb + '\n'
            start_output_str += '\n'

            for token, lb in zip(sentence, pred_end):
                end_output_str += token + '\t' + lb + '\n'
            end_output_str += '\n'

        return ner_eval, bound_eval, result_str, (start_output_str, end_output_str)

    def store_predictions(self):
        predictions = []

        for i, doc in enumerate(self._dataset.documents):
            tokens = doc.tokens
            pred_entities = self._pred_entities[i]
            pred_boundaries = self._pred_boundaries[i]

            # convert entities
            converted_entities = []
            for entity in pred_entities:
                entity_span = entity[:2]
                span_tokens = util.get_span_tokens(tokens, entity_span)
                entity_type = entity[2].identifier
                converted_entity = dict(type=entity_type, start=span_tokens[0].index, end=span_tokens[-1].index + 1)
                converted_entities.append(converted_entity)
            converted_entities = sorted(converted_entities, key=lambda e: e['start'])

            # convert boundaries = []
            converted_boundaries = []
            for boundary in pred_boundaries:
                boundary_idx = boundary[1]
                boundary_start = boundary_idx - 1 if boundary_idx > 0 else 0
                boundary_end = boundary_idx + 1 if boundary_idx < len(tokens) else len(tokens)
                boundary_tokens = tokens[boundary_start:boundary_end]
                converted_boundary = dict(boundary=boundary, boundary_tokens=boundary_tokens)
                converted_boundaries.append(converted_boundary)
            # converted_boundaries = sorted(converted_boundaries, key=lambda e: e['boundary'])

            doc_predictions = dict(tokens=[t.phrase for t in tokens], entities=converted_entities,
                                   boundaries=converted_boundaries)
            predictions.append(doc_predictions)

        # store as json
        label, epoch = self._dataset_label, self._epoch
        with open(self._predictions_path % (label, epoch), 'w') as predictions_file:
            json.dump(predictions, predictions_file)

    def store_examples(self):
        if jinja2 is None:
            warnings.warn("Examples cannot be stored since Jinja2 is not installed.")
            return

        entity_examples = []
        boundary_examples = []
        boundary_examples_nec = []

        for i, doc in enumerate(self._dataset.documents):
            # entities
            entity_example = self._convert_example(doc, self._gt_entities[i], self._pred_entities[i],
                                                   include_entity_types=True, to_html=self._entity_to_html)
            entity_examples.append(entity_example)

            boundary_example = self._convert_example(doc, self._gt_boundaries[i], self._pred_boundaries[i],
                                                     include_entity_types=False, to_html=self._entity_to_html,
                                                     convert_boundary=True)
            boundary_examples.append(boundary_example)
        #
        #     # relations
        #     # without entity types
        #     rel_example = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i],
        #                                         include_entity_types=False, to_html=self._rel_to_html)
        #     rel_examples.append(rel_example)
        #
        #     # with entity types
        #     rel_example_nec = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i],
        #                                             include_entity_types=True, to_html=self._rel_to_html)
        #     rel_examples_nec.append(rel_example_nec)
        #
        label, epoch = self._dataset_label, self._epoch

        # entities
        print(self._examples_path % ('entities', label, epoch))
        self._store_examples(entity_examples[:self._example_count],
                             file_path=self._examples_path % ('entities', label, epoch),
                             template='entity_examples.html')

        self._store_examples(sorted(entity_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('entities_sorted', label, epoch),
                             template='entity_examples.html')

        # boundaries
        self._store_examples(boundary_examples[:self._example_count],
                             file_path=self._examples_path % ('boundaries', label, epoch),
                             template='entity_examples.html')

        # # relations
        # # without entity types
        # self._store_examples(rel_examples[:self._example_count],
        #                      file_path=self._examples_path % ('rel', label, epoch),
        #                      template='relation_examples.html')
        #
        # self._store_examples(sorted(rel_examples[:self._example_count],
        #                             key=lambda k: k['length']),
        #                      file_path=self._examples_path % ('rel_sorted', label, epoch),
        #                      template='relation_examples.html')
        #
        # # with entity types
        # self._store_examples(rel_examples_nec[:self._example_count],
        #                      file_path=self._examples_path % ('rel_nec', label, epoch),
        #                      template='relation_examples.html')
        #
        # self._store_examples(sorted(rel_examples_nec[:self._example_count],
        #                             key=lambda k: k['length']),
        #                      file_path=self._examples_path % ('rel_nec_sorted', label, epoch),
        #                      template='relation_examples.html')

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            gt_boundaries = doc.boundaries
            gt_entities = doc.entities

            # convert ground truth entities for precision/recall/f1 evaluation
            sample_gt_entities = [entity.as_tuple() for entity in gt_entities]
            if self._BD_include_type:
                sample_gt_boundaries = []
                for boundary in gt_boundaries:
                    bs, be, b_type = boundary.as_tuple()
                    if b_type == 3:
                        sample_gt_boundaries.append((bs, be, self._boundary_types['start']))
                        sample_gt_boundaries.append((bs, be, self._boundary_types['end']))
                    if b_type == 1:
                        sample_gt_boundaries.append((bs, be, self._boundary_types['start']))
                    if b_type == 2:
                        sample_gt_boundaries.append((bs, be, self._boundary_types['end']))

            else:
                sample_gt_boundaries = [(*boundary.as_tuple()[:-1], self._pseudo_boundary_type) for boundary in
                                        gt_boundaries]

            if self._no_overlapping:
                sample_gt_entities, sample_gt_boundaries = self._remove_overlapping(sample_gt_entities,
                                                                                    sample_gt_boundaries)

            if self._detect_entity_token:
                sample_entity_gt_words = [0] * len(doc.encoding)
                for entity in gt_entities:
                    for i in range(*entity.span):
                        sample_entity_gt_words[i] = 1
                self._gt_entity_words.append(sample_entity_gt_words)
            if self._detect_upper:
                sample_token_upper_gt = doc.has_upper
                self._gt_token_upper.append(sample_token_upper_gt)
            self._gt_entities.append(sample_gt_entities)
            self._gt_boundaries.append(sample_gt_boundaries)

    def _convert_pred_entities(self, pred_types: torch.tensor, pred_spans: torch.tensor, pred_scores: torch.tensor):
        converted_preds = []

        for i in range(pred_types.shape[0]):
            label_idx = pred_types[i].item()
            entity_type = self._input_reader.get_entity_type(label_idx)

            start, end = pred_spans[i].tolist()
            score = pred_scores[i].item()

            converted_pred = (start, end, entity_type, score)
            converted_preds.append(converted_pred)

        return converted_preds

    def _convert_pred_relations(self, pred_rel_types: torch.tensor, pred_entity_spans: torch.tensor,
                                pred_entity_types: torch.tensor, pred_scores: torch.tensor):
        converted_rels = []
        check = set()

        for i in range(pred_rel_types.shape[0]):
            label_idx = pred_rel_types[i].item()
            pred_rel_type = self._input_reader.get_relation_type(label_idx)
            pred_head_type_idx, pred_tail_type_idx = pred_entity_types[i][0].item(), pred_entity_types[i][1].item()
            pred_head_type = self._input_reader.get_entity_type(pred_head_type_idx)
            pred_tail_type = self._input_reader.get_entity_type(pred_tail_type_idx)
            score = pred_scores[i].item()

            spans = pred_entity_spans[i]
            head_start, head_end = spans[0].tolist()
            tail_start, tail_end = spans[1].tolist()

            converted_rel = ((head_start, head_end, pred_head_type),
                             (tail_start, tail_end, pred_tail_type), pred_rel_type)
            converted_rel = self._adjust_rel(converted_rel)

            if converted_rel not in check:
                check.add(converted_rel)
                converted_rels.append(tuple(list(converted_rel) + [score]))

        return converted_rels

    def _convert_pred_boundaries(self, boundary_peds: torch.tensor, boundary_span: List[set],
                                 pred_scores: torch.tensor, boundary_type=1):
        converted_preds = []

        for i in range(boundary_peds.shape[0]):
            boundary_idx = boundary_peds[i].item()
            start, end = boundary_span[i]
            score = pred_scores[i].item()
            if boundary_type == 1:
                converted_pred = (start, end, self._boundary_types['start'], score, boundary_idx)
            if boundary_type == 2:
                converted_pred = (start, end, self._boundary_types['end'], score, boundary_idx)
            converted_preds.append(converted_pred)

        return converted_preds

    def _convert_boundaries_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                                       include_boundary_types: bool = False, include_score: bool = False):
        assert len(gt) == len(pred)

        # either include or remove entity types based on setting
        def convert(t):
            if not include_boundary_types:
                # remove  type and score for evaluation
                c = [t[0], t[1], self._pseudo_boundary_type]
            else:
                c = [t[0], t[1], t[2]]

            if include_score and len(t) > 3:
                # include prediction scores
                c.append(t[3])

            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])

        return converted_gt, converted_pred

    def _remove_overlapping(self, entities):
        non_overlapping_entities = []

        for entity in entities:
            if not self._is_overlapping(entity, entities):
                non_overlapping_entities.append(entity)
        return non_overlapping_entities

    def _is_overlapping(self, e1, entities):
        for e2 in entities:
            if self._check_overlap(e1, e2):
                return True

        return False

    def _check_overlap(self, e1, e2):
        if e1 == e2 or e1[1] <= e2[0] or e2[1] <= e1[0]:
            return False
        else:
            return True

    def _adjust_rel(self, rel: Tuple):
        adjusted_rel = rel
        if rel[-1].symmetric:
            head, tail = rel[:2]
            if tail[0] < head[0]:
                adjusted_rel = tail, head, rel[-1]

        return adjusted_rel

    def _convert_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                            include_entity_types: bool = False, include_score: bool = False):
        assert len(gt) == len(pred)

        # either include or remove entity types based on setting
        def convert(t):
            if not include_entity_types:
                # remove entity type and score for evaluation
                if type(t[0]) == int:  # entity
                    c = [t[0], t[1], self._pseudo_entity_type]
                else:  # relation
                    c = [(t[0][0], t[0][1], self._pseudo_entity_type),
                         (t[1][0], t[1][1], self._pseudo_entity_type), t[2]]
            else:
                c = list(t[:3])

            if include_score and len(t) > 3:
                # include prediction scores
                c.append(t[3])

            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])

        return converted_gt, converted_pred

    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False, mode: str = ''):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

        # if mode == 'boundary':
        #     for (sample_gt, sample_pred) in zip(gt, pred):
        #         union = set()
        #         union.update(sample_gt)
        #         union.update(sample_pred)
        #         for s in union:
        #             if s in sample_gt:
        #                 t = s[2]
        #                 gt_flat.append(t)
        #             else:
        #                 gt_flat.append(0)
        #             if s in sample_pred:
        #                 t = s[2]
        #                 pred_flat.append(t)
        #             else:
        #                 pred_flat.append(0)
        #     # types.add(boundaryType(1, 'Boundary'))
        #     # types.add(boundaryType(1, 'start'))
        #     # types.add(boundaryType(2, 'end'))
        #     types = self._boundary_types
        # if mode == 'entity_word':
        #     for (sample_gt, sample_pred) in zip(gt, pred):
        #         union = set()
        #         union.update(sample_gt)
        #         union.update(sample_pred)
        #         for s in union:
        #             if s in sample_gt:
        #                 t = s[1]
        #                 gt_flat.append(t)
        #             else:
        #                 gt_flat.append(0)
        #             if s in sample_pred:
        #                 t = s[1]
        #                 pred_flat.append(t)
        #             else:
        #                 pred_flat.append(0)
        #     # types.add(boundaryType(1, 'Boundary'))
        #     # types.add(boundaryType(1, 'start'))
        #     # types.add(boundaryType(2, 'end'))
        #     types = self._entity_word_types
        # if mode == 'entity':
        #     for (sample_gt, sample_pred) in zip(gt, pred):
        #         union = set()
        #         union.update(sample_gt)
        #         union.update(sample_pred)
        #         for s in union:
        #             if s in sample_gt:
        #                 t = s[2]
        #                 gt_flat.append(t.index)
        #                 types.add(t)
        #             else:
        #                 gt_flat.append(0)
        #
        #             if s in sample_pred:
        #                 t = s[2]
        #                 pred_flat.append(t.index)
        #                 types.add(t)
        #             else:
        #                 pred_flat.append(0)
        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)
            for s in union:
                if s in sample_gt:
                    t = s[-1]
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[-1]
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(0)

        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t.index for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t.short_name))
            results.append('\n')

        if len(types) > 1:
            results.append('\n')
            # micro
            results.append(row_fmt % self._get_row(micro, 'micro'))
            results.append('\n')

            # macro
            results.append(row_fmt % self._get_row(macro, 'macro'))

        results_str = ''.join(results)
        self._results_str = results_str
        print(results_str)
        return results_str

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    def _convert_example(self, doc: Document, gt: List[Tuple], pred: List[Tuple],
                         include_entity_types: bool, to_html, convert_boundary=False):
        encoding = doc.encoding

        if not convert_boundary:
            gt, pred = self._convert_by_setting([gt], [pred], include_entity_types=include_entity_types,
                                                include_score=True)
            gt, pred = gt[0], pred[0]

        # get micro precision/recall/f1 scores
        if gt or pred:
            pred_s = [p[:3] for p in pred]  # remove score
            precision, recall, f1 = self._score([gt], [pred_s])[:3]
        else:
            # corner case: no ground truth and no predictions
            precision, recall, f1 = [100] * 3

        scores = [p[-1] for p in pred]
        pred = [p[:-1] for p in pred]
        union = set(gt + pred)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for s in union:
            if convert_boundary:
                type_verbose = 'boundary'
            else:
                type_verbose = s[2].verbose_name

            if s in gt:
                if s in pred:
                    score = scores[pred.index(s)]
                    tp.append((to_html(s, encoding, boundary=convert_boundary), type_verbose, score))
                else:
                    fn.append((to_html(s, encoding, boundary=convert_boundary), type_verbose, -1))
            else:
                score = scores[pred.index(s)]
                fp.append((to_html(s, encoding, boundary=convert_boundary), type_verbose, score))

        tp = sorted(tp, key=lambda p: p[-1], reverse=True)
        fp = sorted(fp, key=lambda p: p[-1], reverse=True)

        text = self._prettify(self._text_encoder.decode(encoding))
        return dict(text=text, tp=tp, fn=fn, fp=fp, precision=precision, recall=recall, f1=f1, length=len(doc.tokens))

    def _convert_example_boundary(self, doc: Document, gt: List[Tuple], pred: List[Tuple],
                                  include_entity_types: bool, to_html):
        encoding = doc.encoding

        # get micro precision/recall/f1 scores
        if gt or pred:
            pred_s = [p[:2] for p in pred]  # remove score
            gt_flat = [0] * (len(doc.tokens) + 1)
            pred_flat = [0] * (len(doc.tokens) + 1)
            for b in gt:
                gt_flat[b[0]] = 1
            for b in pred:
                pred_flat[b[0]] = 1
            precision, recall, f1 = self._compute_boundary_metrics(gt_flat, pred_flat, [1], print_results=False)
            # precision, recall, f1 = self._score([gt], [pred_s])[:3]
        else:
            # corner case: no ground truth and no predictions

            precision, recall, f1 = [100] * 3

        scores = [p[-1] for p in pred]

        # boundary_represent_mode=1
        # if boundary_represent_mode==1
        #     pred = [p for p in pred]
        union = set(gt + pred)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for s in union:
            type_verbose = s[2].verbose_name

            if s in gt:
                if s in pred:
                    score = scores[pred.index(s)]
                    tp.append((to_html(s, encoding), type_verbose, score))
                else:
                    fn.append((to_html(s, encoding), type_verbose, -1))
            else:
                score = scores[pred.index(s)]
                fp.append((to_html(s, encoding), type_verbose, score))

        tp = sorted(tp, key=lambda p: p[-1], reverse=True)
        fp = sorted(fp, key=lambda p: p[-1], reverse=True)

        text = self._prettify(self._text_encoder.decode(encoding))
        return dict(text=text, tp=tp, fn=fn, fp=fp, precision=precision, recall=recall, f1=f1, length=len(doc.tokens))

    def _entity_to_html(self, entity: Tuple, encoding: List[int], boundary=False):
        start, end = entity[:2]
        if boundary:
            entity_type = 'boundary'
        else:
            entity_type = entity[2].verbose_name

        tag_start = ' <span class="entity">'
        tag_start += '<span class="type">%s</span>' % entity_type

        ctx_before = self._text_encoder.decode(encoding[:start])
        e1 = self._text_encoder.decode(encoding[start:end])
        ctx_after = self._text_encoder.decode(encoding[end:])

        html = ctx_before + tag_start + e1 + '</span> ' + ctx_after
        html = self._prettify(html)

        return html

    def _rel_to_html(self, relation: Tuple, encoding: List[int]):
        head, tail = relation[:2]
        head_tag = ' <span class="head"><span class="type">%s</span>'
        tail_tag = ' <span class="tail"><span class="type">%s</span>'

        if head[0] < tail[0]:
            e1, e2 = head, tail
            e1_tag, e2_tag = head_tag % head[2].verbose_name, tail_tag % tail[2].verbose_name
        else:
            e1, e2 = tail, head
            e1_tag, e2_tag = tail_tag % tail[2].verbose_name, head_tag % head[2].verbose_name

        segments = [encoding[:e1[0]], encoding[e1[0]:e1[1]], encoding[e1[1]:e2[0]],
                    encoding[e2[0]:e2[1]], encoding[e2[1]:]]

        ctx_before = self._text_encoder.decode(segments[0])
        e1 = self._text_encoder.decode(segments[1])
        ctx_between = self._text_encoder.decode(segments[2])
        e2 = self._text_encoder.decode(segments[3])
        ctx_after = self._text_encoder.decode(segments[4])

        html = (ctx_before + e1_tag + e1 + '</span> '
                + ctx_between + e2_tag + e2 + '</span> ' + ctx_after)
        html = self._prettify(html)

        return html

    def _prettify(self, text: str):
        text = text.replace('_start_', '').replace('_classify_', '').replace('<unk>', '').replace('⁇', '')
        text = text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        return text

    def _store_examples(self, examples: List[Dict], file_path: str, template: str):
        template_path = os.path.join(SCRIPT_PATH, 'templates', template)

        # read template
        with open(os.path.join(SCRIPT_PATH, template_path)) as f:
            template = jinja2.Template(f.read())

        # write to disc
        template.stream(examples=examples).dump(file_path)
