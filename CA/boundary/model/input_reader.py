import json
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
from typing import Iterable, List, Tuple

from tqdm import tqdm
from transformers import BertTokenizer

from model import util
from model.entities import Dataset, EntityType, RelationType, Entity, Relation, Document


class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, logger: Logger = None,
                 boundary_represent_mode: int = None, DB_include_type: bool = None,
                 detect_upper: bool = None, detect_entity_token: bool = None):
        types = json.load(open(types_path), object_pairs_hook=OrderedDict)  # entity + relation types

        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()
        self._relation_types = OrderedDict()
        self._idx2relation_type = OrderedDict()

        self._detect_upper = detect_upper
        self._detect_entity_token = detect_entity_token

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type

        # relations
        # add 'None' relation type
        none_relation_type = RelationType('None', 0, 'None', 'No Relation')
        self._relation_types['None'] = none_relation_type
        self._idx2relation_type[0] = none_relation_type

        self._neg_entity_count = neg_entity_count
        self._neg_rel_count = neg_rel_count
        self._max_span_size = max_span_size
        self._boundary_represent_mode = boundary_represent_mode
        self._DB_include_type = DB_include_type

        self._datasets = dict()

        self._tokenizer = tokenizer
        self._logger = logger

        self._vocabulary_size = tokenizer.vocab_size
        self._context_size = -1

        self.upper_chars = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

        pos = ['[PAD]', '[START]', '[NN]', '[JJS]', '[RB]', '[JJ]', '[UH]', '[NNP]', '[CD]', '[ADD]', '[SYM]',
               '[-LRB-]', '[HYPH]', '[WDT]',
               '[POS]', '[WRB]', '[PRP]', '[WP]', '[TO]', '[WP$]', '[PRP$]', '[JJR]', '[MD]', '[RBS]', '[EX]', '[,]',
               '[NNS]', '[LS]', '[CC]', '[``]', '[DT]', '[VBN]', '[VBZ]', '[PDT]', "['']", '[VBG]', '[:]', '[IN]',
               '[NFP]', '[GW]', '[RBR]', '[.]', '[$]', '[NNPS]', '[VBP]', '[FW]', '[RP]', '[-RRB-]', '[VB]', '[VBD]',
               '[(]', '[)]', '[*]', '[-]', '[NNS|FW]', '[JJ|NN]', '[VBN|JJ]', '[IN|PRP$]', '[NN|NNS]', '[CT]', '[N]',
               '[NN|CD]', '[JJ|NNS]', '[PP]', '[JJ|VBG]', '[VBP|VBZ]', '[VBG|NN]', '[NN|DT]', '[]', '[VBG|JJ]',
               '[IN|CC]', '[JJ|VBN]', '[XT]', '[AFX]']
        self.pos2idx = dict(zip(pos, [i for i in range(len(pos))]))
        chars = ['[PAD]', '[START]', '8', 'k', 'i', 'n', 'w', '{', '\xa0', 'A', 'r', 'z', '[', '^', 'C', '/', '&', '*',
                 'b', '#', 'O', 'X', '6', 'W', 'K', 'Y', 'D', '5', 'L', '7', 'y', 'Z', 'u', 'T', 'v', 'J', '(', '?',
                 'j',
                 '3', '-', '1', '0', 'U', 'V', '!', 'E', '`', 'a', '_', 'h', 'H', ']', 'I', 'P', 'm', 'Q', 'p', '4',
                 'R', '2', 't', '|', 'N', ':', 'l', 'e', 'M', 'c', 'F', 'd', 'q', 'x', '.', "'", 'g', 'o', 'f', 'B',
                 'G', ')', '"', '=', '9', 'S', '$', 's', ';', '@', ',', '%', '+', '聽', '[END]']
        self.char2idx = dict(zip(chars, [i for i in range(len(chars))]))

        words = [line.strip() for line in open('data/ace_genia_word.txt', encoding='utf8').readlines()]
        self.word2idx = dict(zip(words, [i for i in range(len(words))]))

        self.language = 'en'
        self.t_set = set()

    @abstractmethod
    def read(self, datasets):
        pass

    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def get_relation_type(self, idx) -> RelationType:
        relation = self._idx2relation_type[idx]
        return relation

    def _calc_context_size(self, datasets: Iterable[Dataset]):
        sizes = []

        for dataset in datasets:
            for doc in dataset.documents:
                sizes.append(len(doc.encoding))

        context_size = max(sizes)
        return context_size

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def context_size(self):
        return self._context_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, logger: Logger = None,
                 boundary_present_mode: int = None, DB_include_type: bool = None,
                 detect_upper: bool = None, detect_entity_token: bool = None
                 ):
        super().__init__(types_path, tokenizer, neg_entity_count, neg_rel_count, max_span_size, logger,
                         boundary_present_mode, DB_include_type, detect_upper, detect_entity_token)

    def read(self, dataset_paths):
        for dataset_label, dataset_path in dataset_paths.items():
            dataset = Dataset(dataset_label, self._relation_types, self._entity_types, self._neg_entity_count,
                              self._neg_rel_count, self._max_span_size, self._boundary_represent_mode,
                              self._DB_include_type)
            self._parse_dataset(dataset_path, dataset)
            self._datasets[dataset_label] = dataset

        self._context_size = self._calc_context_size(self._datasets.values())

    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path, encoding='utf8'))
        i = 0
        n = 0
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            i += 1
            # if i<4:
            #     continue
            self._parse_document(document, dataset)
            # if i == 32:
            #     break

    def _parse_document(self, doc, dataset) -> Document:

        jtokens = doc['tokens']
        jentities = doc['entities']
        file_key = ''

        if self.language == 'en':
            pos_tags = [0] + [self.pos2idx[pos] for pos in doc['pos-tags']] + [0]
            word_encodings = [0] + [self.word2idx[word] if self.word2idx.__contains__(word) else self.word2idx['[UNK]']
                                    for word in doc['tokens']] + [0]
            last_jtokens = doc['last_tokens']
            next_jtokens = doc['next_tokens']
        else:
            pos_tags = [0]
            word_encodings = [0]
            last_jtokens = []
            next_jtokens = []

        # parse tokens
        doc_tokens, doc_encoding, char_encodings, has_upper = self._parse_tokens(jtokens, last_jtokens, next_jtokens,
                                                                                 dataset)

        # parse entity mentions and entity boundary
        entities, boundaries = self._parse_entities_and_boundaries(jentities, doc_tokens, dataset)

        document = dataset.create_document(file_key, doc_tokens, entities, boundaries, doc_encoding, pos_tags,
                                           char_encodings, word_encodings, has_upper)

        return document

    def _parse_tokens(self, jtokens, last_jtokens, next_jtokens, dataset):
        doc_tokens = []

        # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
        doc_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]'), self._tokenizer.convert_tokens_to_ids('[PAD]')]
        char_encodings = [[0]]
        has_upper = [0, 0]

        # parse tokens
        for i, token_phrase in enumerate(jtokens):
            if self.language == 'en':
                char_encodings.append([self.char2idx[ch] for ch in token_phrase])
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))
            token = dataset.create_token(i, span_start, span_end, token_phrase)
            doc_tokens.append(token)
            doc_encoding += token_encoding

            if self._detect_upper:
                has_upper += [1 if set(tk).intersection(self.upper_chars) else 0 for tk in
                              self._tokenizer.convert_ids_to_tokens(token_encoding)]

        doc_encoding += [self._tokenizer.convert_tokens_to_ids('[PAD]')]
        doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]
        char_encodings += [[0]]
        has_upper += [0, 0]

        # # 用上下句子来填充
        if len(last_jtokens) or len(next_jtokens):
            if len(doc_encoding) < 128:
                for i, token_phrase in enumerate(last_jtokens):
                    token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
                    if len(doc_encoding) + len(token_encoding) >= 370:
                        break
                    doc_encoding += token_encoding
                doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]
                if len(doc_encoding) < 370:
                    for i, token_phrase in enumerate(next_jtokens):
                        token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
                        if len(doc_encoding) + len(token_encoding) >= 370:
                            break
                        doc_encoding += token_encoding
                    doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]

        # 用上下句子来填充
        # for token_phrase in last_jtokens:
        #     token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
        #     doc_encoding += token_encoding
        # doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]
        # for token_phrase in next_jtokens:
        #     token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
        #     doc_encoding += token_encoding
        # doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]

        return doc_tokens, doc_encoding, char_encodings, has_upper

    def _parse_entities_and_boundaries(self, jentities, doc_tokens, dataset) -> Tuple[List[Entity], List[int]]:
        entities = []
        boundaries = []
        # [ 北 京 大 学]
        start_boundary_idx = []
        end_boundary_idx = []
        for _, jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']
            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase, start, end)
            entities.append(entity)

            if start not in start_boundary_idx:
                start_boundary_idx.append(start)

            if self._boundary_represent_mode == 1:
                end -= 1
            if end not in end_boundary_idx:
                end_boundary_idx.append(end)
        both_boundary_idx = list(set(start_boundary_idx).intersection(set(end_boundary_idx)))
        start_boundary_idx = list(set(start_boundary_idx).difference(set(both_boundary_idx)))
        end_boundary_idx = list(set(end_boundary_idx).difference(set(both_boundary_idx)))
        for idx in start_boundary_idx:
            boundary = dataset.create_boundary(idx, 1, doc_tokens, self._boundary_represent_mode)
            boundaries.append(boundary)
        for idx in end_boundary_idx:
            boundary = dataset.create_boundary(idx, 2, doc_tokens, self._boundary_represent_mode)
            boundaries.append(boundary)
        for idx in both_boundary_idx:
            boundary = dataset.create_boundary(idx, 3, doc_tokens, self._boundary_represent_mode)
            boundaries.append(boundary)

        boundaries.sort(key=lambda t: t.idx, reverse=False)

        return entities, boundaries
