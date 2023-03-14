from collections import OrderedDict
from typing import List
from torch.utils.data import Dataset as TorchDataset

from model import sampling


class RelationType:
    def __init__(self, identifier, index, short_name, verbose_name, symmetric=False):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name
        self._symmetric = symmetric

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    @property
    def symmetric(self):
        return self._symmetric

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, RelationType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)


class EntityType:
    def __init__(self, identifier, index, short_name, verbose_name):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, EntityType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)


class Token:
    def __init__(self, tid: int, index: int, span_start: int, span_end: int, phrase: str):
        self._tid = tid  # ID within the corresponding dataset
        self._index = index  # original token index in document

        self._span_start = span_start  # start of token span in document (inclusive)
        self._span_end = span_end  # end of token span in document (exclusive)
        self._phrase = phrase

    @property
    def index(self):
        return self._index

    @property
    def span_start(self):
        return self._span_start

    @property
    def span_end(self):
        return self._span_end

    @property
    def span(self):
        return self._span_start, self._span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Token):
            return self._tid == other._tid
        return False

    def __hash__(self):
        return hash(self._tid)

    def __str__(self):
        return self._phrase

    def __repr__(self):
        return self._phrase


class TokenSpan:
    def __init__(self, tokens):
        self._tokens = tokens

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    def __getitem__(self, s):
        if isinstance(s, slice):
            return TokenSpan(self._tokens[s.start:s.stop:s.step])
        else:
            return self._tokens[s]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)


class Entity:
    def __init__(self, eid: int, entity_type: EntityType, tokens: List[Token], phrase: str, start, end):
        self._eid = eid  # ID within the corresponding dataset

        self._entity_type = entity_type

        self._tokens = tokens
        self._phrase = phrase
        self.original_span_start = start
        self.original_span_end = end

    def as_tuple(self):
        return self.span_start, self.span_end, self._entity_type

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def original_span(self):
        return self.original_span_start, self.original_span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self._eid == other._eid
        return False

    def __hash__(self):
        return hash(self._eid)

    def __str__(self):
        return self._phrase


class Boundary:
    def __init__(self, bid: int, idx: int, boundary_type: int, tokens: List[Token], phrase: str, mid: int):
        self._bid = bid  # ID within the corresponding dataset
        self.idx = idx
        self._boundary_type = boundary_type
        self._tokens = tokens
        self._phrase = phrase
        self._mid = mid

    def as_tuple(self):
        return self.span_start, self.span_end, self._boundary_type

    @property
    def boundary_type(self):
        return self._boundary_type

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def phrase(self):
        return self._phrase

    @property
    def mid(self):
        return self._mid

    def __eq__(self, other):
        if isinstance(other, Boundary):
            return self._bid == other._bid
        return False

    def __hash__(self):
        return hash(self._bid)

    def __str__(self):
        return self._phrase


class Relation:
    def __init__(self, rid: int, relation_type: RelationType, head_entity: Entity,
                 tail_entity: Entity, reverse: bool = False):
        self._rid = rid  # ID within the corresponding dataset
        self._relation_type = relation_type

        self._head_entity = head_entity
        self._tail_entity = tail_entity

        self._reverse = reverse

        self._first_entity = head_entity if not reverse else tail_entity
        self._second_entity = tail_entity if not reverse else head_entity

    def as_tuple(self):
        head = self._head_entity
        tail = self._tail_entity
        head_start, head_end = (head.span_start, head.span_end)
        tail_start, tail_end = (tail.span_start, tail.span_end)

        t = ((head_start, head_end, head.entity_type),
             (tail_start, tail_end, tail.entity_type), self._relation_type)
        return t

    @property
    def relation_type(self):
        return self._relation_type

    @property
    def head_entity(self):
        return self._head_entity

    @property
    def tail_entity(self):
        return self._tail_entity

    @property
    def first_entity(self):
        return self._first_entity

    @property
    def second_entity(self):
        return self._second_entity

    @property
    def reverse(self):
        return self._reverse

    def __eq__(self, other):
        if isinstance(other, Relation):
            return self._rid == other._rid
        return False

    def __hash__(self):
        return hash(self._rid)


class Document:
    def __init__(self, doc_id: int, file_key, tokens: List[Token], entities: List[Entity], boundaries: List[int],
                 encoding: List[int], pos_tags: List[int], char_encodings: List[List[int]],
                 word_encodings: List[List[int]],
                 has_upper: List[int]):
        self._doc_id = doc_id  # ID within the corresponding dataset

        self._tokens = tokens
        self._entities = entities
        self._boundaries = boundaries
        # self._relations = relations
        self._file_key = file_key
        self._pos_tags = pos_tags

        # byte-pair document encoding including special tokens ([CLS] and [SEP])
        self._encoding = encoding

        self._char_encodings = char_encodings
        self._word_encodings = word_encodings
        self._has_upper = has_upper

    @property
    def has_upper(self):
        return self._has_upper

    @property
    def char_encodings(self):
        return self._char_encodings

    @property
    def word_encodings(self):
        return self._word_encodings

    @property
    def pos_tags(self):
        return self._pos_tags

    @property
    def doc_id(self):
        return self._doc_id

    @property
    def file_key(self):
        return self._file_key

    @property
    def entities(self):
        return self._entities

    @property
    def boundaries(self):
        return self._boundaries

    @property
    def relations(self):
        return self._relations

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def encoding(self):
        return self._encoding

    @encoding.setter
    def encoding(self, value):
        self._encoding = value

    def __eq__(self, other):
        if isinstance(other, Document):
            return self._doc_id == other._doc_id
        return False

    def __hash__(self):
        return hash(self._doc_id)


class BatchIterator:
    def __init__(self, entities, batch_size, order=None, truncate=False):
        self._entities = entities
        self._batch_size = batch_size
        self._truncate = truncate
        self._length = len(self._entities)
        self._order = order

        if order is None:
            self._order = list(range(len(self._entities)))

        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._truncate and self._i + self._batch_size > self._length:
            raise StopIteration
        elif not self._truncate and self._i >= self._length:
            raise StopIteration
        else:
            entities = [self._entities[n] for n in self._order[self._i:self._i + self._batch_size]]
            self._i += self._batch_size
            return entities


class Dataset(TorchDataset):
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'

    def __init__(self, label, rel_types, entity_types, neg_entity_count,
                 neg_bound_count, max_span_size, boundary_represent_mode, BD_include_type):
        self._label = label
        self._rel_types = rel_types
        self._entity_types = entity_types
        self._neg_entity_count = neg_entity_count
        self._neg_bound_count = neg_bound_count
        self._max_span_size = max_span_size
        self._mode = Dataset.TRAIN_MODE
        self._boundary_represent_mode = boundary_represent_mode
        self._BD_include_type = BD_include_type

        self._fkey2fid = OrderedDict()
        self._fid2doc_ids = OrderedDict()
        self._documents = OrderedDict()
        self._entities = OrderedDict()
        self._relations = OrderedDict()
        self._boundaries = OrderedDict()

        # current ids
        self._doc_id = 0
        self._rid = 0
        self._bid = 0
        self._eid = 0
        self._tid = 0
        self._fid = 0

    def iterate_documents(self, batch_size, order=None, truncate=False):
        return BatchIterator(self.documents, batch_size, order=order, truncate=truncate)

    def iterate_relations(self, batch_size, order=None, truncate=False):
        return BatchIterator(self.relations, batch_size, order=order, truncate=truncate)

    def create_token(self, idx, span_start, span_end, phrase) -> Token:
        token = Token(self._tid, idx, span_start, span_end, phrase)
        self._tid += 1
        return token

    def create_document(self, file_key, tokens, entity_mentions, boundaries, doc_encoding, pos_tags,
                        char_encodings, word_encodings, has_upper) -> Document:
        document = Document(self._doc_id, file_key, tokens, entity_mentions, boundaries, doc_encoding, pos_tags,
                            char_encodings, word_encodings, has_upper)
        self._documents[self._doc_id] = document

        # if self._fkey2fid.__contains__(file_key):
        #     fid = self._fkey2fid[file_key]
        # else:
        #     fid = self._fid
        #     self._fkey2fid[file_key] = fid
        #     self._fid += 1
        # if self._fid2doc_ids.__contains__(fid):
        #     self._fid2doc_ids[fid].append(self._doc_id)
        # else:
        #     self._fid2doc_ids[fid] = [self._doc_id]

        self._doc_id += 1
        return document

    def create_entity(self, entity_type, tokens, phrase, start, end) -> Entity:
        mention = Entity(self._eid, entity_type, tokens, phrase, start, end)
        self._entities[self._eid] = mention
        self._eid += 1
        return mention

    def create_boundary(self, idx, boundary_type, doc_tokens, boundary_present_mode) -> Boundary:
        mid = None
        if boundary_present_mode > 1:
            if idx == 0:
                tokens = doc_tokens[idx:idx + 1]
                mid = tokens[0].span_start
            elif idx == len(doc_tokens):
                tokens = doc_tokens[idx - 1:idx]
                mid = tokens[-1].span_end
            else:
                tokens = doc_tokens[idx - 1:idx + 1]
                mid = tokens[0].span_end
        else:
            tokens = doc_tokens[idx:idx + 1]
        phrase = ' '.join([t.phrase for t in tokens])
        boundary = Boundary(self._bid, idx, boundary_type, tokens, phrase, mid)

        self._boundaries[self._bid] = boundary
        self._bid += 1
        return boundary

    def create_relation(self, relation_type, head_entity, tail_entity, reverse=False) -> Relation:
        relation = Relation(self._rid, relation_type, head_entity, tail_entity, reverse)
        self._relations[self._rid] = relation
        self._rid += 1
        return relation

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index: int):
        doc = self._documents[index]
        if self._mode == Dataset.TRAIN_MODE:
            return sampling.create_train_sample(doc, self._neg_entity_count, self._neg_bound_count,
                                                len(self._entity_types), self._boundary_represent_mode,
                                                self._BD_include_type, self._max_span_size)

        else:
            return sampling.create_eval_sample(doc, self._boundary_represent_mode, self._max_span_size)

    def switch_mode(self, mode):
        self._mode = mode

    def fid2doc_ids(self, fid):
        return self._fid2doc_ids[fid]

    def fkey2fid(self, fkey):
        return self._fkey2fid[fkey]

    @property
    def label(self):
        return self._label

    @property
    def input_reader(self):
        return self._input_reader

    @property
    def documents(self):
        return list(self._documents.values())

    @property
    def entities(self):
        return list(self._entities.values())

    @property
    def relations(self):
        return list(self._relations.values())

    @property
    def document_count(self):
        return len(self._documents)

    @property
    def entity_count(self):
        return len(self._entities)

    @property
    def relation_count(self):
        return len(self._relations)
