import random

import torch
from torch import nn as nn
from transformers import BertConfig, BertTokenizer
from transformers import BertModel, AlbertModel, AlbertConfig
from transformers import BertPreTrainedModel

from model import sampling
from model import util
from model.entities import Dataset, Document


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h


class MLP(nn.Module):
    def __init__(self, input_size, common_size, prob_drop=0.2):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Dropout(prob_drop),
            nn.ReLU(inplace=True),
            # nn.Linear(input_size // 2, input_size // 4),
            # nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, common_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out


class SpERT(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and boundaries """

    VERSION = '1.1'

    # def eval(self):
    #     print('evalllll---------------------------')
    #     self.bert.eval()
    #     return super(SpERT, self).eval()

    def __init__(self, config, tokenizer: BertTokenizer, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, boundary_filter_threshold: float,
                 boundary_represent_mode: int, use_sent_ctx: bool, detect_boundary: bool,
                 use_size_embedding: bool, BD_include_type: bool, dataset: Dataset, detect_upper: bool,
                 detect_entity_token: bool, bert_type: str, max_span_size: int):
        super(SpERT, self).__init__(config)
        self._bert_type = bert_type
        self._max_span_size = max_span_size

        self._detect_upper = detect_upper
        self._detect_entity_token = detect_entity_token
        self._detect_boundary = detect_boundary

        self._use_sent_ctx = use_sent_ctx
        self._use_size_embedding = use_size_embedding
        self._BD_include_type = BD_include_type
        self._dataset = dataset
        self._docs = dataset.documents
        self.bert_config = config
        self._use_pos_tag_embedding = True
        self._use_char_embedding = True
        self._use_word_embedding = True
        self._dense_entity = False
        self._dense_boundary = True

        if self._bert_type == 'bert':
            self.bert = BertModel(config)
        elif self._bert_type == 'albert':
            self.albert = AlbertModel(config)
            self._albert.hidden_size = self.bert_config.hidden_size
            self._dense_h = MLP(self.bert_config.hidden_size, 768, prob_drop=prop_drop)
            self.bert_config.hidden_size = 768

        self.boundary_filter_threshold = boundary_filter_threshold
        self.boundary_represent_mode = boundary_represent_mode

        # add dim of entity max pooling repr
        entity_classify_input_dim = self.bert_config.hidden_size
        # add dim of boundary token repr
        boundary_classify_input_dim = self.bert_config.hidden_size

        if self._use_sent_ctx:
            ctx_w_dim = 512
            self.W_key = torch.nn.Parameter(torch.randn(self.bert_config.hidden_size, ctx_w_dim))
            self.W_query = torch.nn.Parameter(torch.randn(self.bert_config.hidden_size, ctx_w_dim))
            entity_classify_input_dim += self.bert_config.hidden_size

        # Semantic enhancement
        semantic_enhance_dim = 0
        if self._use_pos_tag_embedding:
            self.pos_tag_embeddings = nn.Embedding(80, 200)
            semantic_enhance_dim += 200
        if self._use_char_embedding:
            lstm_hidden_size = 200
            self.char_embedding = nn.Embedding(128, 300)
            self.char_lstm = nn.LSTM(input_size=300, hidden_size=lstm_hidden_size
                                     , num_layers=1, bidirectional=False, batch_first=True)
            semantic_enhance_dim += lstm_hidden_size

        if self._use_word_embedding:
            w_embed = torch.load("./data/word_embedding.pt")
            self.word_embedding = nn.Embedding.from_pretrained(w_embed)
            semantic_enhance_dim += 300

        if self._use_word_embedding or self._use_pos_tag_embedding or self._use_char_embedding:
            sem_enh_lstm_hidden_size = 300
            # self.Semantic_enhance_lstm = nn.LSTM(input_size=semantic_enhance_dim,
            #                                      hidden_size=sem_enh_lstm_hidden_size
            #                                      , num_layers=1, bidirectional=True, batch_first=True)
            # add semantic_enhance_out_dim
            entity_classify_input_dim += semantic_enhance_dim
            boundary_classify_input_dim += semantic_enhance_dim

        # boundary detector
        if self._BD_include_type:
            if self._dense_boundary:
                reduced_dim = 512
                self.dense_boundary_MLP = MLP(boundary_classify_input_dim, reduced_dim, prob_drop=prop_drop)
            else:
                reduced_dim = boundary_classify_input_dim
            if self.boundary_represent_mode == 2:
                self.left_token_score = nn.Linear(boundary_classify_input_dim, 1)
                self.right_token_score = nn.Linear(boundary_classify_input_dim, 1)
                self.boundary_classifier = nn.Linear(reduced_dim * boundary_represent_mode, 2)
                self.boundary_Semantic_interaction1_w = torch.nn.Parameter(
                    torch.randn(2, reduced_dim, reduced_dim))

            else:
                self.boundary_classifier = nn.Linear(boundary_classify_input_dim * boundary_represent_mode, 2)

        else:
            self.boundary_classifier = nn.Linear(boundary_classify_input_dim * boundary_represent_mode, 1)
            self.boundary_Semantic_interaction1_w = torch.nn.Parameter(
                torch.randn(boundary_classify_input_dim, boundary_classify_input_dim))

        if self._use_size_embedding:
            self.size_embeddings = nn.Embedding(100, size_embedding)
            entity_classify_input_dim += size_embedding

        # entity classifier
        # layers

        # self.dense_entity_MLP = MLP(self.bert_config.hidden_size, token_reduced_dim)
        self.entity_classifier = MLP(entity_classify_input_dim, entity_types, prob_drop=prop_drop)

        self.dropout = nn.Dropout(prop_drop)

        # self._cls_token = tokenizer.convert_tokens_to_ids('[CLS]')
        self._token_B = tokenizer.convert_tokens_to_ids('[Entity_S]')
        self._token_E = tokenizer.convert_tokens_to_ids('[Entity_E]')
        # self._entity_types = entity_types

        # weight initialization
        self.bert.init_weights()

        # entity word classifier
        if self._detect_entity_token:
            self.entity_word_classifier = MLP(self.bert_config.hidden_size, 1)
        if self._detect_upper:
            self.lower_upper_classifier = MLP(self.bert_config.hidden_size, 1)

        if bert_type == 'albert':
            self.bert_config.hidden_size = self._albert.hidden_size

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, entity_boundary_masks: torch.tensor, pos_tags: torch.tensor,
                       word_masks: torch.tensor,
                       entity_boundaries_idx: torch.tensor, char_encodings: torch.tensor,
                       ori_entity_masks: torch.tensor, char_masks: torch.tensor, word_encodings: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        # vector, pooler, enc_layers = self.bert(input_ids=encodings, attention_mask=context_masks)
        # h = torch.cat([t.unsqueeze(-1) for t in enc_layers[-4:]], 3).sum(-1)/4

        batch_size = char_encodings.shape[0]
        word_count = char_encodings.shape[1]
        char_count = char_encodings.shape[2]

        semantic_enhance_repr = []

        if self._use_char_embedding:
            char_repr = self.char_embedding(char_encodings)
            char_repr = char_repr.view([batch_size * word_count, char_count, -1])
            char_h = self.char_lstm(char_repr)[0]
            char_h = char_h.view(batch_size, word_count, char_count, -1)
            # m = ((char_masks == 0).float() * (-1e10)).unsqueeze(-1)
            m = char_masks.float()
            # real_char_count = m.sum(dim=2)
            # # 1替换0 避免除以0
            # real_char_count[real_char_count == 0] = 1
            #
            # # 平滑处理 用一个很小的数替代0
            # m[m == 0] = 1e-30
            #
            # # 分母加1避免出现 除以0的情况
            # char_h = (char_h * m).sum(dim=2) / real_char_count
            # t_idx = m.long().sum(dim=2).view(-1) - 1
            # t_idx[t_idx == -1] = 0
            # t_idx = t_idx.tolist()  # [[1, 2, 3], [], []]
            # char_h_idx = [[i for i in range(batch_size)] * word_count, [i for i in range(word_count)] * batch_size,
            #               t_idx]

            char_h = char_h[:, :, char_count - 1, :]
            # char_h = char_h.view(batch_size, word_count, -1)

            semantic_enhance_repr.append(char_h)

        if self._use_pos_tag_embedding:
            pos_repr = self.pos_tag_embeddings(pos_tags)
            semantic_enhance_repr.append(pos_repr)

        if self._use_word_embedding:
            word_repr = self.word_embedding(word_encodings)
            semantic_enhance_repr.append(word_repr)

        semantic_enhance_h = None
        if self._use_word_embedding or self._use_pos_tag_embedding or self._use_char_embedding:
            semantic_enhance_repr = torch.cat(semantic_enhance_repr, dim=2)
            # semantic_enhance_lstm_output, _ = self.Semantic_enhance_lstm(semantic_enhance_repr)
            # semantic_enhance_h = self.dense_Semantic_enhance_Linear(semantic_enhance_lstm_output)
            semantic_enhance_h = semantic_enhance_repr

        if self._bert_type == 'bert':
            h = self.bert(input_ids=encodings, attention_mask=context_masks)[0]
        else:
            h = self.albert(input_ids=encodings, attention_mask=context_masks)[0]
            h = self._dense_h(h)

        batch_size = encodings.shape[0]
        entity_count = entity_masks.shape[1]

        # classify entity words
        entity_token_clf, upper_clf, boundary_clf = None, None, None
        if self._detect_entity_token:
            entity_token_clf = self._classify_entity_token(h)
        if self._detect_upper:
            upper_clf = self._classify_upper_lower(h)
        if self._detect_boundary:
            # classify entity boundaries
            boundary_clf = self._classify_boundaries(encodings, h, entity_boundary_masks, entity_boundaries_idx,
                                                     semantic_enhance_h)
        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes

        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, context_masks,
                                                                size_embeddings,
                                                                semantic_enhance_h, ori_entity_masks)

        # # classify relations
        # h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        # rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
        #     self.rel_classifier.weight.device)
        #
        # # obtain relation logits
        # # chunk processing to reduce memory usage
        # for i in range(0, relations.shape[1], self._max_pairs):
        #     # classify relation candidates
        #     chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
        #                                                 relations, rel_masks, h_large, i)
        #     rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits

        # return entity_clf, rel_clf
        # if last_batch:
        return entity_clf, boundary_clf, entity_token_clf, upper_clf

    def _classify_entity_token(self, h):
        h = self.dropout(h)
        logits = self.entity_word_classifier(h)
        logits = torch.sigmoid(logits)
        return logits

    def _classify_upper_lower(self, h):
        h = self.dropout(h)
        logits = self.lower_upper_classifier(h)
        logits = torch.sigmoid(logits)
        return logits

    def _classify_boundaries(self, encodings, h, boundary_masks, entity_boundary_idx, semantic_enhance_h):

        h_large = h.unsqueeze(1).repeat(1, max(boundary_masks.shape[1], 1), 1, 1)
        # batch boundary_count sentence_length  768

        batch_size = encodings.shape[0]
        boundary_count = boundary_masks.shape[1]

        if self.boundary_represent_mode == 1:
            m = ((boundary_masks == 0).float() * (-1e30)).unsqueeze(-1)
            boundary_repr = m + h_large
            # max pooling
            boundary_repr = boundary_repr.max(dim=2)[0]
            if self._use_word_embedding or self._use_pos_tag_embedding or self._use_char_embedding:
                enh_repr = []
                for i in range(batch_size):
                    enh_repr.append(semantic_enhance_h[i][entity_boundary_idx[i] + 1].unsqueeze(0))
                enh_repr = torch.cat(enh_repr, dim=0)
                boundary_repr = torch.cat([boundary_repr, enh_repr], dim=2)

            boundary_repr = self.dropout(boundary_repr)
            # classify boundary candidates

            # boundary_repr = self.dense_boundary_MLP(boundary_repr)
            # boundary_repr = self.dropout(boundary_repr)
            boundary_logits = self.boundary_classifier(boundary_repr)

        else:
            dual_boundary_masks = boundary_masks
            m1 = ((dual_boundary_masks[:, :, 0, :] == 0).float() * (-1e30)).unsqueeze(-1)
            # batch size,entity_count,seq_length
            m2 = ((dual_boundary_masks[:, :, 1, :] == 0).float() * (-1e30)).unsqueeze(-1)
            boundary_left_repr = (m1 + h_large).max(dim=2)[0]  # batch size,entity_count
            boundary_right_repr = (m2 + h_large).max(dim=2)[0]
            boundary_left_repr[boundary_left_repr == -1e30] = 1e-10
            boundary_right_repr[boundary_right_repr == -1e30] = 1e-10

            # m1 = (dual_boundary_masks[:, :, 0, :].float()).unsqueeze(-1)
            # left_token_count = m1.sum(dim=2)
            # left_token_count[left_token_count == 0] = 1
            # m1[m1 == 0] = 1e-30
            # # batch size,entity_count,seq_length
            # m2 = (dual_boundary_masks[:, :, 1, :].float()).unsqueeze(-1)
            # right_token_count = m2.sum(dim=2)
            # right_token_count[right_token_count == 0] = 1
            # m2[m2 == 0] = 1e-30
            #
            # boundary_left_repr = (m1 * h_large).sum(dim=2) / left_token_count
            #
            # boundary_right_repr = (m2 * h_large).sum(dim=2) / right_token_count

            if self._use_word_embedding or self._use_pos_tag_embedding or self._use_char_embedding:
                left_enh_repr = []
                for i in range(batch_size):
                    left_enh_repr.append(semantic_enhance_h[i][entity_boundary_idx[i]].unsqueeze(0))
                left_enh_repr = torch.cat(left_enh_repr, dim=0)
                boundary_left_repr = torch.cat([boundary_left_repr, left_enh_repr], dim=2)
                right_enh_repr = []
                for i in range(batch_size):
                    right_enh_repr.append(semantic_enhance_h[i][entity_boundary_idx[i] + 1].unsqueeze(0))
                right_enh_repr = torch.cat(right_enh_repr, dim=0)
                boundary_right_repr = torch.cat([boundary_right_repr, right_enh_repr], dim=2)

            boundary_left_repr = self.dense_boundary_MLP(boundary_left_repr)
            boundary_right_repr = self.dense_boundary_MLP(boundary_right_repr)
            tt_left = boundary_left_repr.unsqueeze(2).unsqueeze(-1).transpose(-2, -1)
            t = torch.matmul(tt_left,
                             self.boundary_Semantic_interaction1_w)
            semantic_interaction_1 = torch.matmul(t, boundary_right_repr.unsqueeze(2).unsqueeze(-1))
            semantic_interaction_1 = semantic_interaction_1.view(batch_size, boundary_count, -1)

            boundary_repr = torch.cat([boundary_left_repr, boundary_right_repr], dim=2)
            boundary_logits = self.boundary_classifier(boundary_repr)
            boundary_logits = semantic_interaction_1 + boundary_logits

        boundary_logits = torch.sigmoid(boundary_logits)
        return boundary_logits

    def _classify_entities(self, encodings, h, entity_masks, context_masks,
                           size_embeddings, semantic_enhance_h, ori_entity_masks):
        # max pool entity candidate spans
        batch_size = h.shape[0]
        entity_count = entity_masks.shape[1]
        device = h.device

        h_large = h.unsqueeze(1).repeat(1, entity_count, 1, 1)

        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = m + h_large
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]
        # entity_spans_pool[entity_spans_pool == -1e30] = 0

        # m = entity_masks.unsqueeze(-1).float()
        # entity_token_count = m.sum(dim=2)
        # entity_token_count[entity_token_count == 0] = 1
        # m[m == 0] = 1e-30
        # # applying avg pooling
        # entity_spans_pool = (m * h_large).sum(dim=2) / entity_token_count

        # entity_repr = entity_spans_pool  # batch_size,entity_count,768
        # get cls token as candidate context representation
        # entity_ctx1 = get_token(h, encodings, self._cls_token)

        # encodings_plus = encodings_plus.view(entity_count * batch_size, -1)
        # context_plus_masks = context_plus_masks.view(entity_count * batch_size, -1)

        # h_plus = self.bert(input_ids=encodings_plus, attention_mask=context_plus_masks)[0]
        # h_plus = h_plus.view(batch_size, entity_count, sent_length, -1)
        # entity_spans_pool = m + h_plus
        # entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        entity_repr = entity_spans_pool

        # create candidate representations including context, max pooled span and size embedding
        # h_plus = [util.extend_tensor(self.bert(input_ids=encodings)[0], [entity_count, 512, 768]) for
        #           encodings in encodings_plus]
        if self._use_sent_ctx:
            # token_count = context_masks.sum(dim=1).unsqueeze(-1)
            # token_count[token_count == 0] = 1
            # m = context_masks.unsqueeze(-1)
            # entity_ctx = (h * m).sum(dim=1) / token_count
            # v = entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1)
            #
            _context_masks = context_masks.unsqueeze(1).repeat(1, entity_count, 1).float().to(device)

            query = torch.matmul(entity_repr, self.W_query).unsqueeze(-1)  # batch_size,entity_count,vector_dim,1
            query = query.unsqueeze(2)
            # batch_size,entity_count,sentence_length,vector_dim,1
            keys = torch.matmul(h_large, self.W_key).unsqueeze(-1)
            keys = keys.transpose(3, 4)

            # batch_size,entity_count,sentence_length
            a = torch.matmul(keys, query).view(batch_size, entity_count, -1)
            a = a * _context_masks
            att = torch.softmax(a, dim=2)
            v = (h_large * att.unsqueeze(-1).repeat(1, 1, 1, h_large.shape[-1]))
            v = v.sum(dim=2)
            v = v.view(batch_size, entity_count, -1)

            entity_repr = torch.cat([v, entity_repr], dim=2)

        if self._use_char_embedding or self._use_pos_tag_embedding or self._use_word_embedding:
            o_m = ori_entity_masks.unsqueeze(-1).float()
            entity_token_count = o_m.sum(dim=2)
            # 1替换0 避免除以0
            entity_token_count[entity_token_count == 0] = 1
            o_m[o_m == 0] = 1e-30
            semantic_enhance_h_large = semantic_enhance_h.unsqueeze(1).repeat(1, entity_count, 1, 1)
            # applying avg pooling
            semantic_enhance_repr = (semantic_enhance_h_large * o_m).sum(dim=2) / entity_token_count
            entity_repr = torch.cat([entity_repr, semantic_enhance_repr], dim=2)

        if self._use_size_embedding:
            entity_repr = torch.cat([entity_repr, size_embeddings], dim=2)

        # classify entity candidates
        entity_repr = self.dropout(entity_repr)
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool

    def _forward_eval(self, encodings: torch.tensor, context_masks: torch.tensor, boundaries_masks: torch.tensor,
                      boundaries_sample_masks: torch.tensor, docs: list, dataset: Dataset, pos_tags: torch.tensor,
                      word_masks: torch.tensor, entity_boundaries_idx: torch.tensor, char_encodings: torch.tensor,
                      char_masks: torch.tensor, word_encodings: torch.tensor, entity_token_masks: torch.tensor,
                      entity_masks: torch.tensor, entity_sample_masks: torch.tensor, ori_entity_masks: torch.tensor,
                      entity_spans: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        self._dataset = dataset
        batch_size = char_encodings.shape[0]
        word_count = char_encodings.shape[1]
        char_count = char_encodings.shape[2]

        semantic_enhance_repr = []

        if self._use_char_embedding:
            char_repr = self.char_embedding(char_encodings)
            char_repr = char_repr.view([batch_size * word_count, char_count, -1])
            char_h = self.char_lstm(char_repr)[0]
            char_h = char_h.view(batch_size, word_count, char_count, -1)
            # m = ((char_masks == 0).float() * (-1e10)).unsqueeze(-1)

            m = char_masks.float()
            # real_char_count = m.sum(dim=2)
            # # 1替换0 避免除以0
            # real_char_count[real_char_count == 0] = 1
            #
            # # 平滑处理 用一个很小的数替代0
            # m[m == 0] = 1e-30
            #
            # # 分母加1避免出现 除以0的情况
            # char_h = (char_h * m).sum(dim=2) / real_char_count
            # t_idx = m.long().sum(dim=2).view(-1) - 1
            # t_idx[t_idx == -1] = 0
            # t_idx = t_idx.tolist()
            # char_h_idx = [[i for i in range(batch_size)] * word_count, [i for i in range(word_count)] * batch_size,
            #               t_idx]

            # char_h = char_h[char_h_idx]
            # char_h = char_h.view(batch_size, word_count, -1)
            char_h = char_h[:, :, char_count - 1, :]
            # char_h = char_h.view(batch_size, word_count, -1)

            semantic_enhance_repr.append(char_h)

        if self._use_pos_tag_embedding:
            pos_repr = self.pos_tag_embeddings(pos_tags)
            semantic_enhance_repr.append(pos_repr)

        if self._use_word_embedding:
            word_repr = self.word_embedding(word_encodings)
            semantic_enhance_repr.append(word_repr)

        semantic_enhance_h = None
        if self._use_word_embedding or self._use_pos_tag_embedding or self._use_char_embedding:
            semantic_enhance_repr = torch.cat(semantic_enhance_repr, dim=2)
            # semantic_enhance_lstm_output, _ = self.Semantic_enhance_lstm(semantic_enhance_repr)
            # semantic_enhance_h = self.dense_Semantic_enhance_Linear(semantic_enhance_lstm_output)
            semantic_enhance_h = semantic_enhance_repr

        context_masks = context_masks.float()
        if self._bert_type == 'bert':
            h = self.bert(input_ids=encodings, attention_mask=context_masks)[0]
            # pos_repr = self.bert(input_ids=pos_tags, attention_mask=pos_tag_masks)[0]
        else:
            h = self.albert(input_ids=encodings, attention_mask=context_masks)[0]
            h = self._dense_h(h)
            # pos_repr = self.bert(input_ids=pos_tags, attention_mask=pos_tag_masks)[0]
        # h = self.dense_token_vector(h)

        # vector, pooler, enc_layers = self.bert(input_ids=encodings, attention_mask=context_masks)
        # h = torch.cat([t.unsqueeze(-1) for t in enc_layers[-4:]], 3).sum(-1)/4

        batch_size = encodings.shape[0]
        ctx_size = context_masks.shape[-1]

        # classify entity word
        entity_token_clf, upper_clf, boundary_clf = None, None, None
        entity_token_masks = entity_token_masks.float().unsqueeze(-1)
        if self._detect_entity_token:
            entity_token_clf = self._classify_entity_token(h)
            entity_token_clf = entity_token_clf * entity_token_masks
        if self._detect_upper:
            upper_clf = self._classify_upper_lower(h)
            upper_clf = upper_clf * entity_token_masks
        # classify boundaries
        if self._detect_boundary:
            boundary_clf = self._classify_boundaries(encodings, h, boundaries_masks, entity_boundaries_idx,
                                                     semantic_enhance_h)

            # classify entities
            # ignore boundary candidates that do not constitute an actual entity (based on classifier)
            entity_spans, entity_masks, entity_sample_masks, entity_sizes, ori_entity_masks = self._filter_boundaries(
                boundary_clf, boundaries_sample_masks, ctx_size, docs)
        entity_sizes = ori_entity_masks.long().sum(-1)
        entity_sample_masks = entity_sample_masks.float().unsqueeze(-1)
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, _ = self._classify_entities(encodings, h, entity_masks, context_masks, size_embeddings,
                                                semantic_enhance_h,
                                                ori_entity_masks)

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        entity_clf = entity_clf * entity_sample_masks  # mask

        # obtain entity logits
        return entity_clf, boundary_clf, entity_token_clf, upper_clf, entity_spans

    def _filter_boundaries(self, boundary_clf, boundary_sample_masks, ctx_size, doc):
        batch_size = boundary_clf.shape[0]
        device = boundary_clf.device
        boundary_sample_count = boundary_clf.shape[1]
        boundary_type_num = boundary_clf.shape[2]

        boundary_clf[boundary_clf < self.boundary_filter_threshold] = 0
        boundary_clf = boundary_clf * boundary_sample_masks.unsqueeze(-1).repeat(1, 1, boundary_type_num).float()

        batch_entities = []
        batch_entity_masks = []
        batch_entity_sample_masks = []
        batch_entity_sizes = []
        batch_ori_entity_masks = []

        for i in range(batch_size):
            token_count = len(doc[i].tokens)
            encodings = doc[i].encoding
            entities = []
            entity_masks = []
            entity_sizes = []
            sample_masks = []
            ori_entity_masks = []

            if self._BD_include_type:
                entity_start_boundaries_idx = boundary_clf[i, :, 0].nonzero().view(-1).tolist()
                entity_end_boundaries_idx = boundary_clf[i, :, 1].nonzero().view(-1).tolist()
            else:
                batch_entity_boundaries = boundary_clf[i].nonzero().view(-1)
                entity_start_boundaries_idx = entity_end_boundaries_idx = batch_entity_boundaries.tolist()

            # create entity and masks ,size
            # print(entity_start_boundaries_idx)
            # print(entity_end_boundaries_idx)
            # print(token_count)

            for start in entity_start_boundaries_idx:
                if self.boundary_represent_mode == 1:
                    # 单token表示边界需要添加长度为1的样本，开始和结束的token是同一个
                    _entity_end_boundaries_idx = [b + 1 for b in entity_end_boundaries_idx if b >= start]
                else:  # self.boundary_represent_mode == 2
                    _entity_end_boundaries_idx = [b for b in entity_end_boundaries_idx if b > start]
                for end in _entity_end_boundaries_idx:
                    if end - start > 15:
                        break
                    span = doc[i].tokens[start:end].span
                    entities.append(span)
                    entity_masks.append(sampling.create_entity_mask(*span, ctx_size))
                    ori_entity_masks.append(sampling.create_entity_mask(start + 1, end + 1, token_count + 2))
                    sample_masks.append(1)
                    entity_sizes.append(end - start)

            if not entities:
                # case: no entities
                batch_entities.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_entity_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_entity_sample_masks.append(torch.tensor([0], dtype=torch.bool))
                batch_entity_sizes.append(torch.tensor([0], dtype=torch.long))
                batch_ori_entity_masks.append(torch.tensor([[0] * (token_count + 2)], dtype=torch.bool))
            else:
                batch_entities.append(torch.tensor(entities, dtype=torch.long))
                batch_entity_masks.append(torch.stack(entity_masks))
                batch_entity_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))
                batch_entity_sizes.append(torch.tensor(entity_sizes, dtype=torch.long))
                batch_ori_entity_masks.append(torch.stack(ori_entity_masks))

            # if not boundary_spans:
            #     batch_boundaries.append(torch.tensor([[0, 0]], dtype=torch.long))
            # else:
            #     batch_boundaries.append(torch.tensor(boundary_spans, dtype=torch.long))

        # stack
        batch_entities = util.padded_stack(batch_entities).to(device)
        batch_entity_masks = util.padded_stack(batch_entity_masks).to(device)
        batch_entity_sample_masks = util.padded_stack(batch_entity_sample_masks).to(device)
        # batch_boundaries = util.padded_stack(batch_boundaries).to(device)

        batch_entity_sizes = util.padded_stack(batch_entity_sizes).to(device)

        batch_ori_entity_masks = util.padded_stack(batch_ori_entity_masks).to(device)

        # batch_entity_sizes = torch.tensor(batch_entity_sizes, dtype=torch.long).to(device)
        # batch_entity_sizes = torch.tensor([item.detach().numpy() for item in batch_entity_sizes]).to(device)

        return batch_entities, batch_entity_masks, batch_entity_sample_masks, batch_entity_sizes, \
               batch_ori_entity_masks

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)


# Model access

_MODELS = {
    'spert': SpERT,
}


def get_model(name):
    return _MODELS[name]
