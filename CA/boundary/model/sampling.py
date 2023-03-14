import random

import torch

from model import util
import math

boundary_len = 2


def create_boundary_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_boundary_dual_mask(start, mid, end, context_size):
    if mid == start:
        start -= 1
    if mid == end:
        end += 1
    left = create_boundary_mask(start, mid, context_size)
    right = create_boundary_mask(mid, end, context_size)
    return left, right


def create_boundary_type_one_hot(boundary_type):
    if 0 == boundary_type:
        one_hot = [0, 0]
    elif 0 < boundary_type < 3:
        one_hot = [0, 0]
        one_hot[boundary_type - 1] = 1
    else:
        one_hot = [1, 1]
    return torch.tensor(one_hot, dtype=torch.float)


def create_boundary_mask_test(tokens, idx, context_size, boundary_represent_mode=1):
    if boundary_represent_mode == 1:
        b_span_start = tokens[idx].span_start
        b_span_end = tokens[idx].span_end
        mask = create_boundary_mask(b_span_start, b_span_end, context_size)

    else:  # boundary_represent_mode == 2
        if idx == 0:
            mid = tokens[0].span_start
        elif idx == len(tokens):
            mid = tokens[-1].span_end
        else:
            mid = tokens[idx].span_start

        start = idx - boundary_len if idx - boundary_len > 0 else 0
        end = idx + boundary_len if idx + boundary_len < len(tokens) else len(tokens)
        tokens = tokens[start:end]

        mask = left, right = create_boundary_dual_mask(tokens.span_start, mid, tokens.span_end, context_size)

    return mask


def isOverlap(region1, region2):
    if region1[0] <= region2[0] < region1[1] or region1[0] < region2[1] <= region1[1]:
        return True
    if region2[0] <= region1[0] < region2[1] or region2[0] < region1[1] <= region2[1]:
        return True
    return False


def culBeta(span, pos_entity_spans):
    Beta = 1.0e+5
    for p_span in pos_entity_spans:
        Beta = min(Beta, (abs(p_span[0] - span[0]) + abs(p_span[1] - span[1])) / (p_span[1] - p_span[0]))
    return Beta


def create_train_sample(doc, neg_entity_count: int, neg_boundary_count: int, entity_type_count: int,
                        boundary_represent_mode: int, BD_include_type: bool, max_span_size: int):
    encodings = doc.encoding
    char_encodings = doc.char_encodings
    word_encodings = doc.word_encodings
    token_count = len(doc.tokens)
    context_size = len(encodings)
    pos_tags = doc.pos_tags

    neg_entity_sample_mode = 2  # 1、组合正确边界 2、穷举所有区域 3、均衡样本

    # weather has upper
    has_upper = torch.tensor(doc.has_upper, dtype=torch.float)

    # positive boundaries
    pos_entity_boundaries = doc.boundaries
    pos_entity_boundary_types, pos_entity_boundary_masks = [], []
    dual_pos_entity_boundary_masks = []
    for boundary in pos_entity_boundaries:
        boundary_mask = create_boundary_mask_test(doc.tokens, boundary.idx, context_size, boundary_represent_mode)
        pos_entity_boundary_types.append(boundary.boundary_type)
        if boundary_represent_mode == 1:
            pos_entity_boundary_masks.append(boundary_mask)
        else:
            left, right = boundary_mask
            dual_pos_entity_boundary_masks.append(torch.cat([left, right]).view(2, context_size))
    # negative boundaries
    neg_entity_boundaries_idx, neg_entity_boundary_types, neg_entity_boundary_masks = [], [], []
    dual_neg_entity_boundary_masks = []
    pos_entity_boundaries_idx = [boundary.idx for boundary in pos_entity_boundaries]

    # 双token表示边界要多一个边界样本
    boundary_sample_count = token_count  # if boundary_represent_mode == 1 else token_count + 1

    neg_entity_boundaries_idx = [i for i in range(boundary_sample_count) if i not in pos_entity_boundaries_idx]
    neg_boundary_sample_count = min(neg_boundary_count, len(neg_entity_boundaries_idx))
    neg_entity_boundaries_idx = random.sample(neg_entity_boundaries_idx, neg_boundary_sample_count)

    for idx in neg_entity_boundaries_idx:
        neg_entity_boundary_types.append(0)  # 0代表负样本
        boundary_mask = create_boundary_mask_test(doc.tokens, idx, context_size, boundary_represent_mode)
        if boundary_represent_mode == 1:
            neg_entity_boundary_masks.append(boundary_mask)
        elif boundary_represent_mode == 2:
            left, right = boundary_mask
            # 双token表示边界要创建左右两边的mask，并拼接在一起
            dual_neg_entity_boundary_masks.append(torch.cat([left, right]).view(2, context_size))

    # merge
    entity_boundaries_idx = pos_entity_boundaries_idx + neg_entity_boundaries_idx
    entity_boundary_types = pos_entity_boundary_types + neg_entity_boundary_types
    entity_boundary_masks = pos_entity_boundary_masks + neg_entity_boundary_masks
    dual_boundary_masks = dual_pos_entity_boundary_masks + dual_neg_entity_boundary_masks
    # 排序
    rank = [index for index, value in sorted(list(enumerate(entity_boundaries_idx)), key=lambda x: x[1])]
    entity_boundaries_idx = [entity_boundaries_idx[i] for i in rank]
    entity_boundary_types = [entity_boundary_types[i] for i in rank]

    if boundary_represent_mode == 2:
        entity_boundary_masks = dual_boundary_masks
    entity_boundary_masks = [entity_boundary_masks[i] for i in rank]

    # positive entities
    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes = [], [], [], []
    pos_entity_ori_spans, pos_ori_entity_masks = [], []

    entity_token_types = torch.zeros(context_size, dtype=torch.float)
    for e in doc.entities:
        pos_entity_spans.append(e.span)
        pos_entity_types.append(e.entity_type.index)
        pos_entity_masks.append(create_entity_mask(*e.span, context_size))
        pos_entity_sizes.append(len(e.tokens))
        start, end = e.span
        entity_token_types[start:end] = 1
        pos_entity_ori_spans.append(e.original_span)
        # 原句子前面加了一个PAD,所以下标要加1
        pos_ori_entity_masks.append(create_entity_mask(e.original_span[0] + 1, e.original_span[1] + 1, token_count + 2))

    # negative entities
    BA_neg_entity_spans, BA_neg_entity_sizes, BA_neg_ori_entity_spans = [], [], []
    neg_entity_span_set = set()

    if BD_include_type:
        # if neg_entity_sample_mode == 3:
        #     start_boundaries_idx = [b for b, t in zip(_entity_boundaries_idx, _entity_boundary_types) if
        #                             t not in [2, 12]]
        #     end_boundaries_idx = [b for b, t in zip(_entity_boundaries_idx, _entity_boundary_types) if
        #                           t not in [1, 11]]
        if neg_entity_sample_mode == 2:
            start_boundaries_idx = [b for b, t in zip(entity_boundaries_idx, entity_boundary_types) if
                                    t != 2]
            end_boundaries_idx = [b for b, t in zip(entity_boundaries_idx, entity_boundary_types) if
                                  t != 1]
        if neg_entity_sample_mode in [1, 3]:
            start_boundaries_idx = [b for b, t in zip(pos_entity_boundaries_idx, pos_entity_boundary_types) if
                                    t != 2]
            end_boundaries_idx = [b for b, t in zip(pos_entity_boundaries_idx, pos_entity_boundary_types) if
                                  t != 1]
    else:
        start_boundaries_idx = end_boundaries_idx = entity_boundaries_idx

    # 通过边界组合获得负样本
    neg_sample_probs = []
    for start in start_boundaries_idx:
        if boundary_represent_mode == 1:
            # 单token表示边界需要添加长度为1的样本（开始和结束的token是同一个）
            _entity_end_boundaries_idx = [b + 1 for b in end_boundaries_idx if b >= start]
        else:  # boundary_represent_mode == 2
            _entity_end_boundaries_idx = [b for b in end_boundaries_idx if b > start]
        for end in _entity_end_boundaries_idx:
            if end - start > max_span_size:
                break
            span = doc.tokens[start:end].span
            if span not in pos_entity_spans and span not in neg_entity_span_set:
                BA_neg_ori_entity_spans.append((start, end))
                BA_neg_entity_spans.append(span)
                BA_neg_entity_sizes.append(end - start)
                neg_entity_span_set.add(span)
                neg_sample_probs.append(1.0)

    # sample negative entities
    if neg_entity_sample_mode != 3:
        neg_entity_samples = random.sample(
            list(zip(BA_neg_entity_spans, BA_neg_entity_sizes, BA_neg_ori_entity_spans)),
            min(len(BA_neg_entity_spans), neg_entity_count))

        neg_entity_spans, neg_entity_sizes, neg_ori_entity_spans = zip(
            *neg_entity_samples) if neg_entity_samples else ([], [], [])

    if neg_entity_sample_mode == 3:
        alpha = 0.9
        offset_neg_ori_entity_spans, offset_neg_entity_spans, offset_neg_entity_sizes = [], [], []
        hard_neg_entity_count = int(neg_entity_count * alpha)

        # direction = [1, 1, 0, 1, -1, 0, -1, -1, 1]
        # direction = [0, 1, 0, -1, 0, ]

        weak_neg_entity_spans, weak_neg_entity_sizes, weak_neg_entity_sizes, weak_neg_ori_entity_spans = [], [], [], []

        for offset_start in start_boundaries_idx + random.sample(neg_entity_boundaries_idx,
                                                                 min(len(neg_entity_boundaries_idx), 10)):
            for offset_end in range(offset_start + 1, min(offset_start + max_span_size, token_count)):
                offset_span = doc.tokens[offset_start:offset_end].span
                if offset_span in pos_entity_spans or offset_span in BA_neg_entity_spans:
                    continue
                Overlap = False
                for pos_span in pos_entity_ori_spans:
                    # 检查和每一个pos span是否重叠
                    if isOverlap((offset_start, offset_end), pos_span):
                        Overlap = True
                        break
                if Overlap:
                    offset_neg_ori_entity_spans.append((offset_start, offset_end))
                    offset_neg_entity_spans.append(offset_span)
                    offset_neg_entity_sizes.append(offset_end - offset_start)
                    neg_entity_span_set.add(offset_span)
                    neg_sample_probs.append(math.exp(-culBeta(offset_span, pos_entity_spans)))
                else:
                    weak_neg_ori_entity_spans.append((offset_start, offset_end))
                    weak_neg_entity_spans.append(offset_span)
                    weak_neg_entity_sizes.append(offset_end - offset_start)
                    neg_entity_span_set.add(offset_span)

        hard_neg_ori_entity_spans = list(BA_neg_ori_entity_spans) + list(offset_neg_ori_entity_spans)
        hard_neg_entity_spans = list(BA_neg_entity_spans) + list(offset_neg_entity_spans)
        hard_neg_entity_sizes = list(BA_neg_entity_sizes) + list(offset_neg_entity_sizes)

        total_hard_neg_samples = list(zip(hard_neg_entity_spans, hard_neg_entity_sizes, hard_neg_ori_entity_spans))
        total_index = list(range(len(total_hard_neg_samples)))
        hard_neg_entity_samples = []
        prob_sum = sum(neg_sample_probs)
        neg_sample_probs = [prob / prob_sum for prob in neg_sample_probs]
        for i in range(min(len(total_hard_neg_samples), hard_neg_entity_count)):
            sample_index = random.choices(total_index, weights=neg_sample_probs, k=1)
            neg_sample_probs[sample_index[0]] = 0  # 将已选择的样本的概率置为0,将不在被选中
            hard_neg_entity_samples.append(total_hard_neg_samples[sample_index[0]])

        # hard_neg_entity_samples = random.sample(
        #     list(zip(hard_neg_entity_spans, hard_neg_entity_sizes, hard_neg_ori_entity_spans)),
        #     min(len(hard_neg_entity_spans), hard_neg_entity_count))
        hard_neg_entity_spans, hard_neg_entity_sizes, hard_neg_ori_entity_spans = zip(
            *hard_neg_entity_samples) if hard_neg_entity_samples else ([], [], [])

        weak_neg_entity_count = int(neg_entity_count * (1 - alpha))
        # sample weak negative entities
        neg_entity_samples = random.sample(
            list(zip(weak_neg_entity_spans, weak_neg_entity_sizes, weak_neg_ori_entity_spans)),
            min(len(weak_neg_entity_spans), weak_neg_entity_count))
        weak_neg_entity_spans, weak_neg_entity_sizes, weak_neg_ori_entity_spans = zip(
            *neg_entity_samples) if neg_entity_samples else ([], [], [])
        # merge negative entities
        neg_entity_spans = list(hard_neg_entity_spans) + list(weak_neg_entity_spans)
        neg_entity_sizes = list(hard_neg_entity_sizes) + list(weak_neg_entity_sizes)
        neg_ori_entity_spans = list(hard_neg_ori_entity_spans) + list(weak_neg_ori_entity_spans)

    # create mask
    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]

    # 原句子前面加了一个PAD,所以下标要加1
    neg_ori_entity_masks = [create_entity_mask(span[0] + 1, span[1] + 1, token_count + 2) for span in
                            neg_ori_entity_spans]
    neg_entity_types = [0] * len(neg_entity_spans)

    # merge all entities samples
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes)
    ori_entity_masks = pos_ori_entity_masks + neg_ori_entity_masks

    assert len(entity_masks) == len(entity_sizes) == len(entity_types)
    assert len(entity_boundary_masks) == len(entity_boundaries_idx) == len(entity_boundary_types)

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)
    max_char_encoding_length = max(len(e) for e in char_encodings)
    char_masks = [[1] * (len(e)) for e in char_encodings]

    # padding
    char_encodings = [e + [0] * (max_char_encoding_length - len(e)) for e in char_encodings]
    char_masks = [e + [0] * (max_char_encoding_length - len(e)) for e in char_masks]

    char_encodings = torch.tensor(char_encodings, dtype=torch.long)
    char_masks = torch.tensor(char_masks, dtype=torch.bool)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)
    entity_token_masks = torch.ones(context_size, dtype=torch.bool)
    entity_token_masks[0:2] = 0
    entity_token_masks[context_size - 2:context_size] = 0

    word_encodings = torch.tensor(word_encodings, dtype=torch.long)

    # 前后做了pad 所以长度要+2
    word_masks = torch.ones(token_count + 2, dtype=torch.bool)

    pos_tags = torch.tensor(pos_tags, dtype=torch.long)

    entity_boundaries_idx = torch.tensor(entity_boundaries_idx, dtype=torch.long)

    if BD_include_type:
        entity_boundary_types = [create_boundary_type_one_hot(i) for i in entity_boundary_types]

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation

    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
        ori_entity_masks = torch.stack(ori_entity_masks)
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        ori_entity_masks = torch.zeros([1, token_count + 2], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    if entity_boundary_masks:
        if BD_include_type:
            entity_boundary_types = torch.stack(entity_boundary_types)
        else:
            entity_boundary_types = torch.tensor(entity_boundary_types, dtype=torch.float)
        entity_boundary_masks = torch.stack(entity_boundary_masks)
        # dual_boundary_masks = torch.stack(dual_boundary_masks)
        entity_boundary_sample_masks = torch.ones([entity_boundary_masks.shape[0]], dtype=torch.bool)
    else:
        entity_boundary_types = torch.zeros([1, 2], dtype=torch.float)
        if boundary_represent_mode == 1:
            entity_boundary_masks = torch.zeros([1, 1, context_size], dtype=torch.bool)
        else:
            entity_boundary_masks = torch.zeros([1, 2, context_size], dtype=torch.bool)
        entity_boundary_sample_masks = torch.zeros([1], dtype=torch.bool)


    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_types=entity_types,
                entity_sample_masks=entity_sample_masks,
                entity_boundary_types=entity_boundary_types, entity_boundary_masks=entity_boundary_masks,
                entity_boundary_sample_masks=entity_boundary_sample_masks, doc=doc,
                pos_tags=pos_tags, word_masks=word_masks, entity_boundaries_idx=entity_boundaries_idx,
                char_encodings=char_encodings, char_masks=char_masks, ori_entity_masks=ori_entity_masks,
                word_encodings=word_encodings, entity_token_types=entity_token_types,
                entity_token_masks=entity_token_masks, has_upper=has_upper)


def create_eval_sample(doc, boundary_represent_mode: int, max_span_size: int):
    encodings = doc.encoding
    char_encodings = doc.char_encodings
    word_encodings = doc.word_encodings
    pos_tags = doc.pos_tags
    token_count = len(doc.tokens)
    context_size = len(encodings)

    has_upper = torch.tensor(doc.has_upper, dtype=torch.float)

    # entity token

    # boundaries candidates
    boundaries_masks = []
    dual_boundary_masks = []

    boundary_sample_count = token_count  # if boundary_represent_mode == 1 else token_count + 1
    for idx in range(boundary_sample_count):
        boundary_mask = create_boundary_mask_test(doc.tokens, idx, context_size, boundary_represent_mode)
        if boundary_represent_mode == 1:
            boundaries_masks.append(boundary_mask)
        elif boundary_represent_mode == 2:
            left, right = boundary_mask
            # 双token表示边界要创建左右两边的mask，并拼接在一起
            dual_boundary_masks.append(torch.cat([left, right]).view(2, context_size))

    if boundary_represent_mode == 2:
        boundaries_masks = dual_boundary_masks

    # entity
    entity_masks, ori_entity_masks, entity_spans = [], [], []
    for start_idx in range(token_count):
        for end_idx in range(start_idx + 1, token_count + 1):
            if end_idx - start_idx > max_span_size:
                break
            start, end = doc.tokens[start_idx:end_idx].span
            entity_masks.append(create_entity_mask(start, end, context_size))
            ori_entity_masks.append(create_entity_mask(start_idx, end_idx, token_count + 2))
            entity_spans.append([start, end])

    entity_boundaries_idx = torch.tensor([i for i in range(boundary_sample_count)], dtype=torch.long)
    pos_tags = torch.tensor(pos_tags, dtype=torch.long)

    # create tensors
    # token indices
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)

    entity_token_masks = torch.ones(context_size, dtype=torch.bool)
    entity_token_masks[0:2] = 0
    entity_token_masks[context_size - 2:context_size] = 0

    max_char_encoding_length = max(len(e) for e in char_encodings)
    char_masks = [[1] * (len(e)) for e in char_encodings]

    # padding
    char_encodings = [e + [0] * (max_char_encoding_length - len(e)) for e in char_encodings]
    char_masks = [e + [0] * (max_char_encoding_length - len(e)) for e in char_masks]

    char_encodings = torch.tensor(char_encodings, dtype=torch.long)
    char_masks = torch.tensor(char_masks, dtype=torch.bool)

    word_encodings = torch.tensor(word_encodings, dtype=torch.long)
    word_masks = torch.ones(token_count + 2, dtype=torch.bool)

    # boundaries
    if boundaries_masks:
        boundaries_masks = torch.stack(boundaries_masks)
        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        boundaries_sample_masks = torch.tensor([1] * boundaries_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no boundaries)
        if boundary_represent_mode == 1:
            boundaries_masks = torch.zeros([1, context_size], dtype=torch.bool)
        else:
            boundaries_masks = torch.zeros([2, context_size], dtype=torch.bool)
        boundaries_sample_masks = torch.zeros([1], dtype=torch.bool)

    # entities
    entity_masks = torch.stack(entity_masks)
    ori_entity_masks = torch.stack(ori_entity_masks)
    entity_sample_masks = torch.ones(entity_masks.shape[0], dtype=torch.bool)
    entity_spans = torch.tensor(entity_spans, dtype=torch.long)

    return dict(encodings=encodings, context_masks=context_masks, boundaries_masks=boundaries_masks,
                boundaries_sample_masks=boundaries_sample_masks, doc=doc,
                pos_tags=pos_tags, word_masks=word_masks, entity_boundaries_idx=entity_boundaries_idx,
                char_encodings=char_encodings, char_masks=char_masks, word_encodings=word_encodings,
                entity_token_masks=entity_token_masks, has_upper=has_upper, entity_masks=entity_masks,
                entity_sample_masks=entity_sample_masks, ori_entity_masks=ori_entity_masks, entity_spans=entity_spans)


def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1  # 要改回来
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()
    for key in keys:
        samples = [s[key] for s in batch]
        if key == 'doc':
            padded_batch[key] = samples

        elif not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:

            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
