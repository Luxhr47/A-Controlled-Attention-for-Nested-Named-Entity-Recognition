from abc import ABC

import torch
from torch.nn import functional as F

import torch


# 二分类
class Focal_loss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=0, size_average=True):
        super(Focal_loss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor([alpha])
        self.size_average = size_average

    def forward(self, logits, label):
        # logits:[b,h,w] label:[b,h,w]
        label_count = label.shape[-1]
        pred = logits.sigmoid()
        pred = pred.view(-1)  # b*h*w
        label = label.view(-1)

        if self.alpha:
            self.alpha = self.alpha.type_as(pred.data)
            alpha_t = self.alpha * label + (1 - self.alpha) * (1 - label)  # b*h*w

        pt = pred * label + (1 - pred) * (1 - label)
        diff = (1 - pt) ** self.gamma

        FL = -1 * alpha_t * diff * pt.log()
        FL = FL.view(-1, 2)

        return FL

        # if self.size_average:
        #     return FL.mean()
        # else:
        #     return FL.sum()


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SpERTLoss(Loss):
    def __init__(self, boundary_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm):
        self._boundary_criterion = boundary_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, entity_logits, boundary_logits, entity_token_logits, upper_logits, entity_types,
                entity_boundary_types,
                entity_token_types, entity_sample_masks, entity_boundary_sample_masks, entity_token_masks, has_upper):

        entity_token_loss, upper_loss, boundary_loss = 0, 0, 0
        # boundary loss
        if boundary_logits is not None:
            boundary_sample_masks = entity_boundary_sample_masks.view(-1).float()
            boundary_count = boundary_sample_masks.sum()
            boundary_logits = boundary_logits.view(-1, boundary_logits.shape[-1])
            boundary_types = entity_boundary_types.view(boundary_logits.shape[0], -1)
            boundary_loss = self._boundary_criterion(boundary_logits, boundary_types)
            boundary_loss = boundary_loss.sum(-1) / boundary_loss.shape[-1]
            boundary_loss = (boundary_loss * boundary_sample_masks).sum() / boundary_count

        entity_token_sample_masks = entity_token_masks.view(-1).float()
        # entity token loss
        if entity_token_logits is not None:
            entity_token_logits = entity_token_logits.view(-1)
            entity_token_types = entity_token_types.view(-1)
            entity_token_loss = self._boundary_criterion(entity_token_logits, entity_token_types)
            entity_token_loss = (entity_token_loss * entity_token_sample_masks).sum() / entity_token_sample_masks.sum()
        # token upper
        if upper_logits is not None:
            upper_logits = upper_logits.view(-1)
            has_upper = has_upper.view(-1)
            upper_loss = self._boundary_criterion(upper_logits, has_upper)
            upper_loss = (upper_loss * entity_token_sample_masks).sum() / entity_token_sample_masks.sum()

        entity_sample_masks = entity_sample_masks.view(-1).float()
        entity_count = entity_sample_masks.sum()
        if entity_count.item() != 0:
            # entity loss
            entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
            entity_types = entity_types.view(-1)

            entity_loss = self._entity_criterion(entity_logits, entity_types)
            entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

            # joint loss
            train_loss = entity_loss + boundary_loss + 0.5 * entity_token_loss + 0.5 * upper_loss
        else:
            # corner case: no positive/negative entity samples
            train_loss = boundary_loss + 0.5 * entity_token_loss + 0.5 * upper_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()
