# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def smooth_l1_loss(cls_score,
            label,
            beta=1.0,
            weight=None,
            reduction='mean',
            avg_factor=None):
    r"""Sigmoid focal loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The ground truth label of the prediction with
            shape (N, \*).
        weight (torch.Tensor, optional): Sample-wise loss weight with shape
            (N, ). Defaults to None.
        gamma (float): The gamma for calculating the modulating factor.
            Defaults to 2.0.
        alpha (float): A balanced form for Focal Loss. Defaults to 0.25.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' ,
            loss is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: Loss.
    """
    assert cls_score.shape == \
        label.shape, 'pred and target should be in the same shape.'    
    loss = F.smooth_l1_loss(
        cls_score, label, beta=beta, reduction='none')
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if cls_score.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    """Focal loss.

    Args:
        gamma (float): Focusing parameter in focal loss.
            Defaults to 2.0.
        alpha (float): The parameter in balanced form of focal
            loss. Defaults to 0.25.
        reduction (str): The method used to reduce the loss into
            a scalar. Options are "none" and "mean". Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self,
                 beta=1.0,
                 reduction='mean',
                 loss_weight=1.0):

        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        r"""Sigmoid focal loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, \*).
            target (torch.Tensor): The ground truth label of the prediction
                with shape (N, \*).
            weight (torch.Tensor, optional): Sample-wise loss weight with shape
                (N, \*). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss into a scalar. Options are "none", "mean" and "sum".
                Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * smooth_l1_loss(
            cls_score=cls_score,
            label=label,
            weight=weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls
