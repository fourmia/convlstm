'''
Author: your name
Date: 2021-06-23 15:49:26
LastEditTime: 2021-06-29 10:28:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \ConvLSTM_pytorch_copy\loss.py
'''
import sys
import math
import torch
import numpy as np
import pytorch_msssim
from torch.nn.modules.loss import _Loss


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x >= threshold).type(x.dtype)
    else:
        return x


def calculate_wgrid(real_grid_device):
    real_grid = real_grid_device.detach().cpu().numpy()

    # w= real_bins_index * np.piecewise(real_grid, [real_grid<0.1, real_grid>=0.1], [0, 1])
    w= np.piecewise(real_grid, [real_grid<=2, (real_grid>=2)&(real_grid<5),(real_grid>=5)&(real_grid<10),
				(real_grid>=10)&(real_grid<30),real_grid>=30], [1,2,5,10,30])
    #print(np.unique(w))
    #print("w.shape:{}, w.max:{}, w.min:{}".format(w.shape, w.max(), w.min()))
    return torch.from_numpy(w)


def iou(pr, gt, eps=1e-7, threshold=0.5):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    gt = _threshold(gt, threshold=0.1)
    pr, gt = _take_channels(pr, gt)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union





class JaccardLoss(_Loss):

    def __init__(self, eps=1E-3, threshold=0.1):
        super().__init__()
        self.eps = eps
        self.threshold = threshold

    def forward(self, y_pr, y_gt):
        #y_pr = self.activation(y_pr)
        return 1 - iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold
        )
"""
class myloss(_Loss):

    def __init__(self):
        super().__init__()

    def forward(self, epochs, y_pred: torch.Tensor, y_true: torch.Tensor, w:torch.Tensor, conv_mask) -> torch.Tensor:
        #loss = 4*(w*torch.abs(y_pred-y_true)).sum() / (w.sum()+0.01) + 2 * JaccardLoss().forward(conv_mask, y_true)
        #loss = 1*(w*torch.abs(y_pred-y_true)).sum() / (w.sum()+0.01) + 2 * DiceLoss('binary').forward(conv_mask, y_true)
        # print(y_pred, y_true)
        # if epochs <50:
        #     mse_coefficient,iou_coefficient,dice_coefficient,l1_coefficient = 1,1,1,0     #1,1,1,0
        # elif (epochs>=50) and (epochs <100):
        #     mse_coefficient,iou_coefficient,dice_coefficient,l1_coefficient = 1.6,0.7,0.7,0             # 1.3, 0.7, 1, 0
        # elif(epochs>=100) and (epochs <200):
        #     mse_coefficient,iou_coefficient,dice_coefficient,l1_coefficient = 2,0.5,0.5,0            #1.8, 0.5, 0.7, 0
        # elif(epochs>=300) and (epochs <400):
        #     mse_coefficient,iou_coefficient,dice_coefficient,l1_coefficient = 2.8,0.1,0.1,0                   #2.8, 0.1, 0.1, 0
        # else:
        #     mse_coefficient,iou_coefficient,dice_coefficient,l1_coefficient = 3,0,0,0
        loss = 1 - pytorch_msssim.ms_ssim(y_pred.unsqueeze(0).unsqueeze(1), y_true.unsqueeze(0).unsqueeze(1), data_range=1, size_average=False)
        # loss = pytorch_msssim.msssim(y_pred, y_true)
        # loss = (w*torch.abs(y_pred-y_true)).sum() + JaccardLoss().forward(y_pred, y_true) + DiceLoss('binary').forward(conv_mask, y_true)
        return loss
"""


class myloss(_Loss):

    def __init__(self):
        super().__init__()

    def forward(self, epochs, y_pred: torch.Tensor, y_true: torch.Tensor, w:torch.Tensor, conv_mask) -> torch.Tensor:
        print('begin_pred:', y_pred.max(), y_pred.min(), y_pred.mean())
        y_pred = torch.pow(10, (((y_pred*70-10)- 17.67)/16)/10)
        print('end_pred:', y_pred.max(), y_pred.min(), y_pred.mean())
        print('begin_true:', y_true.max(), y_true.min(), y_true.mean())
        y_true = torch.pow(math.e, ((y_true *90) -40.695)/16)
        print('end_true:', y_true.max(), y_true.min(), y_true.mean())
        # loss = (w*torch.abs(y_pred-y_true)).mean()
        # loss = w*torch.nn.MSELoss()(y_pred, y_true) 
        #loss = 1*(w*torch.abs(y_pred-y_true)).sum() / (w.sum()+0.01) + 2 * DiceLoss('binary').forward(conv_mask, y_true)
        # print(y_pred, y_true)
        if epochs <50:
            mse_coefficient,iou_coefficient,dice_coefficient,l1_coefficient = 1,1,1,1     #1,1,1,0
        elif (epochs>=50) and (epochs <100):
            mse_coefficient,iou_coefficient,dice_coefficient,l1_coefficient = 1.6,0.7,0.7,1             # 1.3, 0.7, 1, 0
        elif(epochs>=100) and (epochs <200):
            mse_coefficient,iou_coefficient,dice_coefficient,l1_coefficient = 2,0.5,0.5,1            #1.8, 0.5, 0.7, 0
        elif(epochs>=300) and (epochs <400):
            mse_coefficient,iou_coefficient,dice_coefficient,l1_coefficient = 2.5,0,0,1.5                   #2.8, 0.1, 0.1, 0
        else:
            mse_coefficient,iou_coefficient,dice_coefficient,l1_coefficient = 2,0,0,2
        loss = mse_coefficient*torch.nn.MSELoss()(y_pred, y_true) + iou_coefficient*JaccardLoss().forward(y_pred, y_true) + dice_coefficient*DiceLoss('binary').forward(conv_mask, y_true) + 0*(1 - pytorch_msssim.ms_ssim(y_pred.unsqueeze(0).unsqueeze(1), y_true.unsqueeze(0).unsqueeze(1), data_range=1, size_average=False))
        # loss = (w*torch.abs(y_pred-y_true)).sum() + JaccardLoss().forward(y_pred, y_true) + DiceLoss('binary').forward(conv_mask, y_true)
        return loss


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
BINARY_MODE = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"

__all__ = ["DiceLoss"]


def soft_dice_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score

def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x

class DiceLoss(_Loss):

    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """Implementation of Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error 
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # H, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()

