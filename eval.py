'''
Author: your name
Date: 2021-06-22 13:23:25
LastEditTime: 2021-06-23 09:52:14
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \ConvLSTM_pytorch_copy\ConvLSTM_pytorch\eval.py
'''
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
# from torch.nn.modules.loss import _Loss
from loss import myloss, calculate_wgrid


def eval_net(epoch, net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    print(n_val)
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            imgs = imgs.squeeze()
            #imgs = imgs.unsqueeze(0)
            #imgs = imgs.unsqueeze(2)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred__, conv_mask, output = net(imgs)
                mask_pred = mask_pred__[0][:,-1,...]

                #pred = torch.sigmoid(mask_pred)
                #pred = (pred > 0.).float()
                #tot += dice_coeff(pred, true_masks).item()
            w = calculate_wgrid(true_masks)
            tot += myloss()(epoch, mask_pred, true_masks, w.to(device), conv_mask).item()
            pbar.update()

    net.train()
    return tot / n_val
