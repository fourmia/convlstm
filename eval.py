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
from torch.nn.modules.loss import _Loss
class myloss(_Loss):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, w:torch.Tensor) -> torch.Tensor:
        loss = (w*torch.abs(y_pred-y_true)).sum() / w.sum()
        return loss

def calculate_wgrid(real_grid_device, bins = np.arange(0.0, 20, 0.01)):
    real_grid = real_grid_device.detach().cpu().numpy()
    real_bins_index = np.digitize(real_grid,bins)
    real_frequency = [(len(real_bins_index.reshape(-1)) - len(real_bins_index[real_bins_index== i+1])) for i in range(len(bins))]
    for i in range(len(real_frequency)):
        real_bins_index[real_bins_index==i+1]=real_frequency[i]
    w= real_bins_index * np.piecewise(real_grid, [real_grid<0.1, real_grid>=0.1], [0, 1])
    return torch.from_numpy(w.astype('float32')/w.max())


def eval_net(net, loader, device):
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
                _,mask_pred__ = net(imgs)
                mask_pred = mask_pred__

                #pred = torch.sigmoid(mask_pred)
                #pred = (pred > 0.).float()
                #tot += dice_coeff(pred, true_masks).item()
            w = calculate_wgrid(true_masks)
            tot += myloss()(mask_pred, true_masks, w.to(device)).item()
            pbar.update()

    net.train()
    return tot / n_val
