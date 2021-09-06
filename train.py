'''
Author: your name
Date: 2021-06-17 13:50:35
LastEditTime: 2021-06-24 12:50:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \GitHub\ConvLSTM_pytorch\train.py
'''
'''
Author: your name
Date: 2021-06-17 13:50:35
LastEditTime: 2021-06-17 15:33:42
LastEditors: Please set LastEditors
Description: 根据convlstm预测模型
FilePath: \GitHub\ConvLSTM_pytorch\train.py
'''
import os
import sys
import math
import copy
import logging
import numpy as np
from numpy.lib.type_check import real_if_close
import torch
import argparse
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from eval import eval_net
from convlstm import ConvLSTM
from torch.utils.tensorboard import SummaryWriter
from utils.datasets import Dataset
from torch.utils.data import DataLoader, random_split
from loss import myloss, calculate_wgrid

# from torch.nn.modules.loss import _Loss


dir_train_img = r'/home/developer_11/rain_model/data/csv_file/rain_radar_iou_train_allfile.csv'
dir_valid_img = r'/home/developer_11/rain_model/data/csv_file/rain_radar_iou_valid_allfile.csv'
dir_train_img = r'/home/developer_11/rain_model/data/h5_file/train/train_data_no_norm.h5'
dir_valid_img = r'/home/developer_11/rain_model/data/h5_file/valid/valid_data_no_norm.h5'
dir_checkpoint = 'checkpoints_9_6/'

#! 仿照Python-UNET源码实现convLSTM模型训练
def train_net(net,
              device,
              epochs=1000,
              batch_size=20,
              lr=0.1,# 由于训练集和验证集分开，所以无需指定验证集划分，也算是一种思路，无需将文件分为训练集和验证集（尽量不采用此种形式，与Kaggle不一致）
              save_cp=True):
              #img_scale=0.5):     # img_scale 应当不需要

    train_dataset = Dataset(dir_train_img)
    valid_dataset = Dataset(dir_valid_img)
    n_train = len(train_dataset)  # 获取训练集长度，应该是在记录loss中会用到
    n_val = len(valid_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_DEVICE_{device}_tests_no_linear!')    # 创建Tensorboard实例
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''') # 控制台输出模型相关信息，应该也可以写入配置中

    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True)  # MSE
    # criterion = nn.MSELoss()
    criterion = myloss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                true_masks = true_masks.to(device=device, dtype=mask_type)
                # print(imgs.shape)
                imgs = imgs.squeeze()
                imgs = imgs.unsqueeze(0)
                # imgs = imgs.unsqueeze(2)
                # print(imgs.shape)
                masks_pred_, conv_pred, output = net(imgs)
                #print("masks_pred_ shape is:", masks_pred_[0].shape)
                masks_pred = masks_pred_[0][:, -1,...]
                print("mask_pred shape:",masks_pred.shape)
                #print("pred shape:{}, true shape:{}".format(masks_pred_.shape, true_masks.shape))
                #masks_pred = masks_pred_[0][:,-1,...]
                #masks_pred = masks_pred_
                #print("pred shape:{}, true shape:{}".format(masks_pred.shape, true_masks.shape))
                #masks_pred
                # 此处补充动态求权重
                w = calculate_wgrid(true_masks)
                
                loss = criterion(epoch, masks_pred, true_masks, w.to(device), conv_pred)
                # print(loss.item())
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 20) #TODO: 防止梯度爆炸，进行梯度裁剪
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % 500 == 0:
                # if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        print("*"*10 + "This is a test!" + "*"*10)
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        #writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_loss = eval_net(epoch, net, val_loader, device)     #TODO: 根据在val_loss中的表现决定是否调整学习率
                    # if (epoch % 20)==0:
                    scheduler.step(val_loss)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info('Validation myloss: {}'.format(val_loss))
                    writer.add_scalar('myloss/valid', val_loss, global_step)
                    print(imgs.shape)
                    draw_img = imgs.data.max(dim=1)[0].sum(dim=1).unsqueeze(1)
                    draw_img = draw_img.data/draw_img.data.max()
                    writer.add_images('images', draw_img.data, global_step)
                    #writer.add_images('images', imgs, global_step)
                    true_masks = torch.pow(math.e, ((true_masks *90) -40.695)/16)
                    writer.add_images('true', true_masks, global_step)
                    writer.add_images('pred', output, global_step)
                    writer.add_images('conv', masks_pred, global_step)
                    writer.add_images('mask', conv_pred, global_step)
                    #print(conv_pred.min(), conv_pred.mean(), conv_pred.max())

        if save_cp and epoch%10==0:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}_{device}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
    #writer.add_graph(net, imgs)
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    return parser.parse_args()













if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    #net = ConvLSTM(16, [64,32,1], (3, 3), 3, True, True, False)
    net = ConvLSTM(7, [4,2,1], [(3,3),(3, 3),(1,1)], 3, True, True, False)
    #logging.info(f'Network:\n'
                 #f'\t{net.n_channels} input channels\n'
                 #f'\t{net.n_classes} output channels (classes)\n'
                 #f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device)
                  #img_scale=args.scale)
                  #val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

