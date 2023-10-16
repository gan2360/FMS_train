"""
@Project ：Dataset_pre
@File    ：train.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/10/6 19:04
@Des     ：
"""
from rawModel.vaTPose import VaTPose
import torch.nn as nn
import time
import torch.optim as optim
import numpy as np
import io, os
import argparse
from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import cv2
import h5py
from torch.optim.lr_scheduler import ReduceLROnPlateau
from progressbar import ProgressBar
import wandb

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from model import VaTPose, SpatialSoftmax3D
from data_loader import sample_data

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_continue', type=bool, default=False, help='Set true if eval time')
parser.add_argument('--eval', type=bool, default=False, help='Set true if eval time')
parser.add_argument('--dataset_dir', type=str, default='../Data_Preparation/PIMDataset_faster/', help='Experiment path')
parser.add_argument('--train_dir', type=str, default='./train_output/', help='Experiment path')
parser.add_argument('--test_dir', type=str, default='./test_output/', help='test data path')
parser.add_argument('--exp', type=str, default='singlePeople', help='Name of experiment')
parser.add_argument('--ckpt', type=str, default='singlePeople_0.0001_0_best', help='loaded ckpt file')
parser.add_argument('--epoch', type=int, default=500, help='The time steps you want to subsample the dataset to,500')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size,128')
parser.add_argument('--weightdecay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--window', type=int, default=5, help='window around the time step')
parser.add_argument('--subsample', type=int, default=1, help='subsample tile res')
parser.add_argument('--linkLoss', type=bool, default=False, help='use link loss')
parser.add_argument('--exp_image', type=bool, default=False, help='Set true if export predictions as images')
parser.add_argument('--exp_video', type=bool, default=False, help='Set true if export predictions as video')
parser.add_argument('--exp_data', type=bool, default=False, help='Setrue if export predictions as raw data')
parser.add_argument('--exp_L2', type=bool, default=False, help='Set true if export L2 distance')

args = parser.parse_args()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_spatial_keypoint(keypoint):
    b = np.reshape(np.array([-800, -800, 000]), (1, 1, 3))
    resolution = 100
    max = 20
    spatial_keypoint = keypoint * max * resolution + b
    return spatial_keypoint


def get_keypoint_spatial_dis(keypoint_GT, keypoint_pred):
    dis = get_spatial_keypoint(keypoint_pred) - get_spatial_keypoint(keypoint_GT)
    return dis


def remove_small(heatmap, threshold, device):
    z = torch.zeros(heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], heatmap.shape[3], heatmap.shape[4]).to(device)
    heatmap = torch.where(heatmap < threshold, z, heatmap)
    return heatmap


def check_link(min, max, keypoint, device):  # keypoints,关键点坐标， min: , max:,
    BODY_21_pairs = np.array([
        [14, 13], [13, 0], [14, 19], [14, 16], [19, 20], [20, 21], [16, 17], [17, 18], [0, 1],
        [1, 2], [2, 3], [0, 7], [7, 8], [8, 9], [14, 15], [9, 11], [11, 12], [9, 10], [3, 5], [5, 6], [3, 4]])

    keypoint_output = torch.ones(keypoint.shape[0], 20).to(device)

    for f in range(keypoint.shape[0]):  # 可能为了找到各点之间的平方和
        for i in range(20):

            a = keypoint[f, BODY_21_pairs[i, 0]]
            b = keypoint[f, BODY_21_pairs[i, 1]]
            s = torch.sum((a - b) ** 2)

            if s < min[i]:
                keypoint_output[f, i] = min[i] - s
            elif s > max[i]:
                keypoint_output[f, i] = s - max[i]
            else:
                keypoint_output[f, i] = 0

    return keypoint_output


if not os.path.exists(args.train_dir + 'ckpts'):  # 检查点文件通常包含模型的权重参数和优化器状态等信息，可以在需要时恢复训练或进行推断。
    os.makedirs(args.train_dir + 'ckpts')

if not os.path.exists(args.train_dir + 'log'):
    os.makedirs(args.train_dir + 'log')

use_gpu = torch.cuda.is_available()
device = 'cuda:0' if use_gpu else 'cpu'

if args.linkLoss:
    link_min = pickle.load(open(args.train_dir + 'link_min.p', "rb"))
    link_max = pickle.load(open(args.train_dir + 'link_max.p', "rb"))
    link_min = torch.tensor(link_min, dtype=torch.float, device=device)
    link_max = torch.tensor(link_max, dtype=torch.float, device=device)

# train
if not args.eval:
    train_path = args.dataset_dir + 'train/'
    mask = []
    train_dataset = sample_data(train_path, args.window, mask, args.subsample)  # 训练集的类 window=5，数据序列长度；subsample=1，不采样；
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)  # 训练集的加载类
    print(len(train_dataset))
    val_path = args.dataset_dir + 'valid/'
    mask = []
    val_dataset = sample_data(val_path, args.window, mask, args.subsample)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    print(len(val_dataset))

print(args.exp, args.window, args.subsample, device)

if __name__ == '__main__':

    wandb.init(
        project="VaTPose",
        name="SinglePeople_0.0000_0_0",
    )

    np.random.seed(0)
    torch.manual_seed(0)
    model = VaTPose(args.window)
    softmax = SpatialSoftmax3D(16, 16, 20, 22)
    model.to(device)
    softmax.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, verbose=True)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    criterion = nn.MSELoss()
    epochs = -1
    if args.train_continue:
        checkpoint = torch.load(args.train_dir + 'ckpts/' + args.ckpt + '.path.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
        epochs = checkpoint['epoch']
        loss = checkpoint['loss']
        print("ckpt loaded", loss)
        print("Now continue training")
    train_loss_list = np.zeros((1))
    val_loss_list = np.zeros((1))
    best_keypoint_loss = np.inf
    best_val_loss = np.inf

    if args.train_continue:
        best_val_loss = 0.13

    valid_record_count = 0
    train_record_count = 0
    for epoch in range(epochs + 1, args.epoch):
        train_loss = []
        val_loss = []
        print('training')
        bar = ProgressBar(len(train_dataloader))
        for i_batch, sample_batched in bar(enumerate(train_dataloader, 0)):
            model.train(True)
            visual = torch.tensor(sample_batched[0], dtype=torch.float, device=device)
            tactile = torch.tensor(sample_batched[1], dtype=torch.float, device=device)
            heatmap = torch.tensor(sample_batched[2], dtype=torch.float, device=device)
            keypoint = torch.tensor(sample_batched[3], dtype=torch.float, device=device)
            idx = torch.tensor(sample_batched[6], dtype=torch.float, device=device)

            with torch.set_grad_enabled(True):
                heatmap_out = model([visual, tactile])
                heatmap_out = heatmap_out.reshape(-1, 22, 16, 16, 20)  # 3d点的热力图
                heatmap_transform = remove_small(heatmap_out.transpose(2, 3), 1e-2, device)  # 去掉其中最小的点
                keypoint_out, heatmap_out2 = softmax(heatmap_transform * 10)  #
            loss_heatmap = torch.mean((heatmap_transform - heatmap) ** 2 * (heatmap + 0.5) * 2) * 1000  # 热力图的损失
            loss_keypoint = criterion(keypoint_out, keypoint)  # 点的损失
            if args.linkLoss:  # 点之间连线的损失
                loss_link = torch.mean(check_link(link_min, link_max, keypoint_out, device)) * 10
                loss = loss_heatmap + loss_link
            else:
                loss = loss_heatmap  # 热力图的损失
            # 三部曲：梯度清零， 损失函数反向传播求导，权重更新优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.data.item())
            if i_batch % 185 == 0 and i_batch != 0:

                print("[%d/%d] LR: %.6f, Loss: %.6f, Heatmap_loss: %.6f, Keypoint_loss: %.6f, "
                      "k_max_gt: %.6f, k_max_pred: %.6f, k_min_gt: %.6f, k_min_pred: %.6f, "
                      "h_max_gt: %.6f, h_max_pred: %.6f, h_min_gt: %.6f, h_min_pred: %.6f" % (
                          i_batch, len(train_dataloader), get_lr(optimizer), loss.item(), loss_heatmap, loss_keypoint,
                          np.amax(keypoint.cpu().data.numpy()), np.amax(keypoint_out.cpu().data.numpy()),
                          np.amin(keypoint.cpu().data.numpy()), np.amin(keypoint_out.cpu().data.numpy()),
                          np.amax(heatmap.cpu().data.numpy()), np.amax(heatmap_out.cpu().data.numpy()),
                          np.amin(heatmap.cpu().data.numpy()), np.amin(heatmap_out.cpu().data.numpy())))
                if args.linkLoss:
                    print("loss_heatmap:", loss_heatmap.cpu().data.numpy(),
                          "loss_link:", loss_link.cpu().data.numpy(),
                          "loss_keypoint:", loss_keypoint.cpu().data.numpy())
                print("Now running on val set")
                model.train(False)

                keypoint_l2 = []

                bar = ProgressBar(len(val_dataloader))
                for i_batch, sample_batched in bar(enumerate(val_dataloader, 0)):

                    visual = torch.tensor(sample_batched[0], dtype=torch.float, device=device)
                    tactile = torch.tensor(sample_batched[1], dtype=torch.float, device=device)
                    heatmap = torch.tensor(sample_batched[2], dtype=torch.float, device=device)
                    keypoint = torch.tensor(sample_batched[3], dtype=torch.float, device=device)

                    with torch.set_grad_enabled(False):
                        heatmap_out = model([visual, tactile])
                        heatmap_out = heatmap_out.reshape(-1, 22, 16, 16, 20)
                        heatmap_transform = remove_small(heatmap_out.transpose(2, 3), 1e-2, device)
                        keypoint_out, heatmap_out2 = softmax(heatmap_transform * 10)

                    loss_heatmap = torch.mean((heatmap_transform - heatmap) ** 2 * (heatmap + 0.5) * 2) * 1000
                    loss_keypoint = criterion(keypoint_out, keypoint)

                    if args.linkLoss:
                        loss_link = torch.mean(check_link(link_min, link_max, keypoint_out, device)) * 10
                        loss = loss_heatmap + loss_link
                    else:
                        loss = loss_heatmap

                    # if i_batch % 20 == 0 and i_batch != 0:
                    if i_batch % 39 == 0 and i_batch != 0:
                        #
                        print("[%d/%d] LR: %.6f, Loss: %.6f, Heatmap_loss: %.6f, Keypoint_loss: %.6f, "
                              "k_max_gt: %.6f, k_max_pred: %.6f, k_min_gt: %.6f, k_min_pred: %.6f, "
                              "h_max_gt: %.6f, h_max_pred: %.6f, h_min_gt: %.6f, h_min_pred: %.6f" % (
                                  i_batch, len(val_dataloader), get_lr(optimizer), loss.item(), loss_heatmap,
                                  loss_keypoint,
                                  np.amax(keypoint.cpu().data.numpy()), np.amax(keypoint_out.cpu().data.numpy()),
                                  np.amin(keypoint.cpu().data.numpy()), np.amin(keypoint_out.cpu().data.numpy()),
                                  np.amax(heatmap.cpu().data.numpy()), np.amax(heatmap_out.cpu().data.numpy()),
                                  np.amin(heatmap.cpu().data.numpy()), np.amin(heatmap_out.cpu().data.numpy())))
                        #
                        if args.linkLoss:
                            print("loss_heatmap:", loss_heatmap.cpu().data.numpy(),
                                  "loss_link:", loss_link.cpu().data.numpy(),
                                  "loss_keypoint:", loss_keypoint.cpu().data.numpy())

                    # print("val_loss: ", loss.item())
                    val_loss.append(loss.data.item())

                scheduler.step(np.mean(val_loss))

                wandb.log({'loss/valid_loss': int(np.mean(val_loss))}, step=valid_record_count)
                valid_record_count += 1
                print("val_loss_mean:", np.mean(val_loss))
                if np.mean(val_loss) < best_val_loss:
                    print("new_best_keypoint_l2:", np.mean(val_loss))
                    best_val_loss = np.mean(val_loss)

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss, },
                        args.train_dir + 'ckpts/' + args.exp + '_' + str(args.batch_size) + '_' + str(args.lr)
                        + '_' + str(args.window) + '_best' + '.path.tar')

                    # if local_rank == 0:
                    #     torch.save({
                    #         'epoch': epoch,
                    #         'model_state_dict': model.module.state_dict(),
                    #         'optimizer_state_dict': optimizer.state_dict(),
                    #         'loss': loss,},
                    #     args.train_dir + 'ckpts/' + args.exp + '_' + str(args.lr)
                    #         + '_' + str(args.window) + '_best' + '.path.tar')

            avg_train_loss = np.mean(train_loss)
            avg_val_loss = np.mean(val_loss)
            wandb.log({'loss/train_loss': int(avg_train_loss)}, step=train_record_count)
            train_record_count += 1

            avg_train_loss = np.array([avg_train_loss])
            avg_val_loss = np.array([avg_val_loss])

            train_loss_list = np.append(train_loss_list, avg_train_loss, axis=0)
            val_loss_list = np.append(val_loss_list, avg_val_loss, axis=0)

            to_save = [train_loss_list[1:], val_loss_list[1:]]
            pickle.dump(to_save, open(args.train_dir + 'log/' + args.exp +
                                      '_' + str(args.batch_size) + '_' + str(args.lr) + '_' + str(args.window) + '.p',
                                      "wb"))

        print("[%d] Train Loss: %.6f, Valid Loss: %.6f" % (epoch, avg_train_loss, avg_val_loss))





