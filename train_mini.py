"""
@Project ：Dataset_pre
@File    ：train.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/10/6 19:04
@Des     ：
"""
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

from rawModel.vaTPose import VaTPose
from spatialSoftmax3D import SpatialSoftmax3D
from data_loader_mini import PkDataset
from my_utils.utils import LambdaLR

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_continue', type=bool, default=False, help='Set true if eval time')
parser.add_argument('--eval', type=bool, default=False, help='Set true if eval time')
parser.add_argument('--dataset_dir', type=str, default='D:\\Users\\Gan\\FMS_algorithm\\PIMDataset_faster/', help='Experiment path')
parser.add_argument('--train_dir', type=str, default='./train_output/', help='Experiment path')
parser.add_argument('--test_dir', type=str, default='./test_output/', help='test data path')
parser.add_argument('--exp', type=str, default='singlePeople', help='Name of experiment')
parser.add_argument('--ckpt', type=str, default='singlePeople_0.0001_0_best', help='loaded ckpt file')
parser.add_argument('--epoch', type=int, default=200, help='The time steps you want to subsample the dataset to,500')  # 默认500
parser.add_argument('--decay_epoch', type=int, default=35, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size,128')
parser.add_argument('--weightdecay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--window', type=int, default=0.5, help='window around the time step')
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

if not os.path.exists(args.train_dir + 'ckpts'):  # 通常包含模型的权重参数检查点文件和优化器状态等信息，可以在需要时恢复训练或进行推断。
    os.makedirs(args.train_dir + 'ckpts')

if not os.path.exists(args.train_dir + 'log'):
    os.makedirs(args.train_dir + 'log')

use_gpu = torch.cuda.is_available()
device = 'cuda:0' if use_gpu else 'cpu'

if not args.eval:
    train_path = args.dataset_dir
    mask = []
    train_dataset = PkDataset(root=train_path, mode='train' )  # 训练集的类 window=5，数据序列长度；subsample=1，不采样；
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)  # 训练集的加载类
    print(len(train_dataset))
    val_path = args.dataset_dir
    mask = []
    val_dataset = PkDataset(root=val_path, mode='valid')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(len(val_dataset))

if __name__ == '__main__':

    wandb.init(
        project="VaTPose",
        name="SinglePeople_0.0000_0_0",
    )

    np.random.seed(0)
    torch.manual_seed(0)
    model = VaTPose(args.window)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(args.epoch, 0, args.decay_epoch).step)
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
        best_val_loss = 0.02038
    valid_record_count = 0
    train_record_count = 0
    for epoch in range(epochs + 1, args.epoch):
        start_time = time.time()
        train_loss = []
        val_loss = []
        print('training')
        bar = ProgressBar(len(train_dataloader))
        for i_batch, sample_batched in bar(enumerate(train_dataloader, 0)):
            model.train(True)
            visual = torch.tensor(sample_batched["key_points_2d"], dtype=torch.float, device=device)
            tactile = torch.tensor(sample_batched["pressure"], dtype=torch.float, device=device)
            keypoint = torch.tensor(sample_batched["key_points_3d"], dtype=torch.float, device=device)
            with torch.set_grad_enabled(True):
                keypoint_out = model([visual, tactile])
            loss = criterion(keypoint_out, keypoint)  # 点的损失
            # 三部曲：梯度清零， 损失函数反向传播求导，权重更新优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if i_batch % 300 == 0 and i_batch != 0:
                print("epoch:%d, [%d/%d] LR: %.6f, Loss: %.6f, Keypoint_loss: %.6f, "
                      "k_max_gt: %.6f, k_max_pred: %.6f, k_min_gt: %.6f, k_min_pred: %.6f, " % (
                          epoch, i_batch, len(train_dataloader), get_lr(optimizer), loss.item(), loss,
                          np.amax(keypoint.cpu().data.numpy()), np.amax(keypoint_out.cpu().data.numpy()),
                          np.amin(keypoint.cpu().data.numpy()), np.amin(keypoint_out.cpu().data.numpy()),
                        ))
                print("Now running on val set")
                model.train(False)
                keypoint_l2 = []
                bar = ProgressBar(len(val_dataloader))
                for i_batch, sample_batched in bar(enumerate(val_dataloader, 0)):
                    visual = torch.tensor(sample_batched["key_points_2d"], dtype=torch.float, device=device)
                    tactile = torch.tensor(sample_batched["pressure"], dtype=torch.float, device=device)
                    keypoint = torch.tensor(sample_batched["key_points_3d"], dtype=torch.float, device=device)
                    with torch.set_grad_enabled(False):
                        keypoint_out = model([visual, tactile])
                        # heatmap_out = heatmap_out.reshape(-1, 22, 16, 16, 20)
                        # heatmap_transform = remove_small(heatmap_out.transpose(2, 3), 1e-2, device)
                        # keypoint_out, heatmap_out2 = softmax(heatmap_transform * 10)
                    # loss_heatmap = torch.mean((heatmap_transform - heatmap) ** 2 * (heatmap + 0.5) * 2) * 1000
                    loss_keypoint = criterion(keypoint_out, keypoint)
                    loss = loss_keypoint
                    if i_batch % 39 == 0 and i_batch != 0:
                        #
                        print("[%d/%d] LR: %.6f, Loss: %.6f, Keypoint_loss: %.6f, "
                              "k_max_gt: %.6f, k_max_pred: %.6f, k_min_gt: %.6f, k_min_pred: %.6f" % (
                                  i_batch, len(val_dataloader), get_lr(optimizer), loss.item(),
                                  loss_keypoint,
                                  np.amax(keypoint.cpu().data.numpy()), np.amax(keypoint_out.cpu().data.numpy()),
                                  np.amin(keypoint.cpu().data.numpy()), np.amin(keypoint_out.cpu().data.numpy()),
                            ))
                        #

                    val_loss.append(loss.data.item())
                scheduler.step(np.mean(val_loss)) #
                wandb.log({'loss/valid_loss': np.mean(val_loss)}, step=valid_record_count)
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
                    # torch.save(model.state_dict(), 'model_weights.pth')
            avg_train_loss = np.mean(train_loss)
            avg_val_loss = np.mean(val_loss)
            wandb.log({'loss/train_loss': avg_train_loss, "loss/valid_loss": avg_val_loss}, step=train_record_count)
            train_record_count += 1
            avg_train_loss = np.array([avg_train_loss])
            avg_val_loss = np.array([avg_val_loss])
            train_loss_list = np.append(train_loss_list, avg_train_loss, axis=0)
            val_loss_list = np.append(val_loss_list, avg_val_loss, axis=0)
            to_save = [train_loss_list[1:], val_loss_list[1:]]
            pickle.dump(to_save, open(args.train_dir + 'log/' + args.exp +
                                      '_' + str(args.batch_size) + '_' + str(args.lr) + '_' + str(args.window) + '.p',
                                      "wb"))
        time_elapsed = time.time() - start_time
        # scheduler.step()
        print("[%d] Train Loss: %.6f, Valid Loss: %.6f, time_consume: %.6f m, %.6f s" % (epoch, avg_train_loss, avg_val_loss, time_elapsed // 60, time_elapsed % 60))
