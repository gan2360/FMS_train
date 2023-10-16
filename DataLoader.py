"""
@Project ：Dataset_pre
@File    ：DataLoader.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/9/27 15:48
@Des     ：
"""
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import pickle
import time
from torchvision import transforms
from PIL import Image
import cv2
from heatmap_from_keypoints3D import heatmap_from_keypoint


def window_select(log, path, f, idx, window):
    # return visual, tactile, heatmap, keypoint, visual_frame, tactile_frame
    if window == 0:  # window为0的话，直接会返回某个idx的数据
        d = pickle.load(open(path + '/' + str(idx) + '.p', "rb"))
        tactile = d[0]
        keypoint = d[1]
        visual = d[2]
        keypoint, heatmap = heatmap_from_keypoint(keypoint)  # 获取热力图
        return np.reshape(visual, (3, 640, 720)), np.reshape(tactile, (1, 120, 120)), heatmap, keypoint, np.reshape(
            visual, (3, 640, 720)), np.reshape(d[0], (1, 120, 120))
    max_len = log[f + 1]  # log是记录数据集序列的np数组，f代表当前的数据集序列的下标
    min_len = log[f] #
    l = max([min_len, idx - window])
    u = min([max_len, idx + window])

    dh = pickle.load(open(path + '/' + str(idx) + '.p', "rb"))
    keypoint = dh[1]
    keypoint, heatmap = heatmap_from_keypoint(keypoint)

    visual_frame = np.reshape(dh[2], (1, 3, 640, 720))
    tactile_frame = np.reshape(dh[0], (1, 120, 120))

    visual = np.empty((2 * window, 3, 640, 720))
    tactile = np.empty((2 * window, 120, 120))

    if u - l < window:
        tactile = tactile_frame.repeat((2 * window), axis=0)
        visual = visual_frame.repeat((2 * window), axis=0).reshape((-1, 640, 720))
        return visual, tactile, heatmap, keypoint, visual_frame, tactile_frame # 加了frame就是多加一个维度

    if l == min_len:
        for i in range(min_len, min_len + 2 * window):
            d = pickle.load(open(path + '/' + str(i) + '.p', "rb"))
            tactile[i - min_len, :, :] = d[0]
            visual[i - min_len, :, :, :] = d[2]

        visual = visual.reshape((-1, 640, 720))
        return visual, tactile, heatmap, keypoint, visual_frame, tactile_frame

    elif u == max_len:
        for i in range(max_len - 2 * window, max_len):
            d = pickle.load(open(path + '/' + str(i) + '.p', "rb"))
            tactile[i - (max_len - 2 * window), :, :] = d[0]
            visual[i - (max_len - 2 * window), :, :, :] = d[2]

        visual = visual.reshape((-1, 640, 720))
        return visual, tactile, heatmap, keypoint, visual_frame, tactile_frame

    else:
        for i in range(l, u):
            d = pickle.load(open(path + '/' + str(i) + '.p', "rb"))
            tactile[i - l, :, :] = d[0]
            visual[i - l, :, :, :] = d[2]

        visual = visual.reshape((-1, 640, 720))
        return visual, tactile, heatmap, keypoint, visual_frame, tactile_frame


def get_subsample(touch, subsample):  # 用来处理压力矩阵，降低数据，但是保留一定的平均信息
    for x in range(0, touch.shape[1], subsample):
        for y in range(0, touch.shape[2], subsample):
            v = np.mean(touch[:, x:x + subsample, y:y + subsample], (1, 2))
            touch[:, x:x + subsample, y:y + subsample] = v.reshape(-1, 1, 1)

    return touch


class sample_data(Dataset):
    def __init__(self, path, window, mask, subsample):
        self.mask = mask
        self.path = path
        self.window = window
        self.subsample = subsample
        self.log = pickle.load(open(self.path + 'log.p', "rb"))

    def __len__(self):
        if self.mask != []: # mask是对数据集长度的补充
            return self.log[-1] + self.mask[-1]
        else:
            return self.log[-1]
        # return 100

    def __getitem__(self, idx):

        if self.mask != []:
            f = np.where((self.log + self.mask) <= idx)[0][-1]
            local_path = os.path.join(self.path, str(self.log[f]))
            visual, tactile, heatmap, keypoint, visual_frame, tactile_frame = window_select(self.log, local_path, f,
                                                                                            idx - self.mask[f],
                                                                                            self.window)
        else:
            f = np.where(self.log <= idx)[0][-1]  # 找到当前idx所在的数据集序列起始索引
            local_path = os.path.join(self.path, str(self.log[f])) # 拼接分段数据集的根路径
            visual, tactile, heatmap, keypoint, visual_frame, tactile_frame = window_select(self.log, local_path, f,
                                                                                            idx, self.window)

        if self.subsample > 1:
            tactile = get_subsample(tactile, self.subsample)
        return visual, tactile, heatmap, keypoint, visual_frame, tactile_frame, idx


if __name__ == '__main__':
    pass
