"""
@Project ：PyTorch-CycleGAN-master
@File    ：my_datasets.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/10/29 21:00
@Des     ：
"""
import os.path
import pickle

import PIL.Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from my_utils.visual_pressure import show_pressure
from my_utils.visual_3d import show_keypoints
from my_utils.visual_2d import show_2d_kpts
from my_utils.gen_2d_kpts import convert_3dto2d
from my_utils.gen_2d_kpts import generate_2d_gt
extra_matrix = np.zeros((1, 256, 64))

def normalize_with_range(data, max, min):
    data = (data-min)/(max-min)
    return data

def remove_keypoint_artifact(data,threshold):
    for q in range(3):
        frame = data[:,q]
        lower_flag = frame < threshold[q*2]
        upper_flag = frame > threshold[q*2 + 1]
        frame[lower_flag] = threshold[q*2]
        frame[upper_flag] = threshold[q*2 +1]
        data[:,q]=frame
    return data

def normize_keypoints(keypoint):
    '''
    Load triangulated, refined and transformed keypoint coordinate
    Build 3d voxel space
    Normalize keypoint in to 0-1 space, and save
    Generate heatmap with distance
    '''

    xyz_range = [[-800,800],[-800,800],[0,2000]]
    size = [16, 16, 20]
    x_range = xyz_range[0]
    y_range = xyz_range[1]
    z_range = xyz_range[2]
    resolution = [(x_range[1]-x_range[0])/size[0], (y_range[1]-y_range[0])/size[1], (z_range[1]-z_range[0])/size[2]]
    b = np.array([[x_range[0], y_range[0], z_range[0]]])
    threshold = [0,1,0,1,0,1]
    keypoint = normalize_with_range((keypoint - b)/resolution, max(size)-1, 0)
    keypoint = remove_keypoint_artifact(keypoint, threshold)
    return keypoint



class PkDataset(Dataset):

    def __init__(self, root, mode, log_name='log.p'):
        self.data_log = pickle.load(open(os.path.join(root, mode, log_name), 'rb'))
        self.root = root
        self.mode = mode
        self.log_name = log_name
        pass

    def normalize_screen_coordinates(self, X, w, h):
        if X.shape[-1] != 2:
            print('X.shape: ', X.shape)
        assert X.shape[-1] == 2
        return X / w * 2 - [1, h / w]

    def __getitem__(self, index):
        # , 'image': image.reshape((1, 3, 640, 720))
        log_index = np.where(self.data_log <= index)[0][-1] # 没有给第二和第三参数，返回一个元组，元组第一个是原数组满足所给条件的下标构成的数组。所以是【0】【-1】
        data_item = pickle.load(open(os.path.join(self.root, self.mode, str(self.data_log[log_index]), str(index)+'.p'), 'rb'))
        norm_kpts_3d = normize_keypoints(data_item[1]).reshape((1, 22, 3))
        raw_kpts_3d = data_item[1].reshape((1, 22, 3))
        pressure = data_item[0].reshape((1, 120, 120))
        image = data_item[2]
        keypoints_2d = data_item[3]
        norm_keypoints_2d = self.normalize_screen_coordinates(keypoints_2d, 1000, 1002)
        pair_item = {'key_points_3d': norm_kpts_3d, 'pressure': pressure, 'key_points_2d': norm_keypoints_2d.reshape(1, 22, 2), 'image': image.reshape((1, 3, 640, 720))}  # 两个都是tensor格式的, (batch, 22,3),(batch, 120, 120)
        return pair_item

    def __len__(self):
        return self.data_log[-1]
        # return 5847 if self.mode == 'train' else 1254



if __name__ == '__main__':
    root = 'D:\\Fms_train\\Dataset_pre\\PIMDataset_faster\\PIMDataset_faster'
    mode = 'train'
    dataset = PkDataset(root, mode)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    for i, data in enumerate(dataloader):
        pressue = data['pressure'].numpy()[0]
        kpts_3d = data['key_points_3d'].numpy()[0] / 10
        kpts_2d = data['key_points_2d'].numpy()[0]
        show_2d_kpts(kpts_2d)
        show_keypoints(kpts_3d)
        show_pressure(pressue)
        print(data)

