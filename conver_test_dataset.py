"""
@Project ：Dataset_pre
@File    ：convert_data.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/10/2 17:22
@Des     ：
"""
import os
import pickle
import cv2
import numpy as np
from tqdm import trange
from my_utils.visual_2d import show_2d_kpts
from my_utils.visual_3d import show_keypoints as show_3d_kpts
from my_utils.visual_pressure import show_pressure
from ex_files.to_2d_v2 import change_one

test_dataset_root = 'D:\\Users\\Gan\\FMS_algorithm\\PIMDataset\\PIMDataset\\test'

def convert_3dto2d(points_3d):
    # 旋转向量
    R_m = np.array([[1.7942425], [-0.8447133], [0.55471758]])
    # 平移向量
    T = np.array([[50.81130582],
                    [ 78.30606223],
                    [ 185.14989537]])
    # 内参矩阵
    K = np.array([[507.23132758, 0., 638.29784532],
                  [0., 509.27258349, 372.8912697],
                  [0., 0., 1.]])
    # 畸变系数
    dis_n = np.array([[ 0.05535079, -0.12507061,  0.00194982, -0.00197897,  0.04375391]])
    points_2d, _ = cv2.projectPoints(points_3d, R_m, T, K, dis_n)
    points_2d = points_2d.reshape(-1, 2)
    return points_2d

def normalize_screen_coordinates(X, w, h):  # 把像素坐标归一化到-1和1之间
    if X.shape[-1] != 2:
        print('X.shape: ', X.shape)
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]

def process_item(d): # 一个数据对
    if len(d) >= 4: # 如果已经添加了2d坐标点，就直接返回
        return d
    points_3d = d[1] # 获取3d的gt
    two_d_kpts = change_one(points_3d / 1000)
    # two_d_kpts_n = normalize_screen_coordinates(two_d_kpts, 1280, 720)  # 归一化2d坐标
    d.append(two_d_kpts)
    return d


def get_one_item(index):
    log_data = pickle.load(open(test_dataset_root + '/log.p', 'rb'))
    mid_index = np.where(log_data <= index)[0][-1]
    d = pickle.load(open(os.path.join(test_dataset_root, str(log_data[mid_index]), str(index) + '.p'), 'rb'))
    return d

def write_one_item(index, new_d):
    log_data = pickle.load(open(test_dataset_root + '/log.p', 'rb'))
    mid_index = np.where(log_data <= index)[0][-1]
    with open(os.path.join(test_dataset_root, str(log_data[mid_index]), str(index) + '.p'), 'wb') as f:
        pickle.dump(new_d, f)



if __name__ == '__main__':
    # 一个数据项包含：0：（120，120）压力， 1：（22，3）3d，2：（3，640，720）图像，3：（22，2）2d

    log_data = pickle.load(open(test_dataset_root + '/log.p', 'rb')) # [0, 1253, 2506, 3802, 5098, 5175]
    count = 0
    for i in trange(log_data[0], log_data[-1]):
        # d = get_one_item(i)
        # d = process_one_item
        # write_one_item
        pass












