"""
@Project ：Dataset_pre
@File    ：convert_data.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/10/2 17:22
@Des     ：
"""
import pickle
import cv2
import numpy as np
from tqdm import trange


def convert_3dto2d(points_3d):
    # 旋转向量
    R_m = np.array([[ 0.01442425], [-0.05447133],  [-0.01471758]])
    # 平移向量
    T = np.array([[-13.81130582],
                    [ -8.30606223],
                    [ 22.14989537]])
    # 内参矩阵
    K = np.array([[507.23132758,   0.,         638.29784532],
                [ 0.,509.27258349, 372.8912697 ],
                [ 0.,0.,1.]])
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
    # d.pop()
    d[2] = 1
    points_3d = d[1]
    two_d_kpts = convert_3dto2d(points_3d)
    two_d_kpts_n = normalize_screen_coordinates(two_d_kpts, 1280, 720)
    two_d_kpts_n[two_d_kpts_n > 1] = 1
    two_d_kpts_n[two_d_kpts_n < -1] = -1
    d.append(two_d_kpts_n)
    return d

def process_phase(start, end, rootpath):
    pass

def get_one_item(index):
    data_path = 'D:\\Fms_train\\Dataset_pre\\PIMDataset_faster\\PIMDataset_faster\\valid\\0'
    d = pickle.load(open(data_path + '/'+str(index) + '.p', 'rb'))
    print(d)


if __name__ == '__main__':
    # 一个数据项包含：0：（120，120）压力， 1：（22，3）3d，2：（3，640，720）图像，3：（22，2）2d
    data_path = 'D:\\Fms_train\\Dataset_pre\\PIMDataset_faster\\PIMDataset_faster\\valid\\1254'
    train_log_path = 'D:\\Fms_train\\Dataset_pre\\PIMDataset_faster\\PIMDataset_faster\\valid\\temp_log.p'
    """
    # 修改数据集log文件
    log_data = pickle.load(open(train_log_path, 'rb'))
    new_log_data = np.array([0, 1254, 2508])
    with open('D:\\Fms_train\\Dataset_pre\\PIMDataset_faster\\PIMDataset_faster\\valid\\temp_log.p', 'wb') as f:
        pickle.dump(new_log_data, f)
    
    # """

    """
    # convert data # 
    for i in trange(1254, 2058, desc='process dataset'):
        d = pickle.load(open(data_path + '/' + str(i) + '.p', "rb"))
        new_d = process_item(d)
        with open(data_path + '/' + str(i) + '.p', "wb") as f:
            pickle.dump(new_d, f)
    # """
    get_one_item(0)






