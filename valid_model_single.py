import os
import pickle
import numpy as np
import torch
from rawModel.vaTPose import VaTPose
from my_utils.visual_2d import show_2d_kpts
from my_utils.visual_3d import show_keypoints
from my_utils.visual_pressure import show_pressure



model_path = 'D:/Users/Gan/FMS_algorithm/Model_train/FMS_train/train_output/ckpts/singlePeople_32_0.0001_0.5_best.path.tar'
# model_path = 'D:/Users/Gan/FMS_algorithm/singlePeople_61_0.0001_0.5_best.path.tar'
dataset_path_train = 'D:/Users/Gan/FMS_algorithm/PIMDataset_faster/train/0'
dataset_path_valid = 'D:/Users/Gan/FMS_algorithm/PIMDataset_faster/valid/0'
pth_path = 'D:/Users/Gan/FMS_algorithm/Model_train/FMS_train/model_weights.pth'


def normalize_screen_coordinates(X, w, h):
    if X.shape[-1] != 2:
        print('X.shape: ', X.shape)
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def get_one_item(idx):
    d = pickle.load(open(os.path.join(dataset_path_valid, str(idx) + '.p'), 'rb'))
    return d


def normalize_3d(keypoints_pred):
    raw_keypoint_out = keypoints_pred.numpy()[0, 0]
    b = [-800, -800, 0]
    resolution = 100
    scale = 19
    keypoint_out = raw_keypoint_out * scale
    keypoint_out = keypoint_out * resolution + b
    keypoint_out = keypoint_out / 10
    return keypoint_out

def cal_mpjpe(x, y):
    """
    :param x: np格式 （x， x， 22， 3）
    :param y: np格式 （x， x， 22， 3）
    :return: mpjpe
    """

    dist = np.linalg.norm(x - y, axis=-1)
    single_mpjpe = np.mean(dist)
    return single_mpjpe


if __name__ == '__main__':
    checkpoints = torch.load(model_path, map_location='cuda:0')
    vat_pose = VaTPose(0.5)
    vat_pose.load_state_dict(checkpoints['model_state_dict'], False)
    with torch.no_grad():
        vat_pose.eval()
        for i in range(900, 905):
            item = get_one_item(i)
            visual = normalize_screen_coordinates(item[3], 1000, 1002)
            # visual = item[3].reshape((1, 1, 22, 2)).astype(np.float32)
            visual = visual.reshape((1, 1, 22, 2)).astype(np.float32)
            zero_pressure = np.zeros((1, 1, 120, 120)).astype(np.float32)
            pressure = item[0].reshape((1, 1, 120, 120)).astype(np.float32)
            keypoints = item[1]
            keypoints_pred = vat_pose([torch.from_numpy(visual), torch.from_numpy(pressure)])
            keypoint_out = normalize_3d(keypoints_pred)
            show_keypoints(keypoint_out)
            show_keypoints(keypoints / 10)
            # show_2d_kpts(item[3])




