import os
import pickle
import numpy as np
import torch
from rawModel.vaTPose import VaTPose
from my_utils.visual_2d import show_2d_kpts
from my_utils.visual_3d import show_keypoints
from my_utils.visual_pressure import show_pressure



model_path = 'D:/Users/Gan/FMS_algorithm/Model_train/FMS_train/train_output/ckpts/singlePeople_32_0.001_0.5_best.path.tar'
model_path = 'D:/Users/Gan/FMS_algorithm/singlePeople_61_0.0001_0.5_best.path.tar'
dataset_path_train = 'D:/Users/Gan/FMS_algorithm/Model_train/FMS_train/PIMDataset_faster/train/0'
dataset_path_valid = 'D:/Users/Gan/FMS_algorithm/Model_train/FMS_train/PIMDataset_faster/valid/0'
pth_path = 'D:/Users/Gan/FMS_algorithm/Model_train/FMS_train/model_weights.pth'

def get_one_item(idx):
    d = pickle.load(open(os.path.join(dataset_path_valid, str(idx) + '.p'), 'rb'))
    return d

if __name__ == '__main__':
    checkpoints = torch.load(model_path, map_location='cuda:0')
    vat_pose = VaTPose(0.5)
    vat_pose.load_state_dict(checkpoints['model_state_dict'], False)
    # visuals = []
    # pressures = []
    # keypoints = get_one_item(5)[1]
    # for i in range(10):
    #     item = get_one_item(i)
    #     visuals.append(item[3])
    #     pressures.append(item[0])
    # visuals = np.array(visuals).reshape((1, 10, 22, 2)).astype(np.float32)
    # pressures = np.array(pressures).reshape((1, 10, 120, 120)).astype(np.float32)
    with torch.no_grad():
        vat_pose.eval()
        for i in range(1000, 1002):
            item = get_one_item(i)
            visual = item[3].reshape((1, 1, 22, 2)).astype(np.float32)
            zero_pressure = np.zeros((1, 1, 120, 120)).astype(np.float32)
            pressure = item[0].reshape((1, 1, 120, 120)).astype(np.float32)
            # if i == 1000:
            #     pressure = pressure
            # else:
            #     pressure = zero_pressure
            keypoints = item[1]
            keypoints_pred = vat_pose([torch.from_numpy(visual), torch.from_numpy(pressure)])
            keypoint_out = keypoints_pred.numpy()[0, 0]
            b = [-800, -800, 0]
            resolution = 100
            scale = 19
            keypoint_out = keypoint_out * scale
            keypoint_out = keypoint_out * resolution + b
            keypoint_out = keypoint_out / 10
            show_keypoints(keypoint_out)
            show_keypoints(keypoints / 10)
            show_2d_kpts(item[3])




