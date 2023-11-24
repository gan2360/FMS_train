import pickle
import cv2
import numpy as np
import torch

from rawModel.lib.preprocess import h36m_coco_format
from rawModel.lib.hrnet.hrnetModel import HRnetModelPrediction
from rawModel.lib.yolov3.yoloModel import YoloModelPrediction
from rawModel.vaTPose import VaTPose
from my_utils.visual_3d import show_keypoints
from my_utils.visual_2d import show_2d_kpts

video_path = 'D:/Users/Gan/FMS_algorithm/1.mp4'
model_ckpts_path = 'D:/Users/Gan/FMS_algorithm/singlePeople_61_0.0001_0.5_best.path.tar'


class LocalCamera:
    def __init__(self):
        self.camera = cv2.VideoCapture(video_path)

    def get_frame(self):
        return self.camera.read()[1]


local_camera = LocalCamera()
pressure = np.zeros((1, 1, 120, 120)).astype(np.float32)


def denormalize_img(normalized_img):
    img = (normalized_img + 1) / 2
    img = (img * 255).astype(np.uint8)
    return img


def show_1_img():
    file_path = 'D:/Users/Gan/FMS_algorithm/PIMDataset_faster/valid/0/0.p'
    d = pickle.load(open(file_path, 'rb'))
    image = d[2]
    d_img = denormalize_img(image)
    c_img = np.transpose(d_img, (2,1,0))
    cv2.imshow('img', c_img)
    cv2.waitKey(0)

def get_one_item():
    file_path = 'D:/Users/Gan/FMS_algorithm/PIMDataset_faster/valid/0/0.p'
    d = pickle.load(open(file_path, 'rb'))
    return d

def rearrange_17to22(kp2d):
    kp2d_new = []
    temp = [0, 0]
    kp2d_new.append(kp2d[0])
    kp2d_new.append(kp2d[1])
    kp2d_new.append(kp2d[2])
    kp2d_new.append(kp2d[3])
    kp2d_new.append(kp2d[3])
    kp2d_new.append(kp2d[3])
    kp2d_new.append(kp2d[3])
    # kp2d_new.append(temp)
    # kp2d_new.append(temp)
    # kp2d_new.append(temp)

    kp2d_new.append(kp2d[4])
    kp2d_new.append(kp2d[5])
    kp2d_new.append(kp2d[6])
    kp2d_new.append(kp2d[6])
    kp2d_new.append(kp2d[6])
    kp2d_new.append(kp2d[6])
    # kp2d_new.append(temp)
    # kp2d_new.append(temp)
    # kp2d_new.append(temp)

    kp2d_new.append(kp2d[7])
    kp2d_new.append(kp2d[8])
    kp2d_new.append(kp2d[10])

    kp2d_new.append(kp2d[11])
    kp2d_new.append(kp2d[12])
    kp2d_new.append(kp2d[13])

    kp2d_new.append(kp2d[14])
    kp2d_new.append(kp2d[15])
    kp2d_new.append(kp2d[16])

    return np.array(kp2d_new)


def get_pose2D(hrnetModel, yoloModel, frame):
    keypoints, scores = hrnetModel.gen_video_kpts(yoloModel, frame, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    keypoints = keypoints.reshape(17,2)
    # keypoints = normalize_screen_coordinates(keypoints, 1280, 720)
    keypoints = rearrange_17to22(keypoints)
    return  keypoints.reshape(22,2)


def normalize_screen_coordinates(X, w, h):
    if X.shape[-1] != 2:
        print('X.shape: ', X.shape)
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def de_normalize_3d(raw_kpts_3d):
    b = torch.tensor([-800.0, -800.0, 0.0]).to(device)
    resolution = 100
    scale = 19
    pred_3d = raw_kpts_3d * scale
    pred_3d = pred_3d * resolution + b
    pred_3d = pred_3d / 10
    return pred_3d


if __name__ == '__main__':
    device = 'cuda:0'
    pressure = np.zeros((1, 1, 120, 120)).astype(np.float32)
    pressure = torch.from_numpy(pressure).to(device)
    hrnetModel = HRnetModelPrediction()
    yoloModel = YoloModelPrediction()
    img = local_camera.get_frame()
    model = VaTPose(0.5)
    checkpoints = torch.load(model_ckpts_path, map_location='cuda:0')
    model.load_state_dict(checkpoints['model_state_dict'], False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        kpts_2d = get_pose2D(hrnetModel, yoloModel, img)
        norm_kpts_2d = normalize_screen_coordinates(kpts_2d, 1000, 1002)
        norm_kpts_2d = norm_kpts_2d.reshape(1, 1, 22, 2)
        norm_kpts_2d = torch.from_numpy(norm_kpts_2d).float().to(device)
        pred_3d = model([norm_kpts_2d, pressure])
        b = torch.tensor([-800.0, -800.0, 0.0]).to(device)
        resolution = 100
        scale = 19
        pred_3d = pred_3d * scale
        pred_3d = pred_3d * resolution + b
        pred_3d = pred_3d / 10
    d = get_one_item()
    target_3d = d[1]
    show_keypoints(target_3d / 10)

