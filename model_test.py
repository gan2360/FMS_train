import torch

from progressbar import ProgressBar
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
import numpy as np
from test_dataset import PkDataset
from rawModel.vaTPose import VaTPose
from my_utils.visual_3d import show_keypoints
from my_utils.visual_2d import show_2d_kpts
from rawModel.lib.hrnet.hrnetModel import HRnetModelPrediction
from rawModel.lib.yolov3.yoloModel import YoloModelPrediction
from rawModel.lib.preprocess import h36m_coco_format
import cv2

model_ckpts_path = 'D:/Users/Gan/FMS_algorithm/singlePeople_61_0.0001_0.5_best.path.tar'
data_set_root = 'D:\\Users\\Gan\\FMS_algorithm\\PIMDataset_faster/'
video_path = 'D:\\Users\\Gan\\FMS_system\\fms_predict_sdk_v3-single_process\\fms_predict_sdk_v3-single_process\\fmsTestData'


def calculate_mpjpe(model, test_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    mpjpe = 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # 前向传播获取模型的预测结果
            outputs = model(inputs)
            # 计算MPJPE
            mpjpe += calculate_batch_mpjpe(outputs, targets)
            num_samples += 1

    # 计算平均MPJPE
    mpjpe /= num_samples

    return mpjpe

def calculate_batch_mpjpe(outputs, targets):
    # 在这里实现批量计算MPJPE的逻辑
    # outputs 和 targets 的形状都是 (batch_size, num_joints, 3)
    # 可以根据需要自定义计算MPJPE的方法

    # 这里只是一个示例，计算每个关节的欧氏距离并取平均作为MPJPE
    errors = torch.norm(outputs - targets, dim=-1)
    mpjpe = errors.mean()

    return mpjpe.item()


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


def normalize_screen_coordinates(X, w, h):
    if X.shape[-1] != 2:
        print('X.shape: ', X.shape)
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def get_pose2D(hrnetModel, yoloModel, frame):
    keypoints, scores = hrnetModel.gen_video_kpts(yoloModel, frame, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    keypoints = keypoints.reshape(17,2)
    # keypoints = normalize_screen_coordinates(keypoints, 1280, 720)
    keypoints = rearrange_17to22(keypoints)
    return  keypoints.reshape(22,2)


def denormalize_img(normalized_img):
    img = (normalized_img + 1) / 2
    img = (img * 255).astype(np.uint8)
    return img


def de_normalize_3d(raw_kpts_3d, device):
    b = torch.tensor([-800.0, -800.0, 0.0]).to(device)
    resolution = 100
    scale = 19
    pred_3d = raw_kpts_3d * scale
    pred_3d = pred_3d * resolution + b
    pred_3d = pred_3d / 10
    return pred_3d


def flip_2d(kpts_2d):
    mid_axis  = 720 / 2
    kpts_2d[:, 1] = - kpts_2d[:, 1] + mid_axis * 2
    return kpts_2d


def show_2d_on_images(img, kpts_2d):
    kpts_2d = kpts_2d.tolist()
    BODY_22_pairs = np.array(
        [[14, 13], [13, 0], [14, 19], [14, 16], [19, 20], [20, 21], [16, 17], [17, 18], [0, 1], [1, 2], [2, 3], [0, 7],
         [7, 8], [8, 9], [14, 15], [9, 11], [11, 12], [9, 10],
         [3, 5], [5, 6], [3, 4]])
    node_color = (0, 0, 255)
    node_radius = 2
    line_color = (0, 255, 0)
    for point in kpts_2d:
        x,y = point
        cv2.circle(img, (int(x), int(y)), node_radius, node_color, -1)
    for line in BODY_22_pairs:
        start_point = [int(kpts_2d[line[0]][0]), int(kpts_2d[line[0]][1])]
        end_point = [int(kpts_2d[line[1]][0]), int(kpts_2d[line[1]][1])]
        cv2.line(img, start_point, end_point, line_color, thickness=2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    hrnetModel = HRnetModelPrediction()
    yoloModel = YoloModelPrediction()
    test_dataset = PkDataset(data_set_root, "valid")
    model = VaTPose(0.5)
    checkpoints = torch.load(model_ckpts_path, map_location='cuda:0')
    model.load_state_dict(checkpoints['model_state_dict'], False)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    mpjpe = 0.0
    num_samples = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(tqdm(test_dataloader, 'test_model'), 0):
            target_3d = sample_batched["key_points_3d"].float().to(device)
            input_2d = sample_batched["key_points_2d"].float().to(device)
            input_pressure = sample_batched["pressure"].float().to(device)
            zero_pressure = torch.zeros((1, 1, 120, 120), device=device, dtype=torch.float)
            # input_image = sample_batched["image"].cpu().numpy()[0][0]
            # d_img = denormalize_img(input_image)
            # c_img = d_img.transpose(2, 1, 0)
            # try:
            #     kpts_2d = get_pose2D(hrnetModel, yoloModel, c_img)
            #     norm_kpts_2d = normalize_screen_coordinates(kpts_2d, 1000, 1002)
            #     norm_kpts_2d = norm_kpts_2d.reshape(1, 1, 22, 2)
            #     norm_kpts_2d = torch.from_numpy(norm_kpts_2d).float().to(device)
            # except Exception as e:
            #     print(str(e))
            #     continue
            # c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)
            pred_3d = model([input_2d, input_pressure])
            pred_3d = de_normalize_3d(pred_3d, device)

            pred_3d_z = model([input_2d, zero_pressure])
            pred_3d_z = de_normalize_3d(pred_3d_z, device)

            # temp_mpjpe = calculate_batch_mpjpe(pred_3d, target_3d / 10)
            temp_mpjpe = calculate_batch_mpjpe(pred_3d, pred_3d_z)
            mpjpe += temp_mpjpe
            num_samples += 1
        mpjpe /= num_samples
        print(mpjpe)

