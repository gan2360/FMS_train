"""
@Project ：PyTorch-CycleGAN-master
@File    ：gen_2d_kpts.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/10/30 13:46
@Des     ：
"""
import numpy as np
import cv2
import torch
from pyquaternion import Quaternion


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



# 转2d坐标需要用到的中间方法


def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)
    radial = 1 + torch.sum(k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=len(r2.shape) - 1), dim=len(r2.shape) - 1,
                           keepdim=True)
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)
    XXX = XX * (radial + tan) + p * r2
    return f * XXX + c


def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2

def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
    return (v + 2 * (q[..., :1] * uv + uuv))

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
    result = func(*args)
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result

def rotateToQuaternion(rotateMatrix):
    q = Quaternion(matrix=rotateMatrix)
    return np.array([q.x, q.y, q.z, q.w])

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) # Invert rotation, 'from the camera coordinate to the world coordinate' to 'the world coordinate to the camera coordinate'
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) # Rotate and translate


def generate_2d_gt(pos_3d_world):
    res_w = 1280
    res_h = 720
    rotateMatrix = np.array([[0.9629840193487385, 0.2549098739260024, -0.0876512102254038],
                             [-0.037342822039300366, -0.19587481988514502, -0.9799176335677927],
                             [-0.2669593454462486, 0.946918164948794, -0.17910526728412796]])
    translation = np.array([-0.13349062966046233, 0.8989251389492894, 2.2232304848884323])
    orientation = rotateToQuaternion(rotateMatrix)
    focal_length = np.array([531.6271007142337, 532.7617360676755])
    center = np.array([638.0136802738976, 353.18838468381773])
    radial_distortion = np.array([0.04345854346902533, -0.11152720493734536, 0.034112152147019924])
    tangential_distortion = np.array([-0.0002827762029328119, 0.0005317699270945658])
    center = normalize_screen_coordinates(center, w=res_w, h=res_h).astype('float32')
    focal_length = focal_length / res_w * 2
    intrinsic = np.concatenate((focal_length, center, radial_distortion, tangential_distortion))
    pos_3d = world_to_camera(pos_3d_world, R=orientation, t=translation)  # shape = (frame, 17, 3), 将3d数据从世界坐标转换到相机坐标下
    pos_2d = wrap(project_to_2d, pos_3d, intrinsic, unsqueeze=True)  # shape = (frame, 17, 2), 获取每个相机下的3d->2d映射数据
    pos_2d_pixel_space = image_coordinates(pos_2d, w=res_w, h=res_h)  # shape = (frame, 17, 2)，转换到像素空间
    positions_2d = pos_2d_pixel_space.astype('float32')
    return positions_2d

