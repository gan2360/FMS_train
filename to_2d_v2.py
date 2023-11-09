"""
@Project ：Dataset_pre
@File    ：to_2d_v2.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/10/26 18:48
@Des     ：
"""
import pickle

import numpy as np
import torch
from pyquaternion import Quaternion

data_path = 'D:\\Fms_train\\Dataset_pre\\PIMDataset_faster\\PIMDataset_faster\\train\\0'
orientation = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088])
translation = np.array([1.8411070556640625, 4.95528466796875, 1.5634454345703125])

focal_length = np.array([1145.0494384765625, 1143.7811279296875])
center = np.array([512.54150390625, 515.4514770507812])
radial_distortion = np.array([-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043])
tangential_distortion = np.array([-0.0009756988729350269, -0.00142447161488235])


def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]
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


def world_to_camera(X, R, t):
    Rt = wrap(qinverse,
              R)  # Invert rotation, 'from the camera coordinate to the world coordinate' to 'the world coordinate to the camera coordinate'
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t)  # Rotate and translate


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


def world_to_camera(X, R, t):
    Rt = wrap(qinverse,
              R)  # Invert rotation, 'from the camera coordinate to the world coordinate' to 'the world coordinate to the camera coordinate'
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t)  # Rotate and translate


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
def change_one(item):
    pos_3d_world =item[1]
    orientation = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088])
    translation = np.array([1.8411070556640625, 4.95528466796875, 1.5634454345703125])
    focal_length = np.array([1145.0494384765625, 1143.7811279296875])
    center = np.array([512.54150390625, 515.4514770507812])
    radial_distortion = np.array([-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043])
    tangential_distortion = np.array([-0.0009756988729350269, -0.00142447161488235])
    center = normalize_screen_coordinates(center, w=1000, h=1002).astype('float32')
    focal_length = focal_length / 1000 * 2
    intrinsic = np.concatenate((focal_length, center, radial_distortion, tangential_distortion))
    pos_3d = world_to_camera(pos_3d_world, R=orientation,
                             t=translation)  # shape = (frame, 17, 3), 将3d数据从世界坐标转换到相机坐标下
    pos_2d = wrap(project_to_2d, pos_3d, intrinsic, unsqueeze=True)  # shape = (frame, 17, 2), 获取每个相机下的3d->2d映射数据
    pos_2d_pixel_space = image_coordinates(pos_2d, w=1000, h=1002)  # shape = (frame, 17, 2)，转换到像素空间
    positions_2d = pos_2d_pixel_space.astype('float32')
    return positions_2d


if __name__ == '__main__':
    item_0 = pickle.load(open(data_path+'/'+str(0) + '.p', 'rb'))
    two_d = change_one(item_0)
    print(two_d)
    pass
