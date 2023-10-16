import numpy as np
from numpy import float32
from math import log10, floor
import math

def normalize_with_range(data, max, min):
    data = (data-min)/(max-min)
    return data

def softmax(x):
    output = np.exp(x) / np.sum(np.exp(x))
    return output

def round_to_1(data, sig):
    flag = np.where(data==0)
    data[flag] = int(0)
    c,x,y,z = np.where(data>0)
    for i in range(c.shape[0]):
        if data[c[i],x[i],y[i],z[i]] < 1e-2:
            data[c[i],x[i],y[i],z[i]] = 0
        else:
            data[c[i],x[i],y[i],z[i]] = round(data[c[i],x[i],y[i],z[i]], sig-int(floor(log10(abs(data[c[i],x[i],y[i],z[i]]))))-1)
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

def gaussian(dis, mu, sigma):
    return 1/(mu * np.sqrt(2 * math.pi)) * np.exp(-0.5 * ((dis - mu) /sigma)**2)

def heatmap_from_keypoint(keypoint): # (22, 3)
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

    pos_y, pos_x, pos_z = np.meshgrid(
        np.linspace(0., 1., int(size[0])),
        np.linspace(0., 1., int(size[1])),
        np.linspace(0., 1., int(size[2])))

    heatmap = np.zeros((22,int(size[0]),int(size[1]),int(size[2])), dtype=float32)

    b = np.array([[x_range[0], y_range[0], z_range[0]]])
    threshold = [0,1,0,1,0,1]
    keypoint = normalize_with_range((keypoint - b)/resolution, max(size)-1, 0)
    keypoint = remove_keypoint_artifact(keypoint, threshold)

    frame = np.reshape(keypoint, (22,3))

    for k in range(22):
        dis = np.sqrt((pos_x-frame[k,0])**2 + (pos_y-frame[k,1])**2 + (pos_z-frame[k,2])**2)
        g = gaussian(dis, 0.001, 1)
        heatmap[k,:,:,:] = softmax(g) /0.25 #1:0.25; 0.5:0.8
    return keypoint, heatmap

