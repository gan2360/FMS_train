


import torch
import cv2
from rawModel.vaTPose import VaTPose
from my_utils.visual_2d import show_2d_kpts
from my_utils.visual_3d import show_keypoints

testdata_path = 'D:/Users/Gan/FMS_system/fms_predict_sdk_v3-single_process/fms_predict_sdk_v3-single_process/fmsTestData'
hrnet_model_path = 'D:/Users/Gan/FMS_algorithm/hrnet_model.onnx'
yolo_model_path = 'D:/Users/Gan/FMS_algorithm/yolov3.weights'






