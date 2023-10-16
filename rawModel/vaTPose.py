"""
@Project ：fms_predict_sdk_v2
@File    ：vaTPose.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/5/14 15:03
@Des     ：
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
import cv2 

from einops import rearrange
from rawModel.model_s3.vanilla_transformer_encoder import Transformer
from rawModel.model_s3.strided_transformer_encoder import Transformer as Transformer_reduce
from rawModel.transformer import huge_patch14_120_in21k
    
class VaTPose(nn.Module):
    def __init__(self, windowSize):
        super(VaTPose, self).__init__()
        self.windowSize = windowSize
        self.tactile_encoder = huge_patch14_120_in21k(num_joints=22, has_logits=False, windowSize=int(2*windowSize))
  
        self.encoder = nn.Sequential(
            nn.Conv1d(3*22, 256, kernel_size=1),
            nn.BatchNorm1d(256, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )

        self.Transformer = Transformer(3, 256, 512, length=int(2*windowSize))
        self.Transformer_reduce = Transformer_reduce(3, 256, 512, \
            length=int(2*windowSize), stride_num=[3, 9, 13])
        
        self.fcn = nn.Sequential(
            nn.BatchNorm1d(256, momentum=0.1),
            nn.Conv1d(256, 3*22, kernel_size=1)
        )

        self.fcn_1 = nn.Sequential(
            nn.BatchNorm1d(256, momentum=0.1),
            nn.Conv1d(256, 3*22, kernel_size=1)
        )


    def forward(self, input):
        keypoints2d = input[0] # (batch, frames, 22, 2)
        tactile_data = input[1]
        
        tactile_feature = self.tactile_encoder(tactile_data) # (batch, 22, 1)
        shape = tactile_feature.shape
        tactile_feature = tactile_feature.reshape(shape[0], 1, shape[1], 1)
        tactile_feature = tactile_feature.repeat(1, int(2*self.windowSize), 1, 1) # (batch, frames, 22, 1)
        
        x = torch.cat((keypoints2d, tactile_feature), axis=3)
        
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()

        x = self.encoder(x) 
        x = x.permute(0, 2, 1).contiguous()
        
        x = self.Transformer(x) 
        
        x_VTE = x
        x_VTE = x_VTE.permute(0, 2, 1).contiguous()
        x_VTE = self.fcn_1(x_VTE) 
        x_VTE = rearrange(x_VTE, 'b (j c) f -> b f j c', j=J).contiguous() 
        
        x = self.Transformer_reduce(x) 
        
        x = x.permute(0, 2, 1).contiguous() 
        x = self.fcn(x) 
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous() 
        
        return x