import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ssim import msssim
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from osgeo import gdal

NUM_BANDS = 6


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Sequential( #Sequential快速搭建函数
        nn.ReplicationPad2d(1),  #重复填充ReplicationPad2d，重复填充即重复图像的边缘像素值，将新的边界像素值用边缘像素值扩展
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,bias=bias)
    )

def dila_conv(in_channels,out_channels,k,p,d):
    return nn.Conv2d(in_channels,out_channels,k,padding=p,dilation=d)


class Downsampling(nn.Module):
    def __init__(self, SCALE):
        super(Downsampling, self).__init__()
        kernel_size = SCALE
        stride = SCALE
        self.down=nn.AvgPool2d(kernel_size, stride)
    def forward(self, inputs):
        return self.down(inputs)


def interpolate(inputs, size=None, scale_factor=None):
    return F.interpolate(inputs, size=size, scale_factor=scale_factor,
                         mode='bilinear', align_corners=True)

def interpolate2(inputs, size=None, scale_factor=None):
    return F.interpolate(inputs, size=size, scale_factor=scale_factor,
                         mode='bicubic', align_corners=True)


def fillmask(MODIS):
    MODIS[torch.isnan(MODIS)] = 0
    for i in range(0,MODIS.shape[0]):
        for j in range(0, MODIS.shape[1]):
            A = MODIS[i,j,:,:]
            meanMODIS = torch.mean(A[A!=0])
            MODIS[i,j,:,:] = torch.where(MODIS[i,j,:,:] == 0, meanMODIS,MODIS[i,j,:,:])
    return MODIS

class CompoundLoss(nn.Module):
    def __init__(self, alpha=0.3, normalize=True): #0.1
        super(CompoundLoss, self).__init__()
        self.alpha = alpha
        self.normalize = normalize
        self.eps = 1e-10

    def forward(self, predictions, target,w):
        LP_AAD = 0.0
        meantarget = (torch.mean(target) + 0.001)
        for j in range(0, predictions.shape[1]):
            P = predictions[:, j, :, :]
            T = target[:, j, :, :]
            diff = torch.add(P, -T) * meantarget / (torch.mean(target[:, j, :, :]) + 0.001)

            error = torch.mean(torch.sqrt(diff * diff + self.eps))
            LP_AAD = LP_AAD + error
        LP1 = w * LP_AAD / (j + 1)

        LP2 = w * self.alpha * (1.0 - msssim(predictions, target, normalize=self.normalize))

        Loss = LP1 + LP2
        return Loss, LP1, LP2




