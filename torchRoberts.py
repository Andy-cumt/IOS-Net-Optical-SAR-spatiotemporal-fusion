import torch
import torch.nn as nn
import numpy as np
from writeimage import *

class robertsLoss(nn.Module):
    def __init__(self, model):
        super(robertsLoss, self).__init__()
        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    def forward(self, predictions, target, alpha):
        edge = torch.zeros(predictions.shape[0], predictions.shape[1], predictions.shape[2]*predictions.shape[3]).cuda()
        Ax = int(np.floor(0.1 * predictions.shape[2] * predictions.shape[3]))
        edge_T_final = torch.zeros(predictions.shape[2] * predictions.shape[3]).cuda()
        edge_P_final = torch.zeros(predictions.shape[2] * predictions.shape[3]).cuda()
        for i in range(0, edge.shape[0]):
            for j in range(0, edge.shape[1]):
                T = (target[i, j, :, :]).reshape(1, 1, predictions.shape[2],predictions.shape[3])
                P = (predictions[i, j, :, :]).reshape(1, 1, predictions.shape[2],predictions.shape[3])

                edge_T = self.model(T)
                edge_T1 = edge_T[0, :, :]
                edge_T2 = edge_T1.reshape(-1)
                Ta, Tb = torch.topk(edge_T2, Ax,largest=True)
                edge_T_final[Tb[:]] = edge_T2[Tb[:]]

                edge_P = self.model(P)
                edge_P1 = edge_P[0, :, :]
                edge_P2 = edge_P1.reshape(-1)
                Pa, Pb = torch.topk(edge_P2, Ax,largest=True)
                edge_P_final[Pb[:]] = edge_P2[Pb[:]]
                edge[i, j, :] = torch.abs((edge_T_final - edge_P_final) / (edge_T_final + edge_P_final + 0.00001))

        loss = alpha * torch.mean(edge)
        # aa = edge[2,:,:,:]
        # preliminary=aa.detach().cpu().numpy()
        # preliminary = np.squeeze(preliminary)
        # path = "F:\deeplearning\RDCSFM18\prediction//test//Robortsedge.tiff"
        # in_ds = "F:/deeplearning/data4/val//1//01_LC08_031033_20211105cutt.tif"
        # writeimage(preliminary,path,in_ds)
        return loss


class roberts(nn.Module):
    def __init__(self):
        '''
        kernel: shape(out_channels, in_channels, h, w)
        '''
        kernel = np.array([
            [[
                [1, 0],
                [0, -1]
            ]],
            [[
                [0, -1],
                [1, 0]
            ]]
        ])

        super(roberts, self).__init__()
        out_channels, in_channels, h, w = kernel.shape
        self.filter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(h, w),
            padding='same',
            bias=False
        )
        self.filter.weight.data.copy_(torch.from_numpy(kernel.astype('float32')))

    @staticmethod
    def postprocess(outputs, mode=0, weight=None):
        '''
        Input: NCHW
        Output: NHW(mode==1-3) or NCHW(mode==4)

        Params:
            mode: switch output mode(0-4)
            weight: weight when mode==3
        '''
        device = outputs.device
        if mode == 0:
            results = torch.sum(torch.abs(outputs), dim=1)
        elif mode == 1:
            results = torch.sqrt(torch.sum(torch.pow(outputs, 2), dim=1))
        elif mode == 2:
            results = torch.max(torch.abs(outputs), dim=1)
        elif mode == 3:
            if weight is None:
                C = outputs.shape[1]
                weight = torch.from_numpy([1/C] * C, dtype=torch.float32).to(device)
            else:
                weight = torch.from_numpy(weight, dtype=torch.float32).to(device)
            results = torch.einsum('nchw, c -> nhw', torch.abs(outputs), weight)
        elif mode == 4:
            results = torch.abs(outputs)

        return torch.clip(results, 0, 1).float()

    #@torch.no_grad()
    def forward(self, images, mode=0, weight=None):
        '''
        Input: NCHW
        Output: NHW(mode==1-3) or NCHW(mode==4)

        Params:
            images: input tensor of images
            mode: switch output mode(0-4)
            weight: weight when mode==3
        '''
        outputs = self.filter(images)
        return self.postprocess(outputs, mode, weight)


