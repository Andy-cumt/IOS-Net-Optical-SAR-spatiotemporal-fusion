import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ssim import msssim
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from osgeo import gdal
from data import SCALE_FACTOR

NUM_BANDS = 6



class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class Downsampling(nn.Module):
    def __init__(self, SCALE):
        super(Downsampling, self).__init__()
        kernel_size = SCALE
        stride = SCALE
        self.down=nn.AvgPool2d(kernel_size, stride)
    def forward(self, inputs):
        return self.down(inputs)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Sequential(
        nn.ReplicationPad2d(1),  #重复填充ReplicationPad2d，重复填充即重复图像的边缘像素值，将新的边界像素值用边缘像素值扩展
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,bias=bias)
    )

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


def SCA(dw_channel):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),nn.Conv2d(in_channels=dw_channel,
        out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
        groups=1, bias=True),
        )



class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = SCA(dw_channel // 2)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # self.norm1 = nn.InstanceNorm2d(c // 2, affine=True)
        # self.norm2 = nn.InstanceNorm2d(c // 2, affine=True)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class UNAFNetblock(nn.Module):

    def __init__(self, img_channel=10,out_channel=6, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)


        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class fusionblock(nn.Module):

    def __init__(self, in_channel=192,out_channel=64,conv=default_conv):
        super().__init__()
        self.norm = LayerNorm2d(in_channel)
        self.cov1 = conv(in_channels=in_channel, out_channels=in_channel*2//3, kernel_size=3, bias=True)
        self.sca = SCA(out_channel)
        self.sg = SimpleGate()

    def forward(self, x):
        x1 = self.norm(x)
        x2 = self.cov1(x1)
        x3 = self.sg(x2)
        x4 = x3 * self.sca(x3)
        return x4

class fusionblock2(nn.Module):

    def __init__(self, in_channel=192,out_channel=64,conv=default_conv):
        super().__init__()
        self.norm = LayerNorm2d(in_channel)
        self.cov1 = conv(in_channels=in_channel, out_channels=in_channel, kernel_size=3, bias=True)
        self.cov2 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel*2//3, kernel_size=1, padding=0, bias=True)
        self.sca = SCA(out_channel)
        self.sg = SimpleGate()

    def forward(self, x):
        x1 = self.norm(x)
        x2 = self.cov2(self.cov1(x1))
        x3 = self.sg(x2)
        x4 = x3 * self.sca(x3)
        return x4


class TSinjectblock(nn.Module):
    def __init__(self, channel=64):
        super().__init__()
        self.sca = SCA(channel)
    def forward(self, x):
        x = x * self.sca(x) + x
        return x


class TSinjectblock2(nn.Module):
    def __init__(self, channel=64,conv=default_conv):
        super().__init__()
        self.sca = SCA(channel)
        self.cov1 = conv(in_channels=channel, out_channels=channel, kernel_size=3, bias=True)
        self.cov2 = nn.Conv2d(in_channels=channel//2, out_channels=channel, kernel_size=1, padding=0, bias=True)
        self.sg = SimpleGate()
    def forward(self, x):
        # out = x * self.sca(x) + x
        x = self.cov2(self.sg(self.cov1(x)))
        out = x * self.sca(x) + x
        # sim_att = torch.sigmoid(out) - 0.5
        # out = (out + x) * sim_att
        return out



class LMSfusion(nn.Module):
    def __init__(self, conv=default_conv):
        super().__init__()
        img_channel_L = 6
        img_channel_M = 6
        img_channel_S = 2
        out_channel = 6
        width1 = 28
        width2 = 28


        enc_blks = [1, 1, 1]
        middle_blk_num = 1
        dec_blks = [1, 1, 1]

        self.TSI1_1 = TSinjectblock2(width1 * 2)
        self.TSI1_2 = TSinjectblock2(width1 * 2)
        self.TSI1_3 = TSinjectblock2(width1 * 2)
        self.TSI1_4 = TSinjectblock2(width1 * 2)
        self.TSI1_5 = TSinjectblock2(width1 * 2)
        self.TSI1_6 = TSinjectblock2(width1 * 2)

        self.TSI2_1 = TSinjectblock2(width2 * 2)
        self.TSI2_2 = TSinjectblock2(width2 * 2)
        self.TSI2_3 = TSinjectblock2(width2 * 2)
        self.TSI2_4 = TSinjectblock2(width2 * 2)
        self.TSI2_5 = TSinjectblock2(width2 * 2)
        self.TSI2_6 = TSinjectblock2(width2 * 2)


        self.Downsampling = Downsampling(SCALE=8)


        self.NAFBlock1_1 = NAFBlock(c=2 * width1, DW_Expand=2, FFN_Expand=2)
        self.NAFBlock1_2 = NAFBlock(c=2 * width1, DW_Expand=2, FFN_Expand=2)
        self.NAFBlock1_3 = NAFBlock(c=2 * width1, DW_Expand=2, FFN_Expand=2)
        self.NAFBlock1_4 = NAFBlock(c=2 * width1, DW_Expand=2, FFN_Expand=2)
        self.NAFBlock1_5 = NAFBlock(c=2 * width1, DW_Expand=2, FFN_Expand=2)
        self.NAFBlock2_1 = NAFBlock(c=2 * width2, DW_Expand=2, FFN_Expand=2)
        self.NAFBlock2_2 = NAFBlock(c=2 * width2, DW_Expand=2, FFN_Expand=2)
        self.NAFBlock2_3 = NAFBlock(c=2 * width2, DW_Expand=2, FFN_Expand=2)
        self.NAFBlock2_4 = NAFBlock(c=2 * width2, DW_Expand=2, FFN_Expand=2)
        self.NAFBlock2_5 = NAFBlock(c=2 * width2, DW_Expand=2, FFN_Expand=2)

        self.fusionblock1_1 = fusionblock2(in_channel=6 * width1,out_channel=2*width1)
        self.fusionblock1_2 = fusionblock2(in_channel=6 * width1, out_channel=2 * width1)
        self.fusionblock1_3 = fusionblock2(in_channel=6 * width1, out_channel=2 * width1)
        self.fusionblock2_1 = fusionblock2(in_channel=6 * width2, out_channel=2 * width2)
        self.fusionblock2_2 = fusionblock2(in_channel=6 * width2, out_channel=2 * width2)
        self.fusionblock2_3 = fusionblock2(in_channel=6 * width2, out_channel=2 * width2)



        self.covS1_1 = conv(in_channels=img_channel_L, out_channels=width1 * 2, kernel_size=3, bias=True)
        self.covS1_2 = conv(in_channels=img_channel_M, out_channels=width1 * 2, kernel_size=3, bias=True)
        self.covS1_3 = conv(in_channels=img_channel_S, out_channels=width1 * 2, kernel_size=3, bias=True)
        self.covS2_1 = conv(in_channels=width1 * 2, out_channels=width2 * 2, kernel_size=3, bias=True)
        self.covS2_2 = conv(in_channels=width1, out_channels=width2 * 2, kernel_size=3, bias=True)
        self.covS2_3 = conv(in_channels=width1 * 2, out_channels=width2 * 2, kernel_size=3, bias=True)


        self.ending_1 = conv(in_channels=width1 * 2, out_channels=width1, kernel_size=3, bias=True)

        self.ending_2 = conv(in_channels=width2 * 2, out_channels=out_channel, kernel_size=3, bias=True)

    def forward(self, x):
        L = x[0][:, 0:6, :, :]
        M = x[1][:, 0:6, :, :]
        S = x[2][:, 0:2, :, :]


        #Stage 1
        Ld_2 = (self.covS1_1(L))
        M_2 = (self.covS1_2(M))
        Sd_2 = (self.covS1_3(S))

        M_2 = F.interpolate(input=M_2,scale_factor=8,mode='nearest')

        M1 = self.TSI1_1(Ld_2)
        M2 = self.TSI1_2(Sd_2)
        MF_1 = self.fusionblock1_1(torch.cat((M_2, M1, M2), 1))

        Ld_3 = self.NAFBlock1_1(Ld_2)
        Sd_3 = self.NAFBlock1_2(Sd_2)

        M1 = self.TSI1_3(Ld_3)
        M2 = self.TSI1_4(Sd_3)
        MF_2 = self.fusionblock1_2(torch.cat((MF_1, M1, M2), 1))

        Ld_4 = self.NAFBlock1_3(Ld_3)
        Sd_4 = self.NAFBlock1_4(Sd_3)

        M1 = self.TSI1_5(Ld_4)
        M2 = self.TSI1_6(Sd_4)
        MF_3 = self.fusionblock1_3(torch.cat((MF_2, M1, M2), 1))


        Ld_5 = self.NAFBlock2_1(Ld_4)
        Sd_5 = self.NAFBlock2_2(Sd_4)

        M1 = self.TSI2_1(Ld_5)
        M2 = self.TSI2_2(Sd_5)
        MF_4 = self.fusionblock2_1(torch.cat((MF_3, M1, M2), 1))


        output = self.ending_2((MF_4))

        return output
