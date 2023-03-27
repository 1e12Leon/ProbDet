import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov7.nets.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from yolov7.nets.SR_Decoder import Decoder
from yolov7.nets.SR_Encoder import EDSR


class EDSRConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EDSRConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            )

        self.residual_upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            )

    def forward(self, input):
        return self.conv(input)+self.residual_upsampler(input)


class DeepLab(nn.Module):
    def __init__(self, ch, c1=128, c2=512, factor=2):
        super(DeepLab, self).__init__()
        self.sr_decoder = Decoder(c1, c2)
        self.edsr = EDSR(num_channels=ch, input_channel=64, factor=8)
        self.factor = factor

    def forward(self, low_level_feat,x):
        x_sr = self.sr_decoder(x, low_level_feat, self.factor)
        x_sr_up = self.edsr(x_sr)
        return x_sr_up




