import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from nets.backbone import Backbone, Block, Conv, SiLU, Transition, autopad
from utils.attentions import se_block, cbam_block, eca_block, EPSABlock, PSAModule

attention_bocks = [se_block, cbam_block, eca_block, PSAModule]


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=SiLU(), deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                        nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


# ---------------------------------------------------#
#   Scale Sequence
# ---------------------------------------------------#
class reshapeP5(nn.Module):
    def __init__(self):
        super(reshapeP5, self).__init__()
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.upsample5 = nn.Upsample(scale_factor=4, mode="nearest")

    def forward(self, x):
        P5_reshape = self.conv5(x)
        P5_reshape = self.upsample5(P5_reshape)
        return P5_reshape


class reshapeP4(nn.Module):
    def __init__(self):
        super(reshapeP4, self).__init__()
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.upsample4 = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        P4_reshape = self.conv4(x)
        P4_reshape = self.upsample4(P4_reshape)
        return P4_reshape


class reshapeP(nn.Module):
    def __init__(self):
        super(reshapeP, self).__init__()
        # self.resizeP = torch.resize_(16, 128, 80, 80)
        # print(P.shape)
        self.unsqeezeP = torch.unsqueeze(2)

    def forward(self, x):
        P_reshape = self.resizeP(x)
        P_reshape = self.unsqeezeP(P_reshape)
        return P_reshape


class scaleSequence(nn.Module):
    def __init__(self):
        super(scaleSequence, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3d = nn.BatchNorm3d(128, eps=0.001)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pool3d = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)
        self.upsample = nn.Upsample(size=(1, 80, 80))

    def forward(self, x):
        out = self.conv3d(x)
        out = self.bn3d(out)
        out = self.act(out)
        out = self.pool3d(out)
        out = self.upsample(out)
        return out


class ssFPN(nn.Module):
    def __init__(self, c_in, c_out):
        super(ssFPN, self).__init__()
        self.conv3d = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1)
        self.bn3d = nn.BatchNorm2d(128, eps=0.001)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pool3d = nn.AvgPool2d(kernel_size=(1, 1), stride=(2, 2), padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        out = self.conv3d(x)
        out = self.bn3d(out)
        out = self.act(out)
        out = self.pool3d(out)
        out = self.upsample(out)
        return out


# ---------------------------------------------------#
#   SRModule: superYOLO中的超分辨模块
# ---------------------------------------------------#
class CRModule(nn.Module):
    def __init__(self, c_in, c_out):
        super(CRModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.conv(x)
        out = self.dropout(out)
        return out


class SRModule(nn.Module):
    def __init__(self, c_in_low):
        super(SRModule, self).__init__()
        transition_channel = 32
        # -----------------------------------------------#
        #   Encoder
        # -----------------------------------------------#
        self.upsample = nn.Upsample(scale_factor=4, mode="nearest")
        self.cr1 = CRModule(c_in_low, transition_channel * 16)
        self.cr2 = CRModule(transition_channel * 48, transition_channel * 8)  # transition_channel * 48 由concat得到
        self.cr3 = CRModule(transition_channel * 8, transition_channel * 4)
        # -----------------------------------------------#
        #   Decoder
        # -----------------------------------------------#
        self.deconv1 = nn.ConvTranspose2d(transition_channel * 4, transition_channel * 2, kernel_size=2, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(transition_channel * 2, transition_channel, kernel_size=2, stride=2, padding=0)
        self.deconv3 = nn.ConvTranspose2d(transition_channel, transition_channel // 2, kernel_size=2, stride=2, padding=0)
        self.deconv4 = nn.ConvTranspose2d(transition_channel // 2, 3, kernel_size=2, stride=2, padding=0)

    def forward(self, x_low, x_high):
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   low_level为80, 80, 512
        #   high_level为20, 20, 1024
        #   SR模块输出为1280, 1280, 3
        # -----------------------------------------------#
        out_high = self.upsample(x_high)
        out_low = self.cr1(x_low)
        out = torch.cat((out_low, out_high), dim=1)
        out = self.cr2(out)
        out = self.cr3(out)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return out


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, pretrained=False, phi_attention=1):
        super(YoloBody, self).__init__()
        # -----------------------------------------------#
        #   定义了不同yolov7版本的参数
        # -----------------------------------------------#
        transition_channels = {'l': 32, 'x': 40}[phi]
        block_channels = 32
        panet_channels = {'l': 32, 'x': 64}[phi]
        e = {'l': 2, 'x': 1}[phi]
        n = {'l': 4, 'x': 6}[phi]
        ids = {'l': [-1, -2, -3, -4, -5, -6], 'x': [-1, -3, -5, -7, -8]}[phi]
        conv = {'l': RepConv, 'x': Conv}[phi]
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        # -----------------------------------------------#

        # ---------------------------------------------------#
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80, 80, 512
        #   40, 40, 1024
        #   20, 20, 1024
        # ---------------------------------------------------#
        self.backbone = Backbone(transition_channels, block_channels, n, phi, pretrained=pretrained)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 16)
        self.conv_for_P5 = Conv(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2 = Conv(transition_channels * 32, transition_channels * 8)
        self.conv3_for_upsample1 = Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e,
                                         n=n, ids=ids)

        self.conv_for_P4 = Conv(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1 = Conv(transition_channels * 16, transition_channels * 4)
        self.conv3_for_upsample2 = Block(transition_channels * 8, panet_channels * 2, transition_channels * 4, e=e, n=n,
                                         ids=ids)

        self.down_sample1 = Transition(transition_channels * 4, transition_channels * 4)
        self.conv3_for_downsample1 = Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e,
                                           n=n, ids=ids)

        self.down_sample2 = Transition(transition_channels * 8, transition_channels * 8)
        self.conv3_for_downsample2 = Block(transition_channels * 32, panet_channels * 8, transition_channels * 16, e=e,
                                           n=n, ids=ids)

        self.rep_conv_1 = conv(transition_channels * 4, transition_channels * 8, 3, 1)
        self.rep_conv_2 = conv(transition_channels * 8, transition_channels * 16, 3, 1)
        self.rep_conv_3 = conv(transition_channels * 16, transition_channels * 32, 3, 1)

        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)
        self.phi_attention = phi_attention
        # self.reshape5 = reshapeP5()
        # self.reshape4 = reshapeP4()
        # self.scale_sequence = ssFPN(c_in=384, c_out=128)
        # self.scale_sequence = scaleSequence()
        # self.sr = SRModule(c_in_low=512)

        if phi_attention >= 1 and phi_attention <= 3:
            # self.feat1_attention = attention_bocks[phi_attention - 1](512)
            self.P3_attention = attention_bocks[phi_attention - 1](128)
            self.P4_attention = attention_bocks[phi_attention - 1](256)
            self.P5_attention = attention_bocks[phi_attention - 1](512)
            # self.feat2_attention = attention_bocks[phi_attention - 1](1024)
            # self.feat3_attention = attention_bocks[phi_attention - 1](1024)

        if phi_attention == 4:
            self.P3_attention = attention_bocks[phi_attention - 1](256, 256)
            self.P4_attention = attention_bocks[phi_attention - 1](256, 256)
            self.P5_attention = attention_bocks[phi_attention - 1](512, 512)

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self

    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)
        # print(feat1.shape)
        # if self.phi_attention >= 1 and self.phi_attention <= 3:
        # feat1 = self.feat1_attention(feat1)
        # feat2 = self.feat2_attention(feat2)

        # sr_out = self.sr.forward(feat1, feat3)

        P5 = self.sppcspc(feat3)
        P5_conv = self.conv_for_P5(P5)
        P5_upsample = self.upsample(P5_conv)
        # print(P5.shape)

        P4 = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        P4 = self.conv3_for_upsample1(P4)
        P4_conv = self.conv_for_P4(P4)
        P4_upsample = self.upsample(P4_conv)
        # print(P4.shape)

        P3 = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        P3 = self.conv3_for_upsample2(P3)
        # print(P3.shape)
        # -----------------------------------------------#
        #   尺度不变特征序列
        # -----------------------------------------------#
        """P5_reshape = self.reshape5(P5)
        P4_reshape = self.reshape4(P4)
        P5_reshape = P5_reshape.unsqueeze(2)
        P4_reshape = P4_reshape.unsqueeze(2)
        P3_reshape = P3.unsqueeze(2)
        sequence = torch.cat((P5_reshape, P4_reshape, P3_reshape), dim=2)
        # print(sequence.shape)
        sequence = self.scale_sequence(sequence)
        sequence = sequence.squeeze(2)
        sequence = torch.cat((sequence, P3), dim=1)"""

        if 1 <= self.phi_attention <= 4:
            # sequence = self.P3_attention(sequence)
            P3 = self.P3_attention(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        if 1 <= self.phi_attention <= 4:
            P4 = self.P4_attention(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)

        if 1 <= self.phi_attention <= 4:
            P5 = self.P5_attention(P5)

        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)
        # ---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size, 75, 80, 80)
        # ---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        # ---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size, 75, 40, 40)
        # ---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        # ---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size, 75, 20, 20)
        # ---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)

        return [out0, out1, out2]  # , sr_out
