import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from CrossMaxvit import MaxViTBlock as crossmax
from Maxvit import MaxViTBlock as Max


class SAM(nn.Module):
    def __init__(self, nf=48, use_residual=True, learnable=True):
        super(SAM, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)
        self.pixel_shuffle = nn.PixelShuffle(2)
        if self.learnable:
            self.conv_noshared = nn.Sequential(nn.Conv2d(nf , nf, 3, 1, 1, bias=True),
                                             nn.GELU())
            self.conv_shared = nn.Sequential(nn.Conv2d(nf*2 , nf, 3, 1, 1, bias=True),
                                             nn.GELU())
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, x1, x2):
        b,c,h,w = x1.shape
        x1_mean = torch.mean(x1.reshape(b,c,h*w), dim=-1, keepdim=True).reshape(b,c,1,1)
        x1_std = torch.std(x1.reshape(b,c,h*w), dim=-1, keepdim=True).reshape(b,c,1,1)
        x2_normed = self.norm_layer(x2)
        style = torch.cat([x1, x2], dim=1)
        style = self.conv_shared(style)
        gamma = self.conv_gamma(style)
        beta = self.conv_beta(style)
        if self.learnable:
            if self.use_residual:
                gamma = gamma + x1_std
                beta = beta + x1_mean
        out = x2_normed * gamma + beta
        return out

class Select(nn.Module):  # conv
    def __init__(self, in_nc=48, out_nc=32, kernel_size=1, stride=1, padding=0, visual=False):
        super(Select, self).__init__()
        self.conv = nn.Conv2d(in_nc, 1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.Sigmoid()

    def forward(self, x, complement):
        x = self.conv(x)
        x = self.act(x)
        out = x * complement
        return out



def conv(in_channels, out_channels, kernel_size, bias=False, stride=1, groups=1):
    return nn.Sequential(nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride, groups=groups)
    )


class T_conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(T_conv, self).__init__()
        self.conv11 = nn.Sequential(
            conv(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=3, bias=bias),
            nn.GroupNorm(4, out_channels // 4))
        self.conv33 = nn.Sequential(
            conv(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=3))
        # nn.BatchNorm2d(in_channels // 4))
        self.conv55 = nn.Sequential(
            conv(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=3))
        # nn.BatchNorm2d(in_channels // 4))
        self.conv77 = nn.Sequential(
            conv(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=3))
        # nn.BatchNorm2d(in_channels // 4))

        # self.act = nn.ReLU()

        self.conv11_ = nn.Sequential(
            conv(in_channels=out_channels, out_channels=out_channels // 4, kernel_size=31, groups=out_channels // 4,
                 bias=bias),
            nn.GroupNorm(4, out_channels // 4))
        self.conv33_ = nn.Sequential(
            conv(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=3))
        # nn.BatchNorm2d(in_channels // 4))
        self.conv55_ = nn.Sequential(
            conv(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=3))
        # nn.BatchNorm2d(in_channels // 4))
        self.conv77_ = nn.Sequential(
            conv(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=3))
        # nn.BatchNorm2d(in_channels // 4))

    def forward(self, x):
        conv11 = self.conv11(x)
        conv33 = self.conv33(conv11)
        conv55 = self.conv55(conv33)
        conv77 = self.conv77(conv55)
        out = torch.cat((conv11, conv33, conv55, conv77), dim=1)
        conv11_ = self.conv11_(out)
        conv33_ = self.conv33_(conv11_ + conv77)
        conv55_ = self.conv55_(conv33_ + conv55)
        conv77_ = self.conv77_(conv55_ + conv33)
        out_ = torch.cat((conv11_, conv33_, conv55_, conv77_), dim=1) + out

        # out = self.act(torch.cat((conv11, conv33, conv55, conv77), dim=1))
        return out_


class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)


class f_ConvLayer(torch.nn.Module):  # 融合卷积层
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(f_ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        # out = self.batch_norm(out)
        out = F.leaky_relu(out, negative_slope=0.1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2 * channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)

    def forward(self, x):
        x = torch.cat((x, self.conv1(x)), dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x


class DenseBlock_t(nn.Module):
    def __init__(self, channels):
        super(DenseBlock_t, self).__init__()
        self.conv1 = T_conv(channels, channels)
        self.conv2 = T_conv(2 * channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)

    def forward(self, x):
        out_1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        out_2 = torch.cat((x, out_1), dim=1)
        out_2 = F.leaky_relu(self.conv2(out_2), negative_slope=0.2)
        out_2 = torch.cat((x, out_2), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return out_2


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


class RGBD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RGBD, self).__init__()
        self.dense = DenseBlock(in_channels)
        self.convdown = Conv1(3 * in_channels, out_channels)
        self.sobelconv = Sobelxy(in_channels)
        self.convup = Conv1(in_channels, out_channels)

    def forward(self, x):
        x1 = self.dense(x)
        x1 = self.convdown(x1)
        x2 = self.sobelconv(x)
        x2 = self.convup(x2)
        return F.leaky_relu(x1 + x2, negative_slope=0.1)


class MaxFA_Module(nn.Module):  #Fatt
    """ feature attention module"""
    def __init__(self, in_dim):
        super(MaxFA_Module, self).__init__()

        self.t1_adap_1 = SAM()
        self.select1 = Select()
        self.sam = crossmax(in_dim,in_dim)



    def forward(self, x2, x1):

        t1_b_fea_tmp = self.t1_adap_1(x2, x1)
        t1_cat_1 = self.sam(x2, t1_b_fea_tmp)  # 32
        x_cat_1 = x2 + t1_cat_1
        t1_cat_1 = self.select1(x1, x_cat_1)
        x_out = x2 + t1_cat_1


        return x_out



class MACTFusion(nn.Module):
    def __init__(self, output):
        super(MACTFusion, self).__init__()
        vis_ch = [16, 32, 48]
        inf_ch = [16, 32, 48]
        output = 1
        self.vis_conv = T_conv(1, vis_ch[0])
        self.vis_rgbd1 = RGBD(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = RGBD(vis_ch[1], vis_ch[2])
        self.vis_max = Max(vis_ch[2], vis_ch[2])
        self.vis_cross = MaxFA_Module(vis_ch[2])

        self.inf_conv = T_conv(1, inf_ch[0])
        self.inf_rgbd1 = RGBD(inf_ch[0], inf_ch[1])
        self.inf_rgbd2 = RGBD(inf_ch[1], inf_ch[2])
        self.inf_max = Max(inf_ch[2], inf_ch[2])
        self.inf_cross = MaxFA_Module(vis_ch[2])


        block = []
        block += [f_ConvLayer(2 * vis_ch[2], vis_ch[2], 1, 1),
                  f_ConvLayer(vis_ch[2], vis_ch[1], 3, 1),
                  f_ConvLayer(vis_ch[1], output, 3, 1)]
        self.decode = nn.Sequential(*block)

    def forward(self, image_vis, image_ir):
        x_vis_origin = image_vis[:, :1]
        x_inf_origin = image_ir
        # encode
        x_vis_p = self.vis_conv(x_vis_origin)
        x_vis_p1 = self.vis_rgbd1(x_vis_p)
        x_vis_p2 = self.vis_rgbd2(x_vis_p1)
        x_vis_p2 = self.vis_max(x_vis_p2)

        x_inf_p = self.inf_conv(x_inf_origin)
        x_inf_p1 = self.inf_rgbd1(x_inf_p)
        x_inf_p2 = self.inf_rgbd2(x_inf_p1)
        x_inf_p2 = self.inf_max(x_inf_p2)

        x_vis_cross = self.vis_cross(x_vis_p2,x_inf_p2)
        x_inf_cross = self.inf_cross(x_inf_p2,x_vis_p2)
        # decode
        x = self.decode(torch.cat((x_vis_cross, x_inf_cross), dim=1))

        return x


# if __name__ == "__main__":
#     net = MACTFusion(1)
#     x1 = torch.rand((4, 1, 256, 256))
#     x2 = torch.rand((4, 1, 256, 256))
#     # net()
#     import thop
#
#     print(net)
#
#     print(thop.clever_format(thop.profile(net, (x1, x2))))
#     print()

#