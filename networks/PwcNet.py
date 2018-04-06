import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from correlation_package.modules.correlation import Correlation
from resample2d_package.modules.resample2d import Resample2d
from channelnorm_package.modules.channelnorm import ChannelNorm

from submodules import *
'Parameter count , 39,175,298 '

class PwcNet(nn.Module):

    def __init__(self, div_flow=20):
        super(PwcNet, self).__init__()

        self.div_flow = div_flow

        self.py_conv1   = conv(False,  3, 16, stride=2, bias=True)
        self.py_conv1_b = conv(False, 16, 16, bias=True)
        self.py_conv2   = conv(False, 16, 32, stride=2, bias=True)
        self.py_conv2_b = conv(False, 32, 32, bias=True)
        self.py_conv3   = conv(False, 32, 64, stride=2, bias=True)
        self.py_conv3_b = conv(False, 64, 64, bias=True)
        self.py_conv4   = conv(False, 64, 96, stride=2, bias=True)
        self.py_conv4_b = conv(False, 96, 96, bias=True)
        self.py_conv5   = conv(False, 96, 128,stride=2, bias=True)
        self.py_conv5_b = conv(False, 128,128, bias=True)
        self.py_conv6   = conv(False, 128,196,stride=2, bias=True)
        self.py_conv6_b = conv(False, 196,196, bias=True)

        # fixed size correlation layer
        # the correlation layer output is always (4+1)*(4+1) = 81
        self.corr = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)
        self.corr_activation = nn.LeakyReLU(0.1,inplace=True)

        # every dense block will have output that is
        # inplanes + 128 + 128 + 96 + 64 + outplanes = 448
        self.dense6 = DenseBlock(False, 81,  32)
        # the input channel is 81+128+2+2=213
        self.dense5 = DenseBlock(False, 213, 32)
        # the input channle is 81+96+2+2 =181
        self.dense4 = DenseBlock(False, 181, 32)
        # the input channel is 81+64+2+2 =149
        self.dense3 = DenseBlock(False, 149, 32)
        # the input channel is 81+32+2+2 =117
        self.dense2 = DenseBlock(False, 117, 32)

        self.upsample_conv6to5 = nn.ConvTranspose2d(81+448,  2, 4, 2, 1, bias=True)
        self.upsample_conv5to4 = nn.ConvTranspose2d(213+448, 2, 4, 2, 1, bias=True)
        self.upsample_conv4to3 = nn.ConvTranspose2d(181+448, 2, 4, 2, 1, bias=True)
        self.upsample_conv3to2 = nn.ConvTranspose2d(149+448, 2, 4, 2, 1, bias=True)

        self.predict_flow6 = predict_flow(448+81)
        self.predict_flow5 = predict_flow(448+81+128+2+2)
        self.predict_flow4 = predict_flow(448+81+96+2+2)
        self.predict_flow3 = predict_flow(448+81+64+2+2)
        self.predict_flow2 = predict_flow(448+81+32+2+2)

        # all the flow have different upsampling weights
        self.upsample_flow6to5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsample_flow5to4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsample_flow4to3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsample_flow3to2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)

        # the warping layer
        self.resample = Resample2d()
        # the context network
        self.context_net = nn.Sequential(
            conv(False, 117+448, 128, bias=True),
            conv(False, 128,128, dilation=2, bias=True),
            conv(False, 128,128, dilation=4, bias=True),
            conv(False, 128,96,  dilation=8, bias=True),
            conv(False, 96, 64,  dilation=16, bias=True),
            conv(False, 64, 32, bias=True),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        x1 = x[:,0:3,:,:]
        x2 = x[:,3::,:,:]

        # encoder
        x1_conv1a = self.py_conv1(x1)
        x1_conv1b = self.py_conv1_b(x1_conv1a)
        x1_conv2a = self.py_conv2(x1_conv1b)
        x1_conv2b = self.py_conv2_b(x1_conv2a)
        x1_conv3a = self.py_conv3(x1_conv2b)
        x1_conv3b = self.py_conv3_b(x1_conv3a)
        x1_conv4a = self.py_conv4(x1_conv3b)
        x1_conv4b = self.py_conv4_b(x1_conv4a)
        x1_conv5a = self.py_conv5(x1_conv4b)
        x1_conv5b = self.py_conv5_b(x1_conv5a)
        x1_conv6a = self.py_conv6(x1_conv5b)
        x1_conv6b = self.py_conv6_b(x1_conv6a)

        x2_conv1a = self.py_conv1(x2)
        x2_conv1b = self.py_conv1_b(x2_conv1a)
        x2_conv2a = self.py_conv2(x2_conv1b)
        x2_conv2b = self.py_conv2_b(x2_conv2a)
        x2_conv3a = self.py_conv3(x2_conv2b)
        x2_conv3b = self.py_conv3_b(x2_conv3a)
        x2_conv4a = self.py_conv4(x2_conv3b)
        x2_conv4b = self.py_conv4_b(x2_conv4a)
        x2_conv5a = self.py_conv5(x2_conv4b)
        x2_conv5b = self.py_conv5_b(x2_conv5a)
        x2_conv6a = self.py_conv6(x2_conv5b)
        x2_conv6b = self.py_conv6_b(x2_conv6a)
        # Correlation 6 layer
        out_corr6 = self.corr_activation(self.corr(x1_conv6b, x2_conv6b))

        # DenseNet block 6
        out_conv6   = self.dense6(out_corr6)
        out_conv6_up = self.upsample_conv6to5(out_conv6)
        # predict flow at layer 6
        flow6 = self.predict_flow6(out_conv6)
        # pwc layer 5
        flow6_up = self.upsample_flow6to5(flow6)
        scale_5 = self.div_flow / float(1<<5)
        flow6s_up = scale_5 * flow6_up
        out_warp5 = self.resample(x2_conv5b, flow6s_up)
        out_corr5 = self.corr_activation(self.corr(x1_conv5b, out_warp5))

        concat5 = torch.cat((out_corr5, x1_conv5b, flow6_up, out_conv6_up), dim=1)

        # DenseNet Block 5
        out_conv5 = self.dense5(concat5)
        out_conv5_up = self.upsample_conv5to4(out_conv5)
        # predict flow at layer 5
        flow5 = self.predict_flow5(out_conv5)
        # pwc layer 4
        flow5_up = self.upsample_flow5to4(flow5)
        scale_4 = self.div_flow / float(1<<4)
        flow5_ups = scale_4 * flow5_up
        out_warp4 = self.resample(x2_conv4b, flow5_ups)
        out_corr4 = self.corr_activation(self.corr(x1_conv4b, out_warp4))

        concat4 = torch.cat((out_corr4, x1_conv4b, flow5_up, out_conv5_up), dim=1)

        # Dense Block 4
        out_conv4 = self.dense4(concat4)
        out_conv4_up = self.upsample_conv4to3(out_conv4)
        # predict flow at layer 4
        flow4 = self.predict_flow4(out_conv4)
        # pwc layer 3
        flow4_up = self.upsample_flow4to3(flow4)
        scale_3 = self.div_flow / float(1<<3)
        flow4_ups = scale_3 * flow4_up
        out_warp3 = self.resample(x2_conv3b, flow4_ups)
        out_corr3 = self.corr_activation(self.corr(x1_conv3b, out_warp3))

        concat3 = torch.cat((out_corr3, x1_conv3b, flow4_up, out_conv4_up), dim=1)

        # Dense Block 3
        out_conv3 = self.dense3(concat3)
        out_conv3_up = self.upsample_conv3to2(out_conv3)
        # predict flow at layer 3
        flow3 = self.predict_flow3(out_conv3)
        # pwc layer 2
        flow3_up = self.upsample_flow3to2(flow3)
        scale_2 = self.div_flow / float(1<<2)
        flow3_ups= scale_2 * flow3_up
        out_warp2 = self.resample(x2_conv2b, flow3_ups)
        out_corr2 = self.corr_activation(self.corr(x1_conv2b, out_warp2))

        concat2 = torch.cat((out_corr2, x1_conv2b, flow3_up, out_conv3_up), dim=1)

        out_conv2 = self.dense2(concat2)
        flow_dc = self.context_net(out_conv2)
        flow2 = self.predict_flow2(out_conv2)

        flow_mixed2 = flow_dc + flow2
        flow = self.upsample(flow_mixed2)

        if self.training:
            return flow, flow3, flow4, flow5, flow6
        else:
            return flow

class DenseBlock(nn.Module):

    def __init__(self, batchNorm, inplanes, outplanes=32):
        super(DenseBlock, self).__init__()

        self.conv1 = conv(batchNorm, inplanes, 128, bias=True)
        self.conv2 = conv(batchNorm, inplanes+128, 128, bias=True)
        self.conv3 = conv(batchNorm, inplanes+128+128, 96, bias=True)
        self.conv4 = conv(batchNorm, inplanes+128+128+96, 64, bias=True)
        self.conv5 = conv(batchNorm, inplanes+128+128+96+64, outplanes, bias=True)

    def forward(self, x):
        out1 = self.conv1(x)
        cat1 = torch.cat((x, out1), dim=1)
        out2 = self.conv2(cat1)
        # The concatenation in original PWC-prototxt is not always the same
        # order. Pay attention to the concatenation order here
        cat2 = torch.cat((out2, cat1), dim=1)
        out3 = self.conv3(cat2)
        cat3 = torch.cat((cat2, out3), dim=1)
        out4 = self.conv4(cat3)
        cat4 = torch.cat((cat3, out4), dim=1)
        out5 = self.conv5(cat4)
        cat5 = torch.cat((cat4, out5), dim=1)
        return cat5

if __name__ == '__main__':

    from torch.autograd import Variable

    net = PwcNet()
    net.cuda()
    net.train()

    x = torch.ones((1, 6, 128, 128)).cuda()
    x = Variable(x, requires_grad=False)

    flow = net.forward(x)
