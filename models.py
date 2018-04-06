import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from networks.resample2d_package.modules.resample2d import Resample2d
from networks.channelnorm_package.modules.channelnorm import ChannelNorm

from networks import FlowNetC
from networks import FlowNetS
from networks import FlowNetSD
from networks import FlowNetFusion
from networks import PwcNet

from networks.submodules import *
'Parameter count = 162,518,834'

class FlowNet2(nn.Module):

    def __init__(self, args, batchNorm=False, div_flow = 20.):
        super(FlowNet2,self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        if args.fp16:
            self.resample1 = nn.Sequential(
                            tofp32(),
                            Resample2d(),
                            tofp16())
        else:
            self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample2 = nn.Sequential(
                            tofp32(),
                            Resample2d(),
                            tofp16())
        else:
            self.resample2 = Resample2d()


        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)

        # Block (FlowNetSD)
        self.flownets_d = FlowNetSD.FlowNetSD(args, batchNorm=self.batchNorm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')

        if args.fp16:
            self.resample3 = nn.Sequential(
                            tofp32(),
                            Resample2d(),
                            tofp16())
        else:
            self.resample3 = Resample2d()

        if args.fp16:
            self.resample4 = nn.Sequential(
                            tofp32(),
                            Resample2d(),
                            tofp16())
        else:
            self.resample4 = Resample2d()

        # Block (FLowNetFusion)
        self.flownetfusion = FlowNetFusion.FlowNetFusion(args, batchNorm=self.batchNorm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
                # init_deconv_bilinear(m.weight)

    def init_deconv_bilinear(self, weight):
        f_shape = weight.size()
        heigh, width = f_shape[-2], f_shape[-1]
        f = np.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([heigh, width])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        min_dim = min(f_shape[0], f_shape[1])
        weight.data.fill_(0.)
        for i in range(min_dim):
            weight.data[i,i,:,:] = torch.from_numpy(bilinear)
        return

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]
        x = torch.cat((x1,x2), dim = 1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2*self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:,3:,:,:], flownetc_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat((x, resampled_img1, flownetc_flow/self.div_flow, norm_diff_img0), dim=1)

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2*self.div_flow)

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x[:,3:,:,:], flownets1_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat((x, resampled_img1, flownets1_flow/self.div_flow, norm_diff_img0), dim=1)

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)

        diff_flownets2_flow = self.resample4(x[:,3:,:,:], flownets2_flow)
        if not diff_flownets2_flow.volatile:
            diff_flownets2_flow.register_hook(save_grad(self.args.grads, 'diff_flownets2_flow'))

        diff_flownets2_img1 = self.channelnorm((x[:,:3,:,:]-diff_flownets2_flow))
        if not diff_flownets2_img1.volatile:
            diff_flownets2_img1.register_hook(save_grad(self.args.grads, 'diff_flownets2_img1'))

        # flownetsd
        flownetsd_flow2 = self.flownets_d(x)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)

        diff_flownetsd_flow = self.resample3(x[:,3:,:,:], flownetsd_flow)
        if not diff_flownetsd_flow.volatile:
            diff_flownetsd_flow.register_hook(save_grad(self.args.grads, 'diff_flownetsd_flow'))

        diff_flownetsd_img1 = self.channelnorm((x[:,:3,:,:]-diff_flownetsd_flow))
        if not diff_flownetsd_img1.volatile:
            diff_flownetsd_img1.register_hook(save_grad(self.args.grads, 'diff_flownetsd_img1'))

        # concat img1 flownetsd, flownets2, norm_flownetsd, norm_flownets2, diff_flownetsd_img1, diff_flownets2_img1
        concat3 = torch.cat((x[:,:3,:,:], flownetsd_flow, flownets2_flow, norm_flownetsd_flow, norm_flownets2_flow, diff_flownetsd_img1, diff_flownets2_img1), dim=1)
        flownetfusion_flow = self.flownetfusion(concat3)

        if not flownetfusion_flow.volatile:
            flownetfusion_flow.register_hook(save_grad(self.args.grads, 'flownetfusion_flow'))

        return flownetfusion_flow

class FlowNet2C(FlowNetC.FlowNetC):
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(FlowNet2C,self).__init__(args, batchNorm=batchNorm, div_flow=20)
        self.rgb_max = args.rgb_max

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]

        # FlownetC top input stream
        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)

        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        out_corr = self.corr(out_conv3a, out_conv3b) # False
        out_corr = self.corr_activation(out_corr)

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)

        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)

        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1,out_deconv3,flow4_up),1)

        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a,out_deconv2,flow3_up),1)

        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return self.upsample1(flow2*self.div_flow)

class FlowNet2S(FlowNetS.FlowNetS):
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(FlowNet2S,self).__init__(args, input_channels = 6, batchNorm=batchNorm)
        self.rgb_max = args.rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)

        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return self.upsample1(flow2*self.div_flow)

class FlowNet2SD(FlowNetSD.FlowNetSD):
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(FlowNet2SD,self).__init__(args, batchNorm=batchNorm)
        self.rgb_max = args.rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)

        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5       = self.predict_flow5(out_interconv5)

        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4       = self.predict_flow4(out_interconv4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3       = self.predict_flow3(out_interconv3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return self.upsample1(flow2*self.div_flow)

class FlowNet2CS(nn.Module):

    def __init__(self, args, batchNorm=False, div_flow = 20.):
        super(FlowNet2CS,self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        if args.fp16:
            self.resample1 = nn.Sequential(
                            tofp32(),
                            Resample2d(),
                            tofp16())
        else:
            self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]
        x = torch.cat((x1,x2), dim = 1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2*self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:,3:,:,:], flownetc_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat((x, resampled_img1, flownetc_flow/self.div_flow, norm_diff_img0), dim=1)

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2*self.div_flow)

        return flownets1_flow

class FlowNet2CSS(nn.Module):

    def __init__(self, args, batchNorm=False, div_flow = 20.):
        super(FlowNet2CSS,self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        if args.fp16:
            self.resample1 = nn.Sequential(
                            tofp32(),
                            Resample2d(),
                            tofp16())
        else:
            self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample2 = nn.Sequential(
                            tofp32(),
                            Resample2d(),
                            tofp16())
        else:
            self.resample2 = Resample2d()


        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]
        x = torch.cat((x1,x2), dim = 1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2*self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:,3:,:,:], flownetc_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat((x, resampled_img1, flownetc_flow/self.div_flow, norm_diff_img0), dim=1)

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2*self.div_flow)

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x[:,3:,:,:], flownets1_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat((x, resampled_img1, flownets1_flow/self.div_flow, norm_diff_img0), dim=1)

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)

        return flownets2_flow

class FlowNetPWC(PwcNet.PwcNet):
    def __init__(self, args, div_flow=20):
        super(FlowNetPWC,self).__init__()
        self.rgb_max = args.rgb_max

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]
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

        return flow * self.div_flow
