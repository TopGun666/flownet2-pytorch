#!/usr/bin/env python2.7

import sys
#sys.path.append('/home/zlv30/develop/flownet2/python')
sys.path.append('/home/zlv30/develop/zlv_caffe/python')
import caffe
from caffe.proto import caffe_pb2
import sys, os

import torch
import torch.nn as nn

import argparse, tempfile
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--caffe_model', type=str, help='input model in hdf5 or caffemodel format')
parser.add_argument('--prototxt_template',type=str, help='prototxt template')
parser.add_argument('--weights_path', type=str, default='model_weights',
    help='path to save the pytorch model weights')

opt = parser.parse_args()

opt.rgb_max = 255
opt.fp16 = False
opt.grads = {}

# load models
sys.path.append(opt.weights_path)

import models
from utils.param_utils import *

width = 256
height = 256

if 'PWC-net/' in opt.caffe_model:
    keys = {'INPUT_WIDTH': width,
            'INPUT_HEIGHT': height,
            'SCALE_WIDTH':width,
            'SCALE_HEIGHT':height,
            'SCALE_RATIO_WIDTH':1.,
            'SCALE_RATIO_HEIGHT':1.,}
    template = '\n'.join(np.loadtxt(opt.prototxt_template, dtype=str, delimiter='\n'))
    for k in keys:
        template = template.replace('$%s'%(k),str(keys[k]))
else:
    keys = {'TARGET_WIDTH': width,
            'TARGET_HEIGHT': height,
            'ADAPTED_WIDTH':width,
            'ADAPTED_HEIGHT':height,
            'SCALE_WIDTH':1.,
            'SCALE_HEIGHT':1.,}
    template = '\n'.join(np.loadtxt(opt.prototxt_template, dtype=str, delimiter='\n'))
    for k in keys:
        template = template.replace('$%s$'%(k),str(keys[k]))

prototxt = tempfile.NamedTemporaryFile(mode='w', delete=True)
prototxt.write(template)
prototxt.flush()

net = caffe.Net(prototxt.name, opt.caffe_model, caffe.TEST)

weights = {}
biases = {}
for k, v in net.params.items():
    weights[k] = np.array(v[0].data).reshape(v[0].data.shape)
    if len(v) > 1:
        # the last upsampling layer in PWC-net does not have bias term
        biases[k] = np.array(v[1].data).reshape(v[1].data.shape)
        print (k, weights[k].shape, biases[k].shape)
    else:
        print (k, weights[k].shape)

if 'FlowNet2/' in opt.caffe_model:
    model = models.FlowNet2(opt)

    parse_flownetc(model.flownetc.modules(), weights, biases)
    parse_flownets(model.flownets_1.modules(), weights, biases, param_prefix='net2_')
    parse_flownets(model.flownets_2.modules(), weights, biases, param_prefix='net3_')
    parse_flownetsd(model.flownets_d.modules(), weights, biases, param_prefix='netsd_')
    parse_flownetfusion(model.flownetfusion.modules(), weights, biases, param_prefix='fuse_')

    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(opt.weights_path, 'FlowNet2_checkpoint.pth.tar'))

elif 'FlowNet2-C/' in opt.caffe_model:
    model = models.FlowNet2C(opt)

    parse_flownetc(model.modules(), weights, biases)
    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(opt.weights_path, 'FlowNet2-C_checkpoint.pth.tar'))

elif 'FlowNet2-CS/' in opt.caffe_model:
    model = models.FlowNet2CS(opt)

    parse_flownetc(model.flownetc.modules(), weights, biases)
    parse_flownets(model.flownets_1.modules(), weights, biases, param_prefix='net2_')

    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(opt.weights_path, 'FlowNet2-CS_checkpoint.pth.tar'))

elif 'FlowNet2-CSS/' in opt.caffe_model:
    model = models.FlowNet2CSS(opt)

    parse_flownetc(model.flownetc.modules(), weights, biases)
    parse_flownets(model.flownets_1.modules(), weights, biases, param_prefix='net2_')
    parse_flownets(model.flownets_2.modules(), weights, biases, param_prefix='net3_')

    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(opt.weights_path, 'FlowNet2-CSS_checkpoint.pth.tar'))

elif 'FlowNet2-CSS-ft-sd/' in opt.caffe_model:
    model = models.FlowNet2CSS(opt)

    parse_flownetc(model.flownetc.modules(), weights, biases)
    parse_flownets(model.flownets_1.modules(), weights, biases, param_prefix='net2_')
    parse_flownets(model.flownets_2.modules(), weights, biases, param_prefix='net3_')

    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(opt.weights_path, 'FlowNet2-CSS-ft-sd_checkpoint.pth.tar'))

elif 'FlowNet2-S/' in opt.caffe_model:
    model = models.FlowNet2S(opt)

    parse_flownetsonly(model.modules(), weights, biases, param_prefix='')
    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(opt.weights_path, 'FlowNet2-S_checkpoint.pth.tar'))

elif 'FlowNet2-SD/' in opt.caffe_model:
    model = models.FlowNet2SD(opt)

    parse_flownetsd(model.modules(), weights, biases, param_prefix='')

    state = {'epoch': 0,
             'state_dict': model.state_dict(),
             'best_EPE': 1e10}
    torch.save(state, os.path.join(opt.weights_path, 'FlowNet2-SD_checkpoint.pth.tar'))

elif 'PWC-net/' in opt.caffe_model:
    model = models.FlowNetPWC(opt)

    parse_pwcnet(model.modules(), weights, biases, param_prefix='')

    state = {'epoch': 0,
            'state_dict': model.state_dict(),
            'best_EPE': 1e10}
    torch.save(state, os.path.join(opt.weights_path, 'PWC-Net_checkpoint.pth.tar'))

else:
    print 'model type cound not be determined from input caffe model %s'%(opt.caffe_model)
    quit()
print ("done converting ", opt.caffe_model)
