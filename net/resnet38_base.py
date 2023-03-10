import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import imutils


# batch norm should be fixed in fine tune for WSSS
class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, input):
        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                            training=False, eps=self.eps)

# normal residual block
class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
        super(ResBlock, self).__init__()
        self.same_shape = (in_channels == out_channels and stride == 1)
        if first_dilation == None: first_dilation = dilation

        self.bn_branch2a = FixedBatchNorm(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride, padding=first_dilation, dilation=first_dilation, bias=False)
        self.bn_branch2b1 = FixedBatchNorm(mid_channels)
        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape: # to make shame channel
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):
        branch2 = F.relu(self.bn_branch2a(x))
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2) if not self.same_shape else x

        branch2 = self.conv_branch2a(branch2)
        branch2 = F.relu(self.bn_branch2b1(branch2))
        branch2 = self.conv_branch2b1(branch2)
        x = branch1 + branch2

        if get_x_bn_relu == True:
            return x, x_bn_relu
        else:
            return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.):
        super(BottleneckBlock, self).__init__()
        self.same_shape = (in_channels == out_channels and stride == 1)

        self.bn_branch2a = FixedBatchNorm(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels//4, 1, stride, bias=False)

        self.bn_branch2b1 = FixedBatchNorm(out_channels//4)
        self.dropout_2b1 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b1 = nn.Conv2d(out_channels//4, out_channels//2, 3, padding=dilation, dilation=dilation, bias=False)

        self.bn_branch2b2 = FixedBatchNorm(out_channels//2)
        self.dropout_2b2 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels//2, out_channels, 1, stride, bias=False)

        if not self.same_shape: # to make shame channel
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):
        branch2 = F.relu(self.bn_branch2a(x))
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2)

        branch2 = self.conv_branch2a(branch2)

        branch2 = F.relu(self.bn_branch2b1(branch2))
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = F.relu(self.bn_branch2b2(branch2))
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu
        else:
            return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1a = nn.Conv2d(3,64,3,padding=1,bias=False)
        self.b2 = ResBlock(64,128,128,stride=2)
        self.b2_1 = ResBlock(128,128,128)
        self.b2_2 = ResBlock(128,128,128)

        self.b3 = ResBlock(128,256,256,stride=2)
        self.b3_1 = ResBlock(256,256,256)
        self.b3_2 = ResBlock(256,256,256)

        self.b4 = ResBlock(256,512,512,stride=2)
        self.b4_1 = ResBlock(512,512,512)
        self.b4_2 = ResBlock(512,512,512)
        self.b4_3 = ResBlock(512,512,512)
        self.b4_4 = ResBlock(512,512,512)
        self.b4_5 = ResBlock(512,512,512)

        self.b5 = ResBlock(512,512,1024,stride=1,first_dilation=1,dilation=2)
        self.b5_1 = ResBlock(1024,512,1024,dilation=2)
        self.b5_2 = ResBlock(1024,512,1024,dilation=2)

        self.b6 = BottleneckBlock(1024,2048,stride=1,dilation=4,dropout=0.3)
        self.b7 = BottleneckBlock(2048,4096,dilation=4,dropout=0.5)
        self.bn7 = nn.BatchNorm2d(4096)

        ###############################################
        # utility
        self.not_training = [self.conv1a]
        self.normalize = imutils.Normalize()

    def forward(self, x):
        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x = self.b4(x)
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x, get_x_bn_relu=True)
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)

        x = self.b7(x)
        conv6 = F.relu(self.bn7(x))

        return dict({'conv4':conv4, 'conv5':conv5, 'conv6':conv6})

    def train(self, mode=True):
        super().train(mode) # from nn.Module

        for layer in self.not_training:
            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False
            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

    def load_pretrained(self, filename): # get weight from pretrained one. use args.pretrained for filename
        self.load_state_dict(torch.load(filename))



class EPS(Net):
    def __init__(self, num_classes, filename=None):
        super().__init__()
        if filename is not None: self.load_pretrained(filename)

        self.fc8 = nn.Conv2d(4096, num_classes, 1, bias=False) # background -> +1
        torch.nn.init.xavier_uniform_(self.fc8.weight)

        ###############################################
        # utility
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8]

    def forward(self, x): # give prediction & CAM
        x = super(EPS, self).forward(x)['conv6'] # output is tuple
        x_cam = self.fc8(x)
        x = F.adaptive_avg_pool2d(x_cam, 1)
        x = x.reshape(x.shape[0], -1)

        return x, x_cam # prediction score & CAM

    def get_parameter_groups(self):
        groups = ([],[],[],[]) # 2 grad false, 2 grad true
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # for weight
                if m.weight.requires_grad == True:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)
                # for bias
                if m.bias != None and m.bias.requires_grad == True:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.weight)
                    else:
                        groups[1].append(m.weight)
        return groups

    def load_pretrained(self, filename): # get weight from pretrained one. use args.pretrained for filename
        super().load_state_dict(torch.load(filename))