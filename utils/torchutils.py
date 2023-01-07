import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
import math


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

class SGDROptimizer(torch.optim.SGD):

    def __init__(self, params, steps_per_epoch, lr=0, weight_decay=0, epoch_start=1, restart_mult=2):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.local_step = 0
        self.total_restart = 0

        self.max_step = steps_per_epoch * epoch_start
        self.restart_mult = restart_mult

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.local_step >= self.max_step:
            self.local_step = 0
            self.max_step *= self.restart_mult
            self.total_restart += 1

        lr_mult = (1 + math.cos(math.pi * self.local_step / self.max_step))/2 / (self.total_restart + 1)

        for i in range(len(self.param_groups)):
            self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.local_step += 1
        self.global_step += 1


def split_dataset(dataset, n_splits):

    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out

def cam2fg_n_bg(cam, sal_img, label, num_classes=20, sal_thres=0.5, tau=0.4):
    '''
    cam image = localization map for C classes & 1 background, BCHW dimension
    saliency map = be in torch 
    image-level label; should be binary index label
    num_classes dont include the background
    '''
    b,c,h,w = cam.shape
    sal_img = F.interpolate(sal_img.unsqueeze(dim=1), size=(h, w)) # .squeeze(dim=1)
    
    # print(sal_img.shape)
    
    # getting saliency map & label map setting
    pred_sal = F.softmax(cam, dim=1)
    label_map = label.reshape(b, num_classes, 1, 1).expand(b, num_classes, h, w).bool()
    label_map_fg = torch.zeros((b, num_classes + 1, h, w)).bool().cuda()
    label_map_bg = torch.zeros((b, num_classes + 1, h, w)).bool().cuda()
    label_map_fg[:, :-1] = label_map.clone()
    label_map_bg[:, num_classes] = 1 # for summing all element of M_c+1, True
    
    # set overlapping ratio & get right label index for indicating the CAM

    overlap_ratio = ((pred_sal[:, :-1].detach() > sal_thres) * (sal_img > sal_thres)).reshape(b, num_classes, -1).sum(-1) / \
        ((pred_sal[:, :-1].detach() > sal_thres) + 1e-5).reshape(b, num_classes, -1).sum(-1) # get overlapping ratio for each channel, use * instead of &
    valid_channel_map = (overlap_ratio > tau).reshape(b, num_classes, 1, 1).expand(b, num_classes, h, w)
    label_map_fg[:,:-1] = label_map * valid_channel_map # instead of & or and 
    label_map_bg[:,:-1] = label_map * (~valid_channel_map)
    
    fg = torch.zeros_like(pred_sal, dtype=torch.float).cuda()
    bg = torch.zeros_like(pred_sal, dtype=torch.float).cuda()
    
    # print(label_map_fg.dtype)
    
    fg[label_map_fg] = pred_sal[label_map_fg]
    bg[label_map_bg] = pred_sal[label_map_bg]
    
    # get right prediction of saliency
    fg = torch.sum(fg, dim=1, keepdim=True).cuda()
    bg = torch.sum(bg, dim=1, keepdim=True).cuda()
    
    return (fg, bg)
            
def psuedo_saliency(fg, bg, lamb = 0.5):
    '''
    use with cam2fg_n_bg
    getting saliency prediction by prediction of foreground & background
    '''
    pred_sal_map = lamb*fg + (1 - lamb)*(1 - bg)
    return pred_sal_map
