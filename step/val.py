import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from val import validate
import importlib

from voc12 import dataloader
from utils import pyutils, torchutils, imutils

def validate(model, data_loader):
    model.eval()
    print('validating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    with torch.no_grad():
        for pack in data_loader:
            # img, label & cuda
            img = pack['img'].cuda(non_blocking=True)
            sal_img = pack['sal_img'].cuda(non_blocking=True)
            label = pack['label'].cuda(non_blocking=True)
            
            # prediction
            out, out_cam = model(img)
            
            # classification loss
            loss_cls = F.multilabel_soft_margin_loss(out[:, :-1], label) # for predicted label and GT lable
            
            # saliency loss ... need to be fixed : sal_img should come from dataloader
            fg, bg = imutils.cam2fg_n_bg(out_cam, sal_img, label) # label should be one hot decoded
            pred_sal = imutils.psuedo_saliency(fg, bg)
            loss_sal = F.mse_loss(pred_sal, sal_img) # for pseudo sal map & saliency map
            
            # total loss
            loss_total = loss_cls + loss_sal

            # adding total loss
            val_loss_meter.add({'loss': loss_total.item()})

    print('loss: %.4f' % (val_loss_meter.pop('loss')))
    model.train()
    return
