##################################################################################################
# TODO
# CAM on C classes & bg
# match C classes of CAM to saliency images : Ms = lamb*Mfg + (1-lamb)*(1-Mbg)
# Mi -> binary by 0.5
# overlapping with saliency map with 0.4
# Mfg = label * Mi
# Mbg = label * Mi + Mc+1
# get Lsal and Lcls : see papers 
##################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from val import validate
import importlib

from voc12 import dataloader
from utils import pyutils, torchutils, imutils

def train(args):

    model = getattr(importlib.import_module('net.resnet38_base'), 'EPS')()
    
    # train & validation & saliency data loading
    train_dataset = dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_dataset = dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    
    
    # getting max_iteration
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    # parameter call & optimizer setting
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    # model train
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    # metric, timer
    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):
        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):
            # img, label
            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)

            # prediction
            x, x_cam = model(img)
            
            # classification loss
            loss_cls = F.multilabel_soft_margin_loss(x, label) # for predicted label and GT lable
            
            # saliency loss ... need to be fixed : sal_img should come from dataloader
            fg, bg = imutils.cam2fg_n_bg(x_cam, sal_img, label) # label should be one hot decoded
            pred_sal = imutils.psuedo_saliency(fg, bg)
            loss_sal = F.mse_loss(pred_sal, sal_img) # for pseudo sal map & saliency map

            # total loss
            loss_total = loss_cls + loss_sal
            
            # loss addition
            avg_meter.add({'loss': loss_total.item()})

            # backpropagation
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        else:
            validate(model, val_data_loader)
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
    