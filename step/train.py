import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import importlib

from voc12 import dataloader
from utils import pyutils, torchutils, imutils
from net.resnet38_base import EPS

def train(args):
    # train & validation & saliency data loading
    train_dataset = dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root, sal_root=args.sal_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_dataset = dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root, sal_root=args.sal_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    # getting max_iteration
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches
    
    # model setting & train mode
    model = EPS(args.num_classes, args.pretrained_path) # model = getattr(importlib.import_module('net.resnet38_base'), 'EPS')(args.num_classes)
    # model.load_pretrained(args.pretrained_path)
    if torch.cuda.device_count() > 1:
        print("There are(is)", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.cuda()
    model.module.train()

    # parameter call & optimizer setting
    param_groups = model.module.get_parameter_groups() # is it okay?
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    # metric, timer
    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):
        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):
            # img, label & cuda
            img = pack['img'].cuda(non_blocking=True) # BCHw
            sal_img = pack['sal_img'].cuda(non_blocking=True) # BHw
            label = pack['label'].cuda(non_blocking=True)

            # prediction
            out, out_cam = model(img)
            
            # classification loss
            loss_cls = F.multilabel_soft_margin_loss(out[:, :-1], label) # for predicted label and GT lable
            
            # saliency loss ... need to be fixed : sal_img should come from dataloader
            fg, bg = torchutils.cam2fg_n_bg(out_cam, sal_img, label) # label should be one hot decoded
            pred_sal = torchutils.psuedo_saliency(fg, bg)
            loss_sal = F.mse_loss(pred_sal.to(torch.float32), \
                F.interpolate(sal_img.unsqueeze(dim=1), size=(pred_sal.shape[-2], pred_sal.shape[-1])).to(torch.float32)) # for pseudo sal map & saliency map

            # total loss
            loss_total = loss_cls + loss_sal
            
            # print(type(loss_total))
            
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

        else: # if one epoch is trained with no error
            validate(model, val_data_loader)
            timer.reset_stage()
            model.module.train()

    torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')


def validate(model, data_loader):
    model.module.eval()
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
            fg, bg = torchutils.cam2fg_n_bg(out_cam, sal_img, label) # label should be one hot decoded
            pred_sal = torchutils.psuedo_saliency(fg, bg)
            # loss_sal = F.mse_loss(pred_sal, sal_img) # for pseudo sal map & saliency map
            loss_sal = F.mse_loss(pred_sal.to(torch.float32), \
                F.interpolate(sal_img.unsqueeze(dim=1), size=(pred_sal.shape[-2], pred_sal.shape[-1])).to(torch.float32))
            
            # total loss
            loss_total = loss_cls + loss_sal

            # adding total loss
            val_loss_meter.add({'loss': loss_total.item()})

    print('loss: %.4f' % (val_loss_meter.pop('loss')))
    model.train()
    return

def run(args):
    train(args)