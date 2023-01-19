import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import importlib

from voc12 import dataloader
from utils import pyutils, torchutils, imutils


def cam2fg_n_bg(cam, sal_img, label, num_classes=20, sal_thres=0.5, tau=0.4):
    '''
    cam image = localization map for C classes & 1 background, BCHW dimension
    saliency map = be in torch 
    image-level label; should be binary index label
    num_classes dont include the background
    '''
    ## getting saliency map
    # pred_sal = F.softmax(cam, dim=1) # This value is so small...
    b,_,h,w = cam.shape

    fg = torch.zeros((b, num_classes + 1, h, w)).float().cuda()
    bg = torch.zeros((b, num_classes + 1, h, w)).float().cuda()
    
    ## set overlapping ratio  for each channel & get right label index for indicating the CAM
    ## sum up for each channel -> get ratio for each channel
    overlap_ratio = ((cam[:, :-1] > sal_thres) * (sal_img > sal_thres)).reshape(b, num_classes, -1).sum(-1) / \
        ((cam[:, :-1] > sal_thres) + 1e-5).reshape(b, num_classes, -1).sum(-1) 
    fg_channel = (overlap_ratio > tau).reshape(b, num_classes, 1, 1).expand(b, num_classes, h, w) ## all the value is False ...

    fg[:,:-1] = cam[:, :-1] * fg_channel.to(torch.float) # valid channel for fg
    bg[:,:-1] = cam[:, :-1] * (~fg_channel).to(torch.float) # valid channel for bg
    bg[:,-1] = cam[:, -1] # for summing all element of M_c+1, True
    
    ## get right prediction of saliency
    ## sum up for all channel dimension
    fg = torch.sum(fg, dim=1).cuda() # BHW
    bg = torch.sum(bg, dim=1).cuda() # BHW
    
    return fg, bg


def psuedo_saliency(fg, bg, lamb = 0.5):
    '''
    use with cam2fg_n_bg
    getting saliency prediction by prediction of foreground & background
    '''
    pred_sal_map = lamb * fg + (1 - lamb) * (1 - bg)
    
    return pred_sal_map # BHW


def validate(model, data_loader):
    if torch.cuda.device_count() > 1:
        model.module.eval()
    else:
        model.eval()
    
    print('validating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter()

    with torch.no_grad():
        for pack in data_loader:
            # img, label & cuda
            img = pack['img'].cuda(non_blocking=True) # BCHW
            sal_img = pack['sal_img'].cuda(non_blocking=True) # BHW
            label = pack['label'].cuda(non_blocking=True)
     
            # prediction
            out, out_cam = model(img)
            out_cam = F.softmax(out_cam, dim=1)
            b, _, h, w = out_cam.shape # get original size
            sal_img = F.interpolate(sal_img.unsqueeze(dim=1), size=(h, w), mode='bilinear')
            
            # classification loss
            loss_cls = F.multilabel_soft_margin_loss(out[:, :-1], label) # for predicted label and GT lable
            
            ## this part is part of the bug    
            fg, bg = cam2fg_n_bg(out_cam, sal_img, label) # label should be one hot decoded
            pred_sal = psuedo_saliency(fg, bg)

            loss_sal = F.mse_loss(pred_sal, sal_img.squeeze(dim=1))

            # total loss
            loss_total = loss_cls + loss_sal

            # adding total loss
            val_loss_meter.add({'loss': loss_total.item()})
            val_loss_meter.add({'loss_cls': loss_cls.item()}) ## debug
            val_loss_meter.add({'loss_sal': loss_sal.item()}) ## debug

    print('validation loss_total: %.4f' % (val_loss_meter.pop('loss')))
    print('validation loss_cls: %.4f' % (val_loss_meter.pop('loss_cls')))
    print('validation loss_sal: %.4f' % (val_loss_meter.pop('loss_sal')))
    
    if torch.cuda.device_count() > 1:
        model.module.train()
    else:
        model.train()
    
    return


def run(args):
    # train & validation & saliency data loading
    train_dataset = dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root, sal_root=args.sal_root,
                                                                resize_long=args.resize_size, hor_flip=True,
                                                                crop_size=args.crop_size, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_dataset = dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root, sal_root=args.sal_root,
                                                              crop_size=args.crop_size)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    
    # model setting & train mode
    model_type = getattr(importlib.import_module('net.resnet38_base'), 'EPS')
    model = model_type(args.num_classes + 1)
    model.load_state_dict(torch.load(args.pretrained_path), strict=False)
    
    if torch.cuda.device_count() > 1:
        print("There are(is)", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model.cuda()
        model.module.train()
        param_groups = model.module.get_parameter_groups()
    else:
        print("There are(is) only 1 GPUs!")
        model.cuda()
        model.train()
        param_groups = model.get_parameter_groups()
    
    # parameter call & optimizer setting
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=args.max_iters)

    # metric, timer
    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()

    # for ep in range(args.cam_num_epoches):
    for it in range(args.max_iters):
        # print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        # for step, pack in enumerate(train_data_loader):
        # img, label & cuda
        pack = next(iter(train_data_loader))
        img = pack['img'].cuda(non_blocking=True) # BCHW
        sal_img = pack['sal_img'].cuda(non_blocking=True) # BHW
        label = pack['label'].cuda(non_blocking=True)
    
        # prediction
        out, out_cam = model(img)
        out_cam = F.softmax(out_cam, dim=1)
        b, _, h, w = out_cam.shape 
        sal_img = F.interpolate(sal_img.unsqueeze(dim=1), size=(h, w), mode='bilinear') # downsize images
        
        # classification loss
        loss_cls = F.multilabel_soft_margin_loss(out[:, :-1], label) # for predicted label and GT lable
        
        # getting predicted saliency
        fg, bg = cam2fg_n_bg(out_cam, sal_img, label) # label should be one hot decoded
        pred_sal = psuedo_saliency(fg, bg)
        
        # saliency loss 
        loss_sal = F.mse_loss(pred_sal, sal_img.squeeze(dim=1))

        # total loss
        loss_total = loss_cls + loss_sal

        # loss addition
        avg_meter.add({'loss': loss_total.item(), 'loss_cls': loss_cls.item(), 'loss_sal': loss_sal.item()})

        # backpropagation
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if (optimizer.global_step-1)%500 == 0:
            timer.update_progress(optimizer.global_step / args.max_iters)

            print('step:%5d/%5d' % (optimizer.global_step - 1, args.max_iters),
                    'loss:%.4f' % (avg_meter.pop('loss')),
                    'loss_cls:%.4f' % (avg_meter.pop('loss_cls')),
                    'loss_sal:%.4f' % (avg_meter.pop('loss_sal')),
                    'etc:%s' % (timer.str_estimated_complete()), flush=True)
        timer.reset_stage()
        
    else: # if one epoch is trained with no error
        validate(model, val_data_loader)
            

    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), args.cam_weights_name)
    else:
        torch.save(model.state_dict(), args.cam_weights_name)