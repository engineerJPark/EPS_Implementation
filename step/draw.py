import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL

import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os

import voc12.dataloader
from utils import torchutils, imutils

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

cudnn.enabled = True

def _work(process_id, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):
        for iter, pack in enumerate(data_loader):
            # load cam npy
            name_str = pack['name'][0]
            label = pack['label'][0] # one hot encoded
            valid_cat = torch.nonzero(label)[:, 0] # nonzero label index for all batch. codepage class number
            
            img = PIL.Image.open(os.path.join(args.voc12_root, 'JPEGImages', name_str + '.jpg'))
            cam_img = np.load(os.path.join(args.cam_npy, name_str + '.npy'), allow_pickle=True).item()
            # cam_img = PIL.Image.open(os.path.join(args.cam_png, name_str + '.png'))
            
            cam_img_pil = []
            for channel_idx in cam_img.keys(): # range(cam_img.shape[0]): # cam img for each class + coloring
                cam_img_pil.append(PIL.Image.fromarray(np.uint8(cm.jet(cam_img[channel_idx]) * 255)))
            for i in range(len(cam_img_pil)): # superpose on image
                plt.imshow(img, alpha = 0.5)
                plt.imshow(cam_img_pil[i], alpha = 0.4)
                plt.savefig("savefile/result/cam_on_img" + '/cam_%s_%s.png' % (name_str, CAT_LIST[valid_cat[i]]))
                plt.clf()

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDataset(
        args.train_list, 
        voc12_root=args.voc12_root, 
        sal_root=args.sal_root
        )
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()