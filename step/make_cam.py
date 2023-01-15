import os
import time
import imageio
import argparse
import importlib
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F
from torch.multiprocessing import Process

from utils import imutils, pyutils
from utils.imutils import HWC_to_CHW
from net.resnet38_base import Normalize
from voc12.dataloader import load_img_id_list, load_image_label_list_from_npy, decode_int_filename


start = time.time()


def parse_args(args):
    if args.dataset == 'voc12':
        args.num_classes = 20
        args.img_root = args.voc12_root + '/JPEGImages'
    # elif args.dataset == 'coco':
    #     args.num_classes = 80
    #     args.img_root = args.coco_root
    else:
        raise Exception('Error')
    
    ## model information
    args.num_classes = 20

    # save path
    args.save_type = list()
    if args.cam_npy is not None:
        os.makedirs(args.cam_npy, exist_ok=True)
        args.save_type.append(args.cam_npy)
    if args.cam_png is not None:
        os.makedirs(args.cam_png, exist_ok=True)
        args.save_type.append(args.cam_png)
    if args.crf:
        args.crf_list = list()
        for t in args.crf_t:
            for alpha in args.crf_alpha:
                crf_folder = os.path.join(args.crf, 'crf_{}_{}'.format(t, alpha))
                os.makedirs(crf_folder, exist_ok=True)
                args.crf_list.append((crf_folder, t, alpha))
                args.save_type.append(args.crf_folder)

    # processors
    args.n_processes_per_gpu = [int(_) for _ in args.n_processes_per_gpu]
    args.n_total_processes = sum(args.n_processes_per_gpu)
    return args


def preprocess(image, scale_list, transform):
    '''
    Image is input by HWC, numpy
    multiscale, transformed, fliped original image
    for get a CAM
    '''
    img_size = image.size
    num_scales = len(scale_list)
    multi_scale_image_list = list()
    multi_scale_flipped_image_list = list()

    # insert multi-scale images
    for s in scale_list:
        target_size = (round(img_size[0] * s), round(img_size[1] * s))
        scaled_image = image.resize(target_size, resample=Image.CUBIC)
        multi_scale_image_list.append(scaled_image)
    # transform the multi-scaled image
    for i in range(num_scales):
        multi_scale_image_list[i] = transform(multi_scale_image_list[i])
    # augment the flipped image
    for i in range(num_scales):
        multi_scale_flipped_image_list.append(multi_scale_image_list[i])
        multi_scale_flipped_image_list.append(np.flip(multi_scale_image_list[i], -1).copy())
    return multi_scale_flipped_image_list


def predict_cam(model, image, label, gpu, args):

    original_image_size = np.asarray(image).shape[:2]
    # preprocess image
    multi_scale_flipped_image_list = preprocess(image, args.cam_scales, args.transform)

    cam_list = list()
    model.eval()
    for i, image in enumerate(multi_scale_flipped_image_list):
        with torch.no_grad():
            image = torch.from_numpy(image).unsqueeze(0)
            image = image.cuda(gpu)
            _, cam = model.forward(image)

            cam = F.softmax(cam, dim=1) # softmax in channel dimension
            cam = F.interpolate(cam, original_image_size, mode='bilinear', align_corners=False)[0] # resize to original image

            ## get foreground and background CAM
            cam_fg = cam[:-1].cpu().numpy() * label.reshape(args.num_classes, 1, 1)
            cam_bg = cam[-1:].cpu().numpy()

            if i % 2 == 1: # fliped image for 2 times periodically
                cam_fg = np.flip(cam_fg, axis=-1)
                cam_bg = np.flip(cam_bg, axis=-1)
            cam_list.append((cam_fg, cam_bg))

    return cam_list


def _crf_with_alpha(image, cam_dict, alpha, t=10):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha) # can be calculated by EPS unique method, but use PSA method
    bgcam_score = np.concatenate((bg_score, v), axis=0) # bg, fg order
    crf_score = imutils.crf_inference(image, bgcam_score, labels=bgcam_score.shape[0], t=t)
    n_crf_al = dict()
    n_crf_al[0] = crf_score[0] # bg, fg order
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key+1] = crf_score[i+1] # bg, fg order
    return n_crf_al


def infer_cam_mp(process_id, image_ids, label_list, cur_gpu, args):
    print('process {} starts...'.format(os.getpid()))

    print(process_id, cur_gpu)
    print('GPU Number:', cur_gpu)
    print('{} images per process'.format(len(image_ids)))

    # model = getattr(importlib.import_module(args.network), 'EPS')(args.model_num_classes)
    method = getattr(importlib.import_module(args.network), 'EPS')
    model = method(args.num_classes).cuda(cur_gpu)
    model.load_state_dict(torch.load(args.cam_weights_name))
    model.eval()
    
    with torch.no_grad():
        for i, (img_id, label) in enumerate(zip(image_ids, label_list)):
            # load image
            # print(img_id) ## debug
            img_id = decode_int_filename(img_id)
            img_path = os.path.join(args.img_root, img_id + '.jpg')
            img = Image.open(img_path).convert('RGB')
            org_img = np.asarray(img)

            # infer cam_list
            cam_list = predict_cam(model, img, label, cur_gpu, args)
            
            ''' strange '''
            cam_np = np.array(cam_list)
            cam_fg = cam_np[:, 0]
            sum_cam = np.sum(cam_fg, axis=0)
            norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5) # normalize for each cam for each row

            cam_dict = {}
            for j in range(args.num_classes): # save for valid labels only
                if label[j] > 1e-5:
                    cam_dict[j] = norm_cam[j] # cam dict dim : CHW, C is valid label. key=valid label, value=valid cam

            ## give threshold and save to png
            h, w = list(cam_dict.values())[0].shape
            tensor = np.zeros((args.num_classes + 1, h, w), np.float32)
            for key in cam_dict.keys():
                tensor[key + 1] = cam_dict[key]
            tensor[0, :, :] = args.cam_thres # give threshold
            pred = np.argmax(tensor, axis=0).astype(np.uint8) # CAM prediction. dim HW, value is 0,1,2,3,4,5, ...

            ## save cam
            if args.cam_npy is not None:
                np.save(os.path.join(args.cam_npy, img_id + '.npy'), cam_dict) # dim CHW, probability

            if args.cam_png is not None:
                imageio.imwrite(os.path.join(args.cam_png, img_id + '.png'), pred) # dim CHW, direct class number for evaluation

            if args.crf is not None: # save CRF CAM in npy
                for folder, t, alpha in args.crf_list:
                    cam_crf = _crf_with_alpha(org_img, cam_dict, alpha, t=t) # channel is ... [bg, fg, fg, fg, ...]
                    np.save(os.path.join(folder, img_id + '.npy'), cam_crf) # dim CHW
                    
                    ## give threshold
                    h, w = list(cam_crf.values())[0].shape
                    tensor_crf = np.zeros((args.num_classes + 1, h, w), np.float32)
                    for key in cam_crf.keys():
                        tensor_crf[key + 1] = cam_crf[key]
                    tensor_crf[0, :, :] = 0 # give threshold, usually args.thr
                    pred_crf = np.argmax(tensor_crf, axis=0).astype(np.uint8) # CAM prediction. dim HW, value is 0,1,2,3,4,5, ...
                    imageio.imwrite(os.path.join(folder, img_id + '.png'), pred_crf) # dim CHW
                    
            if i % 10 == 0:
                print('PID{}, {}/{} is complete'.format(process_id, i, len(image_ids)))


def main_mp(args):
    # skip already saved one
    image_ids = load_img_id_list(args.infer_list)
    label_list = load_image_label_list_from_npy(image_ids) # args.dataset
    n_total_images = len(image_ids)
    assert len(image_ids) == len(label_list)

    saved_list = sorted([file[:-4] for file in os.listdir(args.cam_out_dir)]) ## get CAM npy or png. save_type[0]
    n_saved_images = len(saved_list)
    new_image_ids = list()
    new_label_list = list()
    
    ## append imgs not in saved_list
    for i, name in enumerate(image_ids):
        if name not in saved_list:
            new_image_ids.append(name)
            new_label_list.append(label_list[i])
    image_ids = new_image_ids
    label_list = new_label_list

    n_total_processes = args.n_total_processes
    print('===========================')
    print('OVERALL INFORMATION')
    print('n_gpus:', args.n_gpus)
    print('n_processes_per_gpu', args.n_processes_per_gpu)
    print('n_total_processes:', n_total_processes)
    print('n_total_images:', n_total_images)
    print('n_saved_images:', n_saved_images)
    print('n_images_to_proceed', len(image_ids))
    print('===========================')

    sub_image_ids = list()
    sub_label_list = list()

    # split model and data
    split_size = len(image_ids) // n_total_processes
    for i in range(n_total_processes):
        # split image ids and labels
        if i == n_total_processes - 1:
            sub_image_ids.append(image_ids[split_size * i:])
            sub_label_list.append(label_list[split_size * i:])
        else:
            sub_image_ids.append(image_ids[split_size * i:split_size * (i + 1)])
            sub_label_list.append(label_list[split_size * i:split_size * (i + 1)])

    # multi-process
    torch.multiprocessing.set_start_method('spawn')
    gpu_list = list()
    for idx, num in enumerate(args.n_processes_per_gpu):
        gpu_list.extend([idx for i in range(num)]) # 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3, ... 
    processes = list()
    for idx, process_id in enumerate(range(n_total_processes)):
        proc = Process(target=infer_cam_mp,
                       args=(process_id, sub_image_ids[idx], sub_label_list[idx], gpu_list[idx], args))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()
        

def run(args):
    args.crf_alpha = (4, 32) # for low background constraint, alpha 4, strong background constraint, alpha 32
    args = parse_args(args)
    
    args.transform = torchvision.transforms.Compose([np.asarray, Normalize(), HWC_to_CHW])
    
    main_mp(args)

    print(time.time() - start)