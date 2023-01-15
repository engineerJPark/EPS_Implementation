import random
import numpy as np
import os
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from PIL import Image

def pil_resize(img, size, order):
    if size[0] == img.shape[0] and size[1] == img.shape[1]:
        return img

    if order == 3:
        resample = Image.BICUBIC
    elif order == 2:
        resample = Image.BILINEAR
    elif order == 0:
        resample = Image.NEAREST

    return np.asarray(Image.fromarray(img).resize(size[::-1], resample)) # size get as order for width, height

def pil_rescale(img, scale, order):
    height, width = img.shape[:2]
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def random_resize_long(img, min_long, max_long):
    target_long = random.randint(min_long, max_long)
    
    img_list = []
    if isinstance(img, tuple) == False : img = (img, )
    for sub_img in img:
        h, w = sub_img.shape[:2]

        if w < h:
            scale = target_long / h
        else:
            scale = target_long / w
        # img_list.append(pil_rescale(sub_img, scale, 3))
        img_list.append(pil_rescale(sub_img, scale, 2))
    img_list = img_list if len(img) == 1 else img_list
    return img_list

def random_scale(img, scale_range, order):

    target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

    if isinstance(img, tuple):
        return (pil_rescale(img[0], target_scale, order[0]), pil_rescale(img[1], target_scale, order[1]))
    else:
        return pil_rescale(img[0], target_scale, order)

def random_lr_flip(img):

    if bool(random.getrandbits(1)):
        if isinstance(img, tuple):
            return [np.fliplr(m) for m in img]
        else:
            return np.fliplr(img)
    else:
        return img

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def random_crop(images, cropsize, default_values):
    '''
    do random crop, 
    hollow part will be filled with default_values
    '''
    # print(images.shape)
    # print(images[1].shape)
    # print('first:', len(images))

    if isinstance(images, np.ndarray): images = (images,)
    if isinstance(default_values, int): default_values = (default_values,)

    imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, default_values):
        if len(img.shape) == 3:
            cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
        else:
            cont = np.ones((cropsize, cropsize), img.dtype)*f
        cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
        new_images.append(cont)
        
    # print('second:', len(new_images))
    # print(type(new_images))

    if len(new_images) == 1:
        new_images = new_images[0]
        
    # print(len(new_images))
    # print(new_images.shape)
    # print(type(new_images))
    
    return new_images

def top_left_crop(img, cropsize, default_value):
    
    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[:ch, :cw] = img[:ch, :cw]

    return container

def center_crop(img, cropsize, default_value=0):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    sh = h - cropsize
    sw = w - cropsize

    if sw > 0:
        cont_left = 0
        img_left = int(round(sw / 2))
    else:
        cont_left = int(round(-sw / 2))
        img_left = 0

    if sh > 0:
        cont_top = 0
        img_top = int(round(sh / 2))
    else:
        cont_top = int(round(-sh / 2))
        img_top = 0

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        img[img_top:img_top+ch, img_left:img_left+cw]

    return container

def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))

def crf_inference(img, probs, scale_factor=1, iter=10, n_labels=21, gt_prob=0.7):
    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)
    unary = unary_from_softmax(probs)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(iter)
    return np.array(q).reshape((n_labels, h, w)) # probabilties

def _crf_with_alpha(image, cam_dict, bg_score, alpha, iter=10):
    v = np.array(list(cam_dict.values()))
    bgcam_score = np.concatenate((v, bg_score), axis=0) # last label is background label
    
    crf_score = crf_inference(image, bgcam_score, iter=iter, n_labels=bgcam_score.shape[0])
    n_crf_al = dict()
    n_crf_al[0] = crf_score[0] # [bg, fg, fg, ...]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key+1] = crf_score[i+1] # [bg, fg, fg, ...], keys = valid labels, values = cam image
    return n_crf_al


def img2npy(img, img_name, labels):
    '''
    function for save CAM to npy file
    img is cam
    id is name of input image 
    '''
    # npy exporting
    np.save(os.path.join('./cam_result', img_name + '.npy'), {"labels": labels, "cam": img})
    
    
def npy2img(img_name, npy_path='./cam_result'):
    '''
    function for recall npy to show CAM image
    npy_path is the path for the npy file
    img_name is name of the image
    '''
    data_dict = np.load(npy_path + '_' + img_name).item()
    labels = data_dict['labels']
    cam = data_dict['cam']
    
    return labels, cam



'''
### how to use img2npy & npy2img
for step, pack enumerate(data_loader):
    id = pack['name']
    img2npy(cam, id, labels)

from dataset.dataloader import load_img_name_list
img_name_list = load_img_name_list(img_name_list_path)
for id in img_name_list:
    npy2img(id)
'''