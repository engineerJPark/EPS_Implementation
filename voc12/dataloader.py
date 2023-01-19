
import numpy as np
import torch
from torch.utils.data import Dataset
import os.path
import imageio
from utils import imutils
from torchvision import transforms
import PIL

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
IGNORE = 255

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

N_CAT = len(CAT_LIST)

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST)))) # class name mapped to 0,1,2,3, ...

cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()

def decode_int_filename(int_filename):
    '''
    change filename in int data to string data
    cls_labels.npy saved the image name as integer...
    '''
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]

def load_image_label_from_xml(img_name, voc12_root='./dataset/VOCdevkit/VOC2012'):
    '''
    get class index number from xml files
    '''
    from xml.dom import minidom

    elem_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME, decode_int_filename(img_name) + '.xml'))\
        .getElementsByTagName('name')

    multi_cls_lab = np.zeros((N_CAT), np.float32)

    for elem in elem_list:
        cat_name = elem.firstChild.data # name text getting
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):
    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list):
    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])

def get_img_path(img_name, data_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    if 'VOC' in data_root:
        return os.path.join(data_root, IMG_FOLDER_NAME, img_name + '.jpg')
    elif 'SAL' in data_root:
        return os.path.join(data_root, img_name + '.png')
    else:
        raise("there is no right path")

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.int32)
    return img_name_list

def load_img_id_list(dataset_path):
    return load_img_name_list(dataset_path)

class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_arr = np.asarray(img)
        normalized_img = np.zeros_like(img_arr).astype(np.float32)

        # img is given as HWC dimension
        normalized_img[:, :, 0] = (img_arr[:, :, 0] / 255. - self.mean[0]) / self.std[0]
        normalized_img[:, :, 1] = (img_arr[:, :, 1] / 255. - self.mean[1]) / self.std[1]
        normalized_img[:, :, 2] = (img_arr[:, :, 2] / 255. - self.mean[2]) / self.std[2]
        return normalized_img

class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, sal_root,
                 resize_long=None, rescale=None, img_normal=Normalize(), hor_flip=False,
                 crop_size=None, crop_method=None, color=transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                 to_CHW=True):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.sal_root = sal_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_CHW = to_CHW
        self.color = color

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)
        img = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))
        sal_img = np.asarray(imageio.imread(get_img_path(name_str, self.sal_root)), dtype=np.float32) # imageio duplicate grayscale image to 3 dimension
        sal_img = np.mean(sal_img, axis=-1) # keepdims=True
        
        # sal_img should be preprocessed at the same method
        if self.resize_long:
            img, sal_img = imutils.random_resize_long((img, sal_img), self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img, sal_img = imutils.random_scale((img, sal_img), scale_range=self.rescale, order=3)
            
        if self.color:
            img = self.color(PIL.Image.fromarray(img))
            img = np.asarray(img)
        
        if self.hor_flip:
            img, sal_img = imutils.random_lr_flip((img, sal_img))
            
        if self.img_normal: # img alone
            img = self.img_normal(img)
            sal_img = sal_img / 255.

        if self.crop_size:
            if self.crop_method == "random":
                img, sal_img = imutils.random_crop((img, sal_img), self.crop_size, (0, 0)) 
            else:
                img = imutils.top_left_crop((img), self.crop_size, 0)
                sal_img = imutils.top_left_crop((sal_img), self.crop_size, 0)

        if self.to_CHW:
            img = imutils.HWC_to_CHW(img)

        return {'name': name_str, 'img': img, 'sal_img': sal_img} # for time when you need img only, use dictionary

class VOC12ClassificationDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, sal_root,
                 resize_long=None, rescale=None, img_normal=Normalize(), hor_flip=False,
                 crop_size=None, crop_method=None, color=transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)):
        super().__init__(img_name_list_path, voc12_root, sal_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method, color)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        out['label'] = torch.from_numpy(self.label_list[idx])

        return out
