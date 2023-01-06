'''
evaluate CAM quality by fast histogram method
'''

import os
import time
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        # mask = (label_true >= 0) & (label_true < self.num_classes)
        mask = (label_true >= 0) & (label_true < self.num_classes) & (label_pred < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        recall = np.diag(self.hist) / self.hist.sum(axis=1)
        # recall = np.nanmean(recall)
        precision = np.diag(self.hist) / self.hist.sum(axis=0)
        # precision = np.nanmean(precision)
        TP = np.diag(self.hist)
        TN = self.hist.sum(axis=1) - np.diag(self.hist)
        FP = self.hist.sum(axis=0) - np.diag(self.hist)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_classes), iu))

        return acc, recall, precision, TP, TN, FP, cls_iu, mean_iu, fwavacc


def get_labels(label_file):
    idx2num = list()
    idx2label = list()
    for line in open(label_file).readlines():
        num, label = line.strip().split()
        idx2num.append(num)
        idx2label.append(label)
    return idx2num, idx2label


def run(args):
    # dataset information
    if args.dataset == 'voc12':
        num_classes = 21
        ignore_label = 255
    elif args.dataset == "coco":
        num_classes = 81
        ignore_label = 255
    else:
        raise("No dataset")

    idx2num, idx2label = get_labels(os.path.join(args.dataset, 'labels.txt'))

    mIOU = IOUMetric(num_classes=num_classes)

    img_ids = open(args.val_list).read().splitlines()

    postfix = '.png'

    st = time.time()
    for idx, img_id in tqdm(enumerate(img_ids)):
        gt_path = os.path.join(args.gt_dir, img_id + postfix)
        pred_path = os.path.join(args.pred_dir, img_id + '.png')
        # pred_path = os.path.join(args.pred_dir, img_id + '.npy')

        gt = Image.open(gt_path) # HW
        w, h = gt.size[0], gt.size[1]
        gt = np.array(gt, dtype=np.uint8)  # shape = [h, w], 0-20 is classes, 255 is ingore boundary

        pred = Image.open(pred_path)
        pred = pred.crop((0, 0, w, h))
        pred = np.array(pred, dtype=np.uint8)  # shape = [h, w]
        mIOU.add_batch(pred, gt)

    acc, recall, precision, TP, TN, FP, cls_iu, miou, fwavacc = mIOU.evaluate()

    mean_prec = np.nanmean(precision)
    mean_recall = np.nanmean(recall)

    print(acc)
    with open(args.save_path, 'w') as f:
        f.write("{:>5} {:>20} {:>10} {:>10} {:>10}\n".format('IDX', 'Name', 'IoU', 'Prec', 'Recall'))
        f.write("{:>5} {:>20} {:>10.2f} {:>10.2f} {:>10.2f}\n".format(
            '-', 'mean', miou * 100, mean_prec * 100, mean_recall * 100))
        for i in range(args.num_classes):
            f.write("{:>5} {:>20} {:>10.2f} {:>10.2f} {:>10.2f}\n".format(
                idx2num[i], idx2label[i][:10], cls_iu[i] * 100, precision[i] * 100, recall[i] * 100))
    print("{:>8} {:>8} {:>8} {:>8} {:>8}".format('IDX', 'IoU', 'Prec', 'Recall', 'ACC'))
    print("{:>8} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}".format(
        'mean', miou * 100, mean_prec * 100, mean_recall * 100, np.mean(acc) * 100))