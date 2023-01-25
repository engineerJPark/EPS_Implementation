import argparse
import os
from utils import pyutils
import torch

import torch.backends.cudnn as cudnn
cudnn.enabled = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    if torch.cuda.is_available(): 
        print("=======Use GPU=======")
    else:
        print("=======Only CPU=======")
    

    parser = argparse.ArgumentParser()

    # Environment
    # parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--voc12_root", default='dataset/VOCdevkit/VOC2012', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    parser.add_argument("--sal_root", default='dataset/SALImages', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./SALImage as subdirectory.")
    parser.add_argument("--network", default="net.resnet38_base", type=str)
    parser.add_argument("--n_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--n_processes_per_gpu", nargs='*', type=int)
    parser.add_argument("--n_total_processes", default=1, type=int)

    # Dataset
    parser.add_argument('--dataset', default='voc12')
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument('--gt_dir', default='dataset/VOCdevkit/VOC2012/SegmentationClass', type=str)
    parser.add_argument("--num_classes", default=20, type=int)
    parser.add_argument("--pretrained_path", default="savefile/pretrained/resnet38.pth", type=str)
    
    ## Augmentation
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--resize_size", default=(256, 512), type=int, nargs='*')
    
    # optimizer
    parser.add_argument("--lr", default=0.01, type=float) # 0.01
    parser.add_argument("--wt_dec", default=5e-4, type=float) # 5e-4
    parser.add_argument("--max_iters", default=20000, type=int)

    # hyper-parameters for EPS
    parser.add_argument("--tau", default=0.4, type=float) # is different on actual report. see do.sh 
    parser.add_argument("--lam", default=0.5, type=float)
    parser.add_argument("--sal_thres", default=0.5, type=float)

    # Class Activation Map parameter
    parser.add_argument("--cam_scales", default=(0.5, 1.0, 1.5, 2.0),
                        help="Multi-scale inferences")
    parser.add_argument("--cam_npy", default="savefile/result/cam_npy", type=str) # "savefile/result/cam" , separate two of them by .npy & .png
    parser.add_argument("--cam_png", default="savefile/result/cam_png", type=str) # "savefile/result/cam" , 
    parser.add_argument("--cam_thres", default=0.20, type=float)

    # ### CRF parameter
    parser.add_argument("--crf", default=None, type=str)
    parser.add_argument("--crf_alpha", nargs='*', type=int)
    parser.add_argument("--crf_t", nargs='*', type=int)
    
    # Output Path
    parser.add_argument("--log_name", default="train_eval", type=str)
    parser.add_argument('--eval_save_path', default='eval_log.txt', type=str)
    parser.add_argument("--cam_weights_name", default="savefile/pretrained/resnet38_eps.pth", type=str)
    parser.add_argument("--cam_out_dir", default="savefile/result/cam_npy", type=str) # npy path
    
    # Step
    parser.add_argument("--train_pass", default=False, type=bool)
    parser.add_argument("--make_cam_pass", default=False, type=bool)
    parser.add_argument("--eval_pass", default=False, type=bool)
    parser.add_argument("--draw_pass", default=False, type=bool)

    args = parser.parse_args()
    
    os.makedirs("savefile/pretrained", exist_ok=True)
    os.makedirs("savefile/result", exist_ok=True)
    os.makedirs("savefile/result/cam_npy", exist_ok=True)
    os.makedirs("savefile/result/cam_png", exist_ok=True)
    os.makedirs("savefile/result/cam_on_img", exist_ok=True)

    pyutils.Logger(args.log_name + '.log')
    print(vars(args))
    
    if args.train_pass is True:
        import step.train
        timer = pyutils.Timer('step.train:')
        step.train.run(args)

    if args.make_cam_pass is True:
        import step.make_cam
        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)
    
    if args.eval_pass is True:
        import step.eval
        timer = pyutils.Timer('step.eval:')
        step.eval.run(args)
    
    # if args.draw_pass is True:
    #     import step.draw
    #     timer = pyutils.Timer('step.draw:')
    #     step.draw.run(args)