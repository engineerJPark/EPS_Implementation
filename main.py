import argparse
import os

from utils import pyutils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--voc12_root", default='./dataset/VOCdevkit', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    parser.add_argument("--sal_root", default='./dataset/SALImage', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./SALImage as subdirectory.")

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    # parser.add_argument("--num_classes", default=20, type=int)
    parser.add_argument("--pretrained_path", default="savefile/pretrained/resnet38.pth", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet38", type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    # parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
    #                     help="Multi-scale inferences")
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="net.resnet38_based", type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--n_gpus", type=int, default=2)
    # parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--n_processes_per_gpu", nargs='*', type=int)
    parser.add_argument("--n_total_processes", default=1, type=int)
    # parser.add_argument("--img_root", default='VOC2012', type=str)
    
    ### CRF parameter
    parser.add_argument("--crf", default=None, type=str)
    parser.add_argument("--crf_alpha", nargs='*', type=int)
    parser.add_argument("--crf_t", nargs='*', type=int)
    
    ### CAM parameter
    parser.add_argument("--cam_npy", default="savefile/result/cam", type=str) # "savefile/result/cam" , separate two of them by .npy & .png
    parser.add_argument("--cam_png", default=None, type=str) # "savefile/result/cam" , 
    parser.add_argument("--thr", default=0.20, type=float)
    
    # Evaluation
    # need to be fixed
    parser.add_argument('--dataset', default='voc12', required=True)
    parser.add_argument('--datalist', default='voc12/train.txt', required=True, type=str)
    parser.add_argument('--gt_dir', default='dataset/VOCdevkit/VOC2012/', required=True, type=str)
    parser.add_argument('--pred_dir', default='savefile/cam', required=True, type=str)
    parser.add_argument('--save_path', default='eval_log.txt', required=True, type=str)
    
    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="savefile/pretrained/resnet38.pth", type=str)
    parser.add_argument("--cam_out_dir", default="savefile/result/cam", type=str) # npy path
    

    # Step
    parser.add_argument("--train_pass", default=True)
    parser.add_argument("--make_cam_pass", default=True)
    parser.add_argument("--eval_pass", default=True)
    parser.add_argument("--draw_pass", default=True)


    args = parser.parse_args()
    
    os.makedirs("savefile/pretrained", exist_ok=True)
    os.makedirs("savefile/result", exist_ok=True)
    os.makedirs("savefile/result/cam_on_img", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    
    
    # os.makedirs(args.ir_label_out_dir, exist_ok=True)
    # os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    # os.makedirs(args.ins_seg_out_dir, exist_ok=True)

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
        
    if args.draw_pass is True:
        import step.draw
        timer = pyutils.Timer('step.draw:')
        step.draw.run(args)
