clear
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--batch_size 8 \
--tau 0.4 \
--lam 0.5 \
--cam_thres 0.20 \
--train_list "voc12/train_aug.txt" \
--infer_list "voc12/train.txt" \
--val_list "voc12/train.txt" \
--num_workers 0 \
# --train_pass=False \
# --make_cam_pass=False \
# --eval_pass=False \
# --draw_pass=False \

# --n_gpus 2 \
# --n_processes_per_gpu 1 1 \