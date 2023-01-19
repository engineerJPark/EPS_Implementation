clear
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--tau=0.4 \
--lam=0.9 \
--cam_thres=0.20 \
--crf=None \
--crf_alpha=4 \
--crf_t=10 \
--train_list="voc12/train_aug.txt" \
--infer_list="voc12/train.txt" \
--val_list="voc12/train.txt" \
--num_workers=0 \
--n_gpus 2 \
--n_processes_per_gpu 1 1 \
# --train_pass=False \
# --make_cam_pass=False \
# --eval_pass=False \
# --draw_pass=False \

# fixed gpu numbers, batches, weight loading configuration to strict=False

# --lam=0.9 \