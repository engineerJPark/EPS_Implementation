clear
python main.py \
--train_list="voc12/train_aug.txt" \
--val_list="voc12/train_aug.txt" \
--infer_list="voc12/train_aug.txt" \
--num_workers=0 \
--n_processes_per_gpu=2 \
# --n_processes_per_gpu=2 \
# --train_pass=False \
# --make_cam_pass=False \
# --eval_pass=False \
# --draw_pass=False \