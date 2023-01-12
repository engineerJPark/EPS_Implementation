clear
python main.py \
--cam_thres=0.20
--train_list="voc12/train_aug.txt" \
--val_list="voc12/train_aug.txt" \
--infer_list="voc12/train_aug.txt" \
--num_workers=0 \
--n_processes_per_gpu=2 \
--make_cam_pass=False \
--eval_pass=False \
--draw_pass=False \
# --train_pass=False \

# --crf=True \
# --crf_alpha=4, 32 \
# --crf_t=10 \
