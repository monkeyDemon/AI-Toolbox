#!/bin/bash

train_script="./src/train.py"
weight_save_path="./models/smallervgg.h5"
label_file="./dataset/labels.txt"
label_num=6
gpu_devices='5'
record_save_path="./log/train_log.out"

echo "run ${train_script}"

CUDA_VISIBLE_DEVICES=5 nohup python ${train_script} \
 --weight_save_path=${weight_save_path} \
 --label_file=${label_file} \
 --label_num=${label_num} \
 --gpu_devices=${gpu_devices} \
 > ${record_save_path} 2>&1 &
echo "training start..."

#CUDA_VISIBLE_DEVICES=5 python ${train_script} \
# --weight_save_path=${weight_save_path} \
# --label_file=${label_file} \
# --label_num=${label_num} \
# --gpu_devices=${gpu_devices} \

