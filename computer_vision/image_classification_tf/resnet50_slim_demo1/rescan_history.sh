#!/bin/bash


# rescan images by url record file 
echo "start rescanning by src/rescan_history.py ..."

# model path can be both .pb or .ckpt
#model_path='log/path of frozen pb file' 
model_path='log/path of ckpt file'

url_dir='rescan/directory to save url files'
url_save_path='rescan/path to save the rescan record file'
log_path='log/rescan_log/path to save the rescan log file'
devices='1'
nohup python src/rescan_history.py \
    --weight_path $model_path \
    --url_dir $url_dir \
    --url_save_path $url_save_path \
    --gpu_device $devices \
    > $log_path 2>&1 &
