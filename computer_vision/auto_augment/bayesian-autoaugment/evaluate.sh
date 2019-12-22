#!/bin/bash

# $1 cmd_id
# $2 gpu
# $3 record_dir
# $4 polices_str


cost_save_path=$3"/cost.txt"
checkpoint_save_dir=$3"/checkpoint"


python evaluator/keras_evaluator/train_and_evaluate.py \
	--model="mobilenet" \
    --pretrain_weight='None' \
	--train_dir="./dataset/train" \
	--val_dir="./dataset/valid" \
    --cmd_id=$1 \
	--gpu_num=$2 \
    --polices_str=$4 \
    --class_num=2 \
    --max_sample_per_class=10000 \
    --checkpoint_filepath=${checkpoint_save_dir} \
    --cost_filepath=${cost_save_path} 
	
echo "evaluate finish"
