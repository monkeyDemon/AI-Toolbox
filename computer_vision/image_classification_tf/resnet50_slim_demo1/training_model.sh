#!/bin/bash

# train script
train_script="src/train.py"

# train mode: 'scratch' or 'imagenet' or 'continue'
# 'scratch' means train from scratch without pretrained model.
# 'imagenet' means fine tune on imagenet pretrained model.
# 'continue' means continue training on our previous model
train_mode='imagenet'

# Path to pretrained ResNet-50 model. need when train_mode='imagenet' or 'continue'
# the imagenet pretrained model can be download at: 
# https://github.com/tensorflow/models/tree/master/research/slim#Pretrained
checkpoint_path="model_zoo/resnet_v1_50.ckpt"

# Directory to tfrecord files.
tf_record_dir="./tfrecord"
# Path to label file.
label_path="./tfrecord/labels.txt"

# Path to log directory.
log_dir="./log/train_tf_log"
# Path to log file.
log_file="./log/training_log.out"

# Initial learning rate.
learning_rate=0.00001
# Learning rate decay factor.
learning_rate_decay_factor=0.3
# Number of epochs after which learning rate decays. Note: this flag counts '
# epochs per clone but aggregates per sync replicas. So 1.0 means that '
# each clone will go over full epoch individually, but replicas will go '
# once across all replicas.')
num_epochs_per_decay=7
# Number of classes
num_classes=2
# Number of epochs
epoch_num=50
# Batch size
batch_size=32
# Specify which gpu to be used
gpu_device='0'


echo "run ${train_script}"

nohup python ${train_script} \
    --train_mode=${train_mode} \
    --checkpoint_path=${checkpoint_path} \
    --tf_record_dir=${tf_record_dir} \
    --label_path=${label_path} \
    --log_dir=${log_dir} \
    --learning_rate=${learning_rate} \
    --learning_rate_decay_factor=${learning_rate_decay_factor} \
    --num_epochs_per_decay=${num_epochs_per_decay} \
    --num_samples=${num_samples} \
    --num_classes=${num_classes} \
    --epoch_num=${epoch_num} \
    --batch_size=${batch_size} \
    --gpu_device=${gpu_device} \
    > ${log_file} 2>&1 &

echo "training start..."
