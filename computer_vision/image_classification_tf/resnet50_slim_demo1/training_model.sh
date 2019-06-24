#!/bin/bash

# train script
train_script="src/train.py"

# Directory to tfrecord files.
tf_record_dir="./tfrecord"
# Path to pretrained ResNet-50 model.
# you can download pretrained model here https://github.com/tensorflow/models/tree/master/research/slim
checkpoint_path="path to pretrained model/resnet_v1_50.ckpt"
# train from scratch on imagenet pretrained model or continue training on previous model
train_from_scratch=True
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
    --tf_record_dir=${tf_record_dir} \
    --checkpoint_path=${checkpoint_path} \
    --train_from_scratch=${train_from_scratch} \
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
