#!/bin/bash

#train_script="./src/train_resnet50.py"
#weight_load_path="/data1/ansheng/cv_strategy/model_zoo/keras_pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
#weight_save_path="./models/semantic_vector_resnet50.h5"
#record_save_path="./log/resnet50_trainlog.out"

train_script="./src/train_vgg16.py"
weight_load_path="/data1/ansheng/cv_strategy/model_zoo/keras_pretrained_model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
weight_save_path="./models/semantic_vector_vgg16.h5"
record_save_path="./log/vgg16_trainlog.out"

label_file="./dataset/labels.txt"
label_num=6
gpu_devices='5'

echo "run ${train_script}"

nohup python ${train_script} \
 --weight_load_path=${weight_load_path} \
 --weight_save_path=${weight_save_path} \
 --label_file=${label_file} \
 --label_num=${label_num} \
 --gpu_devices=${gpu_devices} \
 > ${record_save_path} 2>&1 &

echo "training start..."
