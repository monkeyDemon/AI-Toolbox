#!/bin/bash

echo 'start generate tf record files...'

# The dataset root directory containing a set of subdirectories representing
# class names. Each subdirectory should contain PNG or JPG encoded images.
_DATASET_ROOT_PATH="dataset_root_path"

# The directory to save the tfrecord files
_TFRECORD_SAVE_PATH="./tfrecord"

# Give a base name for the dataset
_DATASET_BASE_NAME="dataset_name"

# The number of shards per dataset split.
# also determine the ratio of train and validation set
_TRAIN_NUM_SHARDS=9
_VALID_NUM_SHARDS=1

# Seed for repeatability.
_RANDOM_SEED=0

# Specify uniform image zoom size.
_ZOOM_SIZE=256 

# Specify which gpu to be used
_CUDA_VISIBLE_DEVICES='0'


nohup python src/convert_tf_record.py \
	--dataset_root_path=${_DATASET_ROOT_PATH} \
	--tfrecord_save_path=${_TFRECORD_SAVE_PATH} \
	--dataset_base_name=${_DATASET_BASE_NAME} \
	--train_num_shards=${_TRAIN_NUM_SHARDS} \
	--valid_num_shards=${_VALID_NUM_SHARDS} \
	--random_seed=${_RANDOM_SEED} \
    --zoom_size=${_ZOOM_SIZE} \
	--cuda_visible_devices=${_CUDA_VISIBLE_DEVICES} \
	> log/generate_tf_record.out 2>&1 &
	
