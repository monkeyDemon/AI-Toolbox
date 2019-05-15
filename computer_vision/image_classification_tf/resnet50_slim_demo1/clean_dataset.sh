#!/bin/bash

echo 'start cleaning dataset...'

_SRC_DIR='img_dir_to_clean'
_ERROR_FORMAT_DIR='tmp'
echo "start cleaning $_SRC_DIR"
nohup python src/batch_verify_img_tool.py \
	--src_dir=${_SRC_DIR} \
	--error_format_dir=${_ERROR_FORMAT_DIR} \
	> log/clean_dataset_record.out 2>&1 &
	
