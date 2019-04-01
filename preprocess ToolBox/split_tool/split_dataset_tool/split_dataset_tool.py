# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 11:05:26 2018

split dataset tool

split a dataset directory into training set and validation set

@author: zyb_as
"""

import os
import shutil
import random

# TODO: set parameters
# the dataset directory need to be split
src_dir = './dataset'

# the directory to save the split result
train_dir = './train'
valid_dir = './valid'

# the proportion of the training set in the total data set
train_ratio = 0.8

# shuffle dataset or not
shuffle = True

# ------------------------------------
# check if the save directory exists
if os.path.exists(train_dir) == False:
    os.mkdir(train_dir)
if os.path.exists(valid_dir) == False:
    os.mkdir(valid_dir)

# statistical basic infomation
total_count = 0
file_list = []
for root, dirs, files in os.walk(src_dir):
    for file_name in files:
        total_count += 1
        file_path = root + '/' + file_name
        file_list.append(file_path)

# shuffle the image file list
if shuffle:
    random.shuffle(file_list)

# split the training set
split_idx = int(total_count * train_ratio)
for src_file in file_list[:split_idx]:
    file_name = src_file.split('/')[-1]
    dst_file = os.path.join(train_dir, file_name)
    shutil.copy(src_file, dst_file)

# split the validation set
for src_file in file_list[split_idx:]:
    file_name = src_file.split('/')[-1]
    dst_file = os.path.join(valid_dir, file_name)
    shutil.copy(src_file, dst_file)

        