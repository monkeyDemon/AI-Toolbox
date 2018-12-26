# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:26:52 2018

split folder tool

split a floder into several subfolders

for example:
    folder DATASET save a lot of images
    this demo will create a few subfolders A, B, C...
    and copy image data evenly into individual subfolders

@author: zyb_as
"""

import os
import shutil
import numpy as np

# the folder need to be split
floder_path = './sfw'

# the base name of the subfolders
subfloder_base_path = './sfw_sub'

# specify the number of subfolders
subfloder_num = 3

# find the total file number in the specified floder
file_cnt = 0
for root, dirs, files in os.walk(floder_path):
    for file in files:
        file_cnt += 1


# create subfloder(if doesn't exist)
split_list = np.linspace(0, file_cnt, num = subfloder_num, endpoint=False)
for subfloder_idx in range(1, subfloder_num + 1):
    subfloder_dir = subfloder_base_path + str(subfloder_idx)
    if os.path.exists(subfloder_dir) == False:
        os.mkdir(subfloder_dir)

# split files
counter = 0
subfloder_idx = 1
split_value = split_list[subfloder_idx - 1]
subfloder_dir = ''
for root, dirs, files in os.walk(floder_path):
    for file in files:
        if counter >= split_value:
            subfloder_dir = subfloder_base_path + str(subfloder_idx)
            subfloder_idx += 1
            if subfloder_idx > subfloder_num:
                split_value = file_cnt + 1
            else:
                split_value = split_list[subfloder_idx - 1]
        
        src_file = os.path.join(root, file)
        dst_file = os.path.join(subfloder_dir, file)
        shutil.copyfile(src_file, dst_file)
        counter += 1