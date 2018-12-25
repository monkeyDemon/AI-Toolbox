# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 14:12:43 2018

A batch verify image tool

After downloading a large amount of image data, usually we find that some 
images can not be open, which may be caused by network transmission errors. 
Therefore, before using these images, use this tool to verify the image data,
and move the unreadable image to the specified path.

@author: zyb_yf, zyb_as
"""

import cv2
import os
import numpy
import shutil
import warnings


src_dir = 'the src dir need to be varify'
dst_dir = './unreadable_imgs'

# the file type that we concerned about(the file with other types will be remove)
concern_file_type = ['jpg', 'JPG', 'png', 'PNG', 'bmp', 'BMP', 'jpeg', 'JPEG'] 

# raise the warning as an exception
warnings.filterwarnings('error') 

if os.path.exists(dst_dir) == False:
    os.mkdir(dst_dir)

for root, dirs, files in os.walk(src_dir):
    for file_name in files:
        src_file = os.path.join(root, file_name)
        dst_file = os.path.join(dst_dir, file_name)
        suffix = file_name.split('.')[-1]
        if suffix not in concern_file_type:
            print('not concern file type!', src_file)
            shutil.move(src_file, dst_file)
            continue
        try:
            img = cv2.imread(src_file)
            if type(img) != numpy.ndarray:
                print('type error!', file_name)
                shutil.move(src_file, dst_file)
        except Warning:
            print('A warning raised!', file_name)
            shutil.move(src_file, dst_file)
