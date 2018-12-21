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


src_dir = './need_valid'
dst_dir = './unreadable_imgs'

if os.path.exists(dst_dir) == False:
    os.mkdir(dst_dir)

for file_name in os.listdir(src_dir):
    src_file = os.path.join(src_dir, file_name)
    img = cv2.imread(src_file)
    if type(img) != numpy.ndarray:
        print(file_name, type(img))
        dst_file = os.path.join(dst_dir, file_name)
        shutil.move(src_file, dst_file)
