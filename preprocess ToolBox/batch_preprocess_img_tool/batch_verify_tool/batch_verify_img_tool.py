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
from PIL import Image
import os
import numpy
import shutil
import warnings


src_dir = 'the directory to verify'
dst_dir = './unreadable_imgs'

# raise the warning as an exception
warnings.filterwarnings('error') 

if os.path.exists(dst_dir) == False:
    os.mkdir(dst_dir)

for root, dirs, files in os.walk(src_dir):
    for file_name in files:
        src_file = os.path.join(root, file_name)
        dst_file = os.path.join(dst_dir, file_name)
        try:
            # check by opencv
            img = cv2.imread(src_file)
            if type(img) != numpy.ndarray:
                print('type error!', file_name)
                shutil.move(src_file, dst_file)
                continue
            # check by PIL Image
            img = Image.open(src_file)
        except Warning:
            print('A warning raised!', file_name)
            shutil.move(src_file, dst_file)
        except:
            print('Error occured!', file_name)
            shutil.move(src_file, dst_file)
print('finish')
