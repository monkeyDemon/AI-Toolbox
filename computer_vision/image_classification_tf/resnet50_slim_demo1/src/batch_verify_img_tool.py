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
import tensorflow as tf

flags = tf.app.flags
    
flags.DEFINE_string('src_dir',
                    '',
                    'Path to src images (directory).')
flags.DEFINE_string('error_format_dir',
                    '',
                    'Path to format error images (directory).')
FLAGS = flags.FLAGS

src_dir = FLAGS.src_dir
dst_dir = FLAGS.error_format_dir

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
            img_cv = cv2.imread(src_file)
            if type(img_cv) != numpy.ndarray:
                print('type error!', file_name)
                shutil.move(src_file, dst_file)
                continue
            # check by PIL Image
            img = Image.open(src_file)

            # check channel number
            shape = img_cv.shape
            if len(shape) == 3:
                pass # this image is valid
            elif len(shape) == 1:
                # change channel num to 3 
                print("change {} from gray to rgb".format(file_name))
                img_rgb = cv2.merge((img_cv, img_cv, img_cv))
                cv2.imwrite(src_file, img_rgb)
            else:
                print('channel number error!', file_name)
                shutil.move(src_file, dst_file)
                continue
        except Warning:
            print('A warning raised!', file_name)
            shutil.move(src_file, dst_file)
        except:
            print('Error occured!', file_name)
            shutil.move(src_file, dst_file)
print('finish')
