# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:21:52 2019

demo to use tensorflow api do data augmentation

this is a visualization demo
Here we encapsulate a script preprocessing.py, provide a variety of 
data augmentation methods, we can test these methods by visualize images.

@author: zyb_as
"""

import tensorflow as tf
import cv2
import numpy as np
import preprocessing
import os
from PIL import Image


img_dir = "data_augmentation_test/"
save_dir = "result/"       
repeat_num = 10

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
def main(_):
    with tf.Session() as sess:
        for filename in os.listdir(img_dir):
            print(filename)
            for cnt in range(repeat_num):
                # load image
                img_cv2_data = cv2.imread(os.path.join(img_dir, filename))
                # bgr-> rgb
                b,g,r = cv2.split(img_cv2_data) 
                img = cv2.merge([r,g,b])
    
                # build shape tensor
                shape = img_cv2_data.shape 
                original_img_shape = tf.constant([shape[0], shape[1], 3])
                
                # build image tensor
                img = tf.identity(img)
                
                # do data augmentation
                img = preprocessing.preprocess(img, original_img_shape, is_training=True)
                img = sess.run(img)
                
                # save result
                # img range is [-1,1], we need to map to 0-255 when save
                img = img + 1
                img = img * 128
                img = img.astype(np.uint8) 
                img = Image.fromarray(img)
                save_path = os.path.join(save_dir, filename[:-4] + '_' + str(cnt) + '.jpg')
                img.save(save_path)
    

if __name__ == "__main__":
    tf.app.run()