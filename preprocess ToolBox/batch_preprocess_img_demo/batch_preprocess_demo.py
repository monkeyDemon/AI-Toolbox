# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 14:56:45 2018

demo of batch preprocessing images
only need to specify the image_path and save_path

here show how to batch resizing images

@author: zyb_as
"""

import os
import cv2

def batch_resize(image_path, save_path, target_size):
    for root, dirs, files in os.walk(image_path):
            for filename in files:
                cur_imgpath = os.path.join(root, filename)
                print("current image: " + cur_imgpath)
                
                cur_img = cv2.imread(cur_imgpath)
                cur_img = cv2.resize(cur_img, target_size, interpolation=cv2.INTER_CUBIC)
                
                cur_savepath = os.path.join(save_path, filename)
                cv2.imwrite(cur_savepath, cur_img)
            
            
            
if __name__=="__main__":
    
    image_path = 'the root path where saves the images to modify'
    save_path = 'the path to save the images after preprocess'
    
    target_size = (500, 500)
    batch_resize(image_path, save_path, target_size)