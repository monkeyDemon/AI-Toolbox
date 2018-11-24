# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 11:22:58 2018

selective search demo4
adjustment parameters

read all the pictures in ./image, call the open source library selectivesearch's api
save all the segmentation result in ./segmentation_result 
observe all results to choose the best selective search parameters

you need to install python library selectivesearch first


Selective Search paper:
J.R.R. Uijlings, et al, Selective Search for Object Recognition, IJCV 2012

python implement on github:
https://github.com/AlpacaDB/selectivesearch

install:
$ pip install selectivesearch

@author: zyb_as
"""

import cv2
import selectivesearch
import os



    
       
def main(scale=500, sigma=0.5, min_size=10):
    root_path = './images' 
    save_path = './segmentation_result'
    for root, dirs, files in os.walk(root_path):
        for filename in files:
            img_path = os.path.join(root, filename)
            
            # loading our image
            img = cv2.imread(img_path)
        
            # perform selective search
            img_lbl, regions = selectivesearch.selective_search(
                img, scale=scale, sigma=sigma, min_size=min_size)
            
            # save segmentation result
            img_save_path = os.path.join(save_path, filename)
            cv2.imwrite(img_save_path, img_lbl[:,:,3])

    
    

if __name__ == "__main__":
    scale = 500
    sigma = 0.5
    min_size = 10
    main(scale, sigma, min_size)