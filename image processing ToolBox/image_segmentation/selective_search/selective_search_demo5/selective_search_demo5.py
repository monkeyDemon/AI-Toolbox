# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 14:38:08 2018

selective search demo5

read all the pictures in ./image, call the open source library selectivesearch's api
get the segmentation region from the api result
save the segmentations that most likely corresponding to an true object
all candidate segmentation areas are saved in ./segmentation_result 

you need to install python library selectivesearch first

Selective Search paper:
J.R.R. Uijlings, et al, Selective Search for Object Recognition, IJCV 2012

python implement on github:
https://github.com/AlpacaDB/selectivesearch

install:
$ pip install selectivesearch

@author: zyb_as
"""

import numpy as np
import cv2
import os
import selectivesearch



def saveSegmentationRegion(img_lbl, region, save_root_path, file_name):
    """
    Extract and save the region in image after selective search segmentation
    """
    region_lbl = region['labels']
    
    rect = region['rect']
    x, y, w, h = rect

    seg_lbl = img_lbl[y : y+h, x : x+w, 3]
    seg_img = img_lbl[y : y+h, x : x+w, :3].copy()
    

    # get binary map
    img_bin = np.zeros((h,w))
    for offset_x in range(w):
        for offset_y in range(h):
            if seg_lbl[offset_y, offset_x] in region_lbl:
                img_bin[offset_y, offset_x] = 1
                
    # do close operator(do dilate first than erode) on binary map
    # do close operator is to make the segmentation area more natural
    # build kernel for close operate
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25)) # ellipse kernel
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20)) # ellipse kernel
    #img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)
    img_bin = cv2.dilate(img_bin, kernel_dilate)  # dilate
    img_bin = cv2.erode(img_bin, kernel_erode)   # erode
    
    # use binary map covering irrelevant areas
    for offset_x in range(w):
        for offset_y in range(h):
            if img_bin[offset_y, offset_x] == 0:
                seg_img[offset_y, offset_x, :] = 0
    
    # segmentation image save path: like ./save_root_path/file_name/file_name1.png
    img_save_path = save_root_path + '/' + file_name + '.png'
    cv2.imwrite(img_save_path, seg_img)
        
    
    
    
def main(img_root_path, save_root_path):

    for root, dirs, files in os.walk(img_root_path):
        for file in files:
            print('curent segmentation image: {0}'.format(file))
            file_suffix = file.split('.')[-1]
            file_name = file[: len(file) - len(file_suffix) - 1]
            img_path = os.path.join(root, file)
            
            # build directory to save segmentation result
            save_base_path = save_root_path + '/' + file_name
            if os.path.exists(save_base_path) == False:
                    os.mkdir(save_base_path)
            
            # loading our image
            img = cv2.imread(img_path)
            
            # perform selective search
            img_lbl, regions = selectivesearch.selective_search(
                img, scale=500, sigma=0.5, min_size=50)
            
            
            width = img_lbl.shape[0]
            height = img_lbl.shape[1]
            area = width * height
            
            min_area_threshold = 1600
            diff_area_threshold = area * 0.005
            aspect_ratio_threshold = 6
            
            # save the segmentations that most likely corresponding to an true object
            # regions is  organized in the order they were generated
            # so we read them in reverse order to get the segmentations more likely to correspond an object
            candidates = {}
            count = 0
            for idx in range(len(regions)-1, -1, -1):
                region = regions[idx]
                rect = region['rect']
                size = region['size']                
                
                # excluding regions smaller than min_area
                if size < min_area_threshold:
                    continue
                
                # excluding same rectangle (with different segments)
                if rect in candidates.keys():
                    # here comes a same rectangle (with different segments)
                    if abs(candidates[rect] - size) < diff_area_threshold:
                        # ignore this rectangle if the new come segments is similar  
                        continue
                
                # distorted rects
                x, y, w, h = rect              
                if w / h > aspect_ratio_threshold or h / w > aspect_ratio_threshold:
                    continue
                              
                file_name_count = file_name + '_' + str(count)
                count += 1     
                saveSegmentationRegion(img_lbl, region, save_base_path, file_name_count)

                candidates[rect] = size
    

if __name__ == "__main__":
    root_path = './images' 
    save_root_path = './segmentation_result'
    main(root_path, save_root_path)