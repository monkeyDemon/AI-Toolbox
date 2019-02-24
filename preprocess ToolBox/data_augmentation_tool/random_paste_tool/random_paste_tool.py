# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:04:23 2018

random paste tool

This is a random paste tool that I built to solve a data enhancement requirement in real work. 

I think the idea of this tool is quite interesting, you may wish to give it a try, 
if the problem you care about has the following characteristics:
    1. The sample size is small and data enhancement is required.
    2. The target object you care about always appears in a specific type of background
    3. The background image is easy to collect

Since the real needs encountered in my work are sensitive, 
let us take the following scenario as an example.

Imagine that we are trying to solve a classification problem that 
determine if a bird is included in a given picture.
    1. Suppose the sample of birds is very difficult to collect (although this is not the case -_-)
    2. Images containing birds usually appear in fixed scenes (such as flowers or sky)
    3. The image of the flower is very easy to collect

In this case, we can use our tools. The general steps are:
    1. First, the target object (bird) is segmented from the image to form a object segmentation set.
    2. Then, collect a large number of background images as a background set
    3. Randomly pick a segment from the object set and paste it into a random background image.
    4. Repeat step 3 continuously, and add other data enhancement methods such as zoom, rotate, etc.

I think my idea, compared to the commonly used image enhancement method, 
makes more changes to the image pixels without destroying the distribution law of the real sample.
So it may get better precision and robustness, this needs to be verified by actual problems.

If you are interested in this tool, try it out~


python version: Python 3.5

@author: zyb_as
"""

import os
import random
import cv2
import numpy as np

class RandomPasteTool(object):

    def __init__(self, obj_seg_path, back_path, result_save_path, tag = 'None'):
        self.result_save_path = result_save_path
        self.obj_seg_set = self._loadImages(obj_seg_path)
        self.back_set = self._loadImages(back_path)
        self.obj_set_size = len(self.obj_seg_set)
        self.back_set_size = len(self.back_set)
        self.tag = tag        # give a tag to describe the mission(this is a good habit)
        self.serial_num = 0   # the final image save name will be: tag&serial_num.png
    


    def randomPasteN(self, n):
        """
        random paste n images
        """
        for i in range(n):
            print(i)
            self._randomPasteImg()
            
        
    
    def _randomPasteImg(self):
        # random select object image and background image
        obj_idx = random.randint(0, self.obj_set_size - 1)
        back_idx = random.randint(0, self.back_set_size - 1)
        obj_img = self.obj_seg_set[obj_idx]
        back_img = self.back_set[back_idx]
        back_img = np.copy(back_img)
        
        # get image size of object image and background image
        obj_height = obj_img.shape[0]
        obj_width = obj_img.shape[1]
        back_height = back_img.shape[0]        
        back_width = back_img.shape[1]
        
        # random resize the object image
        obj_back_ratio = random.uniform(0.2, 0.8)
        obj_height = int(back_height * obj_back_ratio)
        obj_width = int(back_width * obj_back_ratio)
        obj_img=cv2.resize(obj_img, (obj_width, obj_height), interpolation=cv2.INTER_CUBIC)
        
        # random the location to paste
        x_lefttop = random.randint(0, back_width - obj_width - 1)
        y_lefttop = random.randint(0, back_height - obj_height - 1)
        
        #print(obj_img.shape)
        #print("{0}, {1}".format(obj_width, obj_height))
        #print(back_img.shape)
        #print("{0}, {1}\n".format(back_width, back_height))
        
        # paste the object image into background image
        for x_offset in range(0, obj_width):
            for y_offset in range(0, obj_height):
                #print(type(obj_img[y_offset][x_offset]))
                if sum(obj_img[y_offset][x_offset]) != 0:
                    back_img[y_lefttop + y_offset][x_lefttop + x_offset] = obj_img[y_offset][x_offset]
        
        # save the random paste result
        cur_save_name = self.tag + '&' + str(self.serial_num) + '.png'
        cur_save_path = os.path.join(self.result_save_path, cur_save_name)
        cv2.imwrite(cur_save_path, back_img)
        self.serial_num += 1
        
            
        
        
    def _loadImages(self, root_path):
        img_list = []
        img_suffix_set = {'jpg', 'jpeg', 'png', 'bmp'}
        for root, dirs, files in os.walk(root_path):
            for file in files:
                suffix = file.split('.')[-1]
                if suffix in img_suffix_set: # judge if fille is a image file
                    cur_img_path = os.path.join(root, file)
                    cur_img = cv2.imread(cur_img_path)
                    img_list.append(cur_img)
        return img_list





if __name__ == "__main__":
    result_save_path = "./random_paste_result"
    obj_seg_path = "./bird_segmentation"
    back_path = "./flowers"
    
    tool = RandomPasteTool(obj_seg_path, back_path, result_save_path, 'bird_paste_flower')
    tool.randomPasteN(10)