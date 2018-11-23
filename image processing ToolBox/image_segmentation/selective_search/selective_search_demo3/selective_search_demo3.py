# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:11:27 2018

selective search demo3
read one image, call the open source library selectivesearch's api
show how to get the segmentation information from the api result
you need to install python library selectivesearch first


Selective Search paper:
J.R.R. Uijlings, et al, Selective Search for Object Recognition, IJCV 2012

python implement on github:
https://github.com/AlpacaDB/selectivesearch

install:
$ pip install selectivesearch

@author: zyb_as
"""

import skimage.data
from skimage import io
import selectivesearch
import matplotlib.pyplot as plt


def main():

    # loading astronaut image
    #img = skimage.data.astronaut()
    
    # loading our image
    img = io.imread('./sheep.jpg')

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=300, sigma=1, min_size=50)
    """
    parameter explanation
    sigma:
        In general we use a Gaussian filter to smooth the image slightly before 
        computing the edge weights, in order to compensate for digitization artifacts.
        We always use a Gaussian with σ = 0.8, which does not produce any 
        visible change to the image but helps remove artifacts.
        (The smaller the sigma, the finer the segmentation result)
    
    min_size:
        If the rect size is reached on min_size, the calculation is stopped.
        
    scale:
        a larger scale causes a preference for larger components
    """
    
    
    # a simple display of how to get the segmentation information from the api result
    # **note** region label is stored in the 4th value of each pixel [r,g,b,(region)]
    
    print(img_lbl.shape)
    
    region_id = 100
    region = regions[region_id]
    print('\nsegmentation information of region {0}:'.format(region_id))
    print(region)
    
    # plot the segmentation info of the resgion_id th region
    r = region['rect']
    print(img_lbl[r[1]:r[1]+r[3], r[0]:r[0]+r[2], 3])
    
    print('\n the total segmentation result visualization')
    plt.imshow(img_lbl[:,:,3])
    plt.show()
    
    

if __name__ == "__main__":
    main()
