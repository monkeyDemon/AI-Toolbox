# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:58:15 2018

selective search 调用demo1
读取一张图片，调用开源库selectivesearch的方法得到目标建议框并展示
使用前需要先看装开源库selectivesearch


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
import matplotlib.patches as mpatches


"""
# 一个最简单的调用示例
img = skimage.data.astronaut()
print(type(img))

img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
print(type(img_lbl))

print(regions[:10])
"""


def main():

    # loading astronaut image
    img = skimage.data.astronaut()
    
    # loading our image
    #img = io.imread('./cake.jpg')

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.8, min_size=10)
    
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
    
    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        
        if w / h > 1.2 or h / w > 1.2:
            continue
        
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    main()
