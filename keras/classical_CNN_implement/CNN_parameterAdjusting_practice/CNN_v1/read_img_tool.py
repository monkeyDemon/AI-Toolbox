# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:37:23 2018

@author: zyb_as
"""
import os
import numpy as np
from PIL import Image 
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import cv2

"""
# 读取硬盘存储的图片数据集为常规格式的训练集数组X,Y的demo
# 请将需要读取的不同类别的数据集按文件夹存放到指定目录datasetRootPath下
# 最终将所有样本构造为trainX和trainY存储于numpy数组中
# 注意预先将所有数据集读入内存不是一种很合理的方法，样本数量不能过多
# 最终返回的结果为channellast格式,及适用于tensorflow的数据格式(B,H,W,C)
"""

# 由于不清楚数据集的大小，这里封装一个动态数组，用于动态添加图像到numpy数组中
class DynamicArray(object):
    def __init__(self, aimImgSize):
        aimsize = [0 for x in range(len(aimImgSize) + 1)]
        aimsize[0] = 100
        aimsize[1:] = [x for x in aimImgSize]
        self._data = np.zeros(aimsize)
        self._size = 0
        self._aimSize = aimsize

    def get_data(self):
        return self._data[:self._size]

    def append(self, value):
        if len(self._data) == self._size:
            self._aimSize[0] = int(len(self._data)*2)
            self._data = np.resize(self._data, tuple(self._aimSize))
        self._data[self._size] = value
        self._size += 1


def process_image_channels(image):
    """
    处理图像通道数
    防止由于图像数据格式的不一致带来的小"惊喜"
    """
    if image.mode == 'RGBA':   # process the 4 channels .png
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
    elif image.mode != 'RGB':  # process the 1 channel image
        image = image.convert("RGB")
    return image


def getDataSet(datasetRootPath, aimSize, verbose=1):
    """
    供外部调用的方法, 读取数据集
    返回数据集X和标签Y
    数据集X格式为numpy array, 每行带代表一个样本
    # datasetRootPath: 存放各个类别图像数据文件夹的根目录
    # aimSize: 生成数据集（numpy array）的大小, 如(224, 224, 3)
    # verbose: 是否打印读取到的文件路径（方便观察进度），默认打印
    """
    categoryLabel = -1
    TrainSetX = DynamicArray(aimSize)
    TrainSetY = DynamicArray([1])
    for root, dirs, files in os.walk(datasetRootPath):
        if(categoryLabel == -1):
            categoryLabel = categoryLabel + 1
            continue
        
        # 正在处理第categoryLabel个类别的图像数据
        print("current category: " + str(categoryLabel))
        
        # 遍历当前类别categoryLabel对应的文件夹root中的所有图像数据
        for filename in files:
            imgpath = os.path.join(root, filename)
            if verbose:
                print(imgpath)
            #curImg = cv2.imread(imgpath)  # opencv 读取中文文件名图片不方便，且格式为BGR
            curImg = Image.open(imgpath)   # 因此这里使用PIL库读取，主要要手动转为nunmpy array
            curImg = process_image_channels(curImg) # 检查并修改图像格式
            curImg = curImg.resize(aimSize[:2]) 
            curImg = np.array(curImg)
            TrainSetX.append(curImg)
            TrainSetY.append(categoryLabel)
        
        categoryLabel = categoryLabel + 1
        print("------------\n\n")
    return TrainSetX.get_data(), TrainSetY.get_data()