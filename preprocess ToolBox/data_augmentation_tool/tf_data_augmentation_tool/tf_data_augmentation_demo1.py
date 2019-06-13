# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:30:49 2019

demo to use tensorflow api do data augmentation

this is a quick start demo
just test the basic tensorflow api and show

@author: zyb_as
"""

import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile('./data_augmentation_test/1.jpg','rb').read()

with tf.Session() as sess:
    print("\n original image")
    img_data = tf.image.decode_jpeg(image_raw_data)
    plt.imshow(img_data.eval())
    plt.show()

    # -----color space transformation-----
    # 将图片的亮度-0.5。
    print("\n brightness - 0.5")
    adjusted = tf.image.adjust_brightness(img_data, -0.5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 将图片的亮度+0.5
    print("\n brightness + 0.5")
    adjusted = tf.image.adjust_brightness(img_data, 0.5)
    plt.imshow(adjusted.eval())
    plt.show()
    
    # 在[-max_delta, max_delta)的范围随机调整图片的亮度。
    print("\n random adjust brightness [-0.5,0.5]")
    adjusted = tf.image.random_brightness(img_data, max_delta=0.5)
    plt.imshow(adjusted.eval())
    plt.show()
     
    # 将图片的对比度-5
    print("\n adjust contrast -5")
    adjusted = tf.image.adjust_contrast(img_data, -5)
    plt.imshow(adjusted.eval())
    plt.show()
    
    # 将图片的对比度+5
    print("\n adjust contrast +5")
    adjusted = tf.image.adjust_contrast(img_data, 5)
    plt.imshow(adjusted.eval())
    plt.show()
    
    # 在[lower, upper]的范围随机调整图的对比度。
    print("\n random adjust contrast [0.1,0.6]")
    adjusted = tf.image.random_contrast(img_data, 0.1, 0.6)
    plt.imshow(adjusted.eval())
    plt.show()
     
    #调整图片的色相
    print("\n adjust hue 0.1")
    adjusted = tf.image.adjust_hue(img_data, 0.1)
    plt.imshow(adjusted.eval())
    plt.show()
     
    # 在[-max_delta, max_delta]的范围随机调整图片的色相。max_delta的取值在[0, 0.5]之间。
    print("\n random adjust hue [0, 0.5]")
    adjusted = tf.image.random_hue(img_data, 0.5)
    plt.imshow(adjusted.eval())
    plt.show()  


    # 将图片的饱和度-5。
    print("\n adjust saturation -5")
    adjusted = tf.image.adjust_saturation(img_data, -5)
    plt.imshow(adjusted.eval())
    plt.show()  

    # 在[lower, upper]的范围随机调整图的饱和度。
    print("\n random adjust saturation [0,5]")
    adjusted = tf.image.random_saturation(img_data, 0, 5)
    plt.imshow(adjusted.eval())
    plt.show()    
    
    # -----position transformation-----
    
    # 上下翻转
    print("\n filp up down")
    flipped1 = tf.image.flip_up_down(img_data)
    plt.imshow(flipped1.eval())
    plt.show()
    
    # 左右翻转
    print("\n filp left right")
    flipped2 = tf.image.flip_left_right(img_data)
    plt.imshow(flipped2.eval())
    plt.show()
    
    #对角线翻转
    print("\n transpose image")
    transposed = tf.image.transpose_image(img_data)
    plt.imshow(transposed.eval())
    plt.show()

    # 填充
    print("\n padding image")
    padding_image = tf.image.resize_image_with_crop_or_pad(img_data,1000,600)
    plt.imshow(padding_image.eval())
    plt.show()
    # -----normalization-----
    
    # 将代表一张图片的三维矩阵中的数字均值变为0，方差变为1。
    adjusted = tf.image.per_image_standardization(img_data)