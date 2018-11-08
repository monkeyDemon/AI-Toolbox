# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:34:41 2018

SIFT特征点检测demo1
计算SIFT特征点并绘制于图像中

@author: zyb_as
"""

import cv2

# 1) 以灰度图的形式读入图片

# 人脸模板
psd_img_1 = cv2.imread('jgz1.jpg', cv2.IMREAD_GRAYSCALE)


# 2) SIFT特征计算
sift = cv2.xfeatures2d.SIFT_create()

psd_kp1, psd_des1 = sift.detectAndCompute(psd_img_1, None)


# 3) 绘制特征点
#画出特征点
im_keypoint = cv2.drawKeypoints(psd_img_1, psd_kp1, psd_img_1, 
                         flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 

cv2.imshow('image', im_keypoint)#展示图片
cv2.waitKey(0)#等待按键按下
cv2.destroyAllWindows()#清除所有窗口