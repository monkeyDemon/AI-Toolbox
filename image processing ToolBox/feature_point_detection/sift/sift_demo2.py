# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:18:14 2018

SIFT特征点检测demo2
准备两张人脸图片，分别检测特征点并尝试匹配
若匹配成功，则可一定概率认为提供的两张人脸图片是同一个人

@author: zyb_as
"""

import cv2
import numpy as np
from PIL import Image


# 1) 以灰度图的形式读入图片

# 人脸模板
psd_img_1 = cv2.imread('jgz2.jpg', cv2.IMREAD_GRAYSCALE)
# 待匹配图片
psd_img_2 = cv2.imread('jgz1.jpg', cv2.IMREAD_GRAYSCALE)

image = Image.fromarray(psd_img_2) # 将图片缩放到统一尺寸便于显示
image = image.resize((480, 550))
psd_img_2 = np.array(image)


# 2) SIFT特征计算
sift = cv2.xfeatures2d.SIFT_create()

psd_kp1, psd_des1 = sift.detectAndCompute(psd_img_1, None)
psd_kp2, psd_des2 = sift.detectAndCompute(psd_img_2, None)

# 3) Flann特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(psd_des1, psd_des2, k=2)
goodMatch = []
for m, n in matches:
    # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
    if m.distance < 0.75*n.distance:
        goodMatch.append(m)
# 增加一个维度
goodMatch = np.expand_dims(goodMatch, 1)
print(goodMatch[:20])

#img_out = cv2.drawMatchesKnn(psd_img_1, psd_kp1, psd_img_2, psd_kp2, goodMatch[:15], None, flags=2)
img_out = cv2.drawMatchesKnn(psd_img_1, psd_kp1, psd_img_2, psd_kp2, goodMatch, None, flags=2)


cv2.imshow('image', img_out)#展示图片
cv2.waitKey(0)#等待按键按下
cv2.destroyAllWindows()#清除所有窗口
