# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:06:18 2018

SIFT特征点检测demo3
准备两张人脸图片，一张作为模板，一张作为待检测图像
分别检测特征点并尝试匹配
使用Ransac算法剔除误匹配点
最终显示匹配结果
若匹配成功，则可一定概率认为提供的两张人脸图片是同一个人

@author: zyb_as
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

template_image = 'jgz2.jpg'          # source image(use as a face template)
detect_image = 'jgz_bqb3.png'       # Image to be detected
min_goodmatch_count = 10             # min number of good matches(best match distance < threshold * second match distance)
min_rightmatch_count = 7             # min number of right matches(after remove false match by ransac)
match_threshold = 0.8                # the threshold to define a good match(best match distance should < threshold * second match distance)

src_img = cv2.imread(template_image,0)
dst_img = cv2.imread(detect_image,0)        

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(src_img,None)
kp2, des2 = sift.detectAndCompute(dst_img,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < match_threshold * n.distance:
        good.append(m)
        
if len(good)>min_goodmatch_count:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # 计算单应性矩阵，同时使用ransac剔除误识别点
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    rightmatch_count = sum(matchesMask)

    # 找到模板图像在目标图像中的映射位置
    h,w = src_img.shape
    # 获取原图像边缘四个角对应的坐标点
    src_corner_pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # 利用单应性矩阵计算四个点在目标图像中的对应位置
    dst_map_pts = cv2.perspectiveTransform(src_corner_pts,M)

    # 将模板图像在目标图像中的位置映射以白色方框的形式绘制出来
    dst_img = cv2.polylines(dst_img,[np.int32(dst_map_pts)],True,255,3, cv2.LINE_AA)
else:
    print("False! Not enough matches are found - %d/%d" % (len(good),min_goodmatch_count))
    matchesMask = None


if rightmatch_count < min_rightmatch_count:
    print("False! Not enough right matches are found after remove false match by ransac- %d/%d" 
                % (rightmatch_count, min_rightmatch_count))
else:
    print("success match! Has %d right matches" % rightmatch_count)    


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(src_img,kp1,dst_img,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()
