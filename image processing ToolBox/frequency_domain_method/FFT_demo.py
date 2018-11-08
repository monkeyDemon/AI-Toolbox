# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:35:27 2018

# 傅里叶变换demo1
# 使用离散傅里叶变换对指定图像进行变换
# 随后使用逆变换将图像复原

@author: zyb_as
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont,ImageDraw 

# 读取图像直接读为灰度图像
img = cv2.imread('./test.png',0) 

# FFT变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# 逆变换
fi = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(fi)


"""
可视化傅里叶变换->得到频域图像->傅里叶反变换恢复的过程
"""

#取幅值：将复数变化成实数
#取对数的目的为了将数据变化到0-255
log_amplitude = np.log(np.abs(fshift))      # log变换后的频域幅值

#log_real = np.log(fshift.real)              # log变换后的频域实部
img_back = np.abs(img_back)                 #出来的是复数，无法显示

plt.subplot(131),plt.imshow(img,'gray'),plt.title('original')
plt.subplot(132),plt.imshow(log_amplitude,'gray'),plt.title('log of FFT real')
plt.subplot(133),plt.imshow(img_back,'gray'),plt.title('recovery')


"""
将整个过程的中间步骤保存为图像
此时，与直接调用imshow不同，需要注意处理图像取值范围
"""

s1 = np.abs(fshift)
s1_angle = np.angle(fshift) #取相位
s1_real = s1*np.cos(s1_angle) #取实部
s1_imag = s1*np.sin(s1_angle) #取虚部


cv2.imwrite("./demo1_result/original.jpg", img)
tmpmax = np.max(log_amplitude)
log_amplitude = log_amplitude*255/tmpmax
cv2.imwrite("./demo1_result/FFT_amplitude.jpg", log_amplitude)
cv2.imwrite("./demo1_result/image_back.jpg", img_back)