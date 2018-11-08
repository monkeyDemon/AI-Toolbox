# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:54:51 2018

# 傅里叶变换demo2
# 使用傅里叶变换及反变换对图像在频域进行肉眼不可见的加密
# 随后对加密后图像再次进行傅里叶变换，严重加密的有效性

@author: zyb_as
"""

import cv2
import numpy as np
from PIL import Image, ImageFont,ImageDraw 


def drawText(arr, pwd = 'test', size = 30):
    """
    向图像中绘制字母或数字
    这里用于向频域图像中写入字母密钥
    arr ndarray 代表频域图像的数组
    pwd string  字符串私钥
    size int    私钥字母大小
    """
    font = ImageFont.truetype('simhei.ttf', size)  # 定义字体
    im = Image.fromarray(arr)                    # 转换为PIL.Image
    draw = ImageDraw.Draw(im)                    # 生成画板类
    
    # 定义坐标（右下角）
    shape = arr.shape
    x,y=(shape[0] - size, shape[1] - len(pwd)* size) # 初始左上角坐标

    # 将私钥写入频域图像
    draw.text([y,x], pwd, font = font, fill = 'white')
    #im.show()    
    return np.array(im)


def drawRectangle(im, size = 10):
    """
    向图像中绘制矩形块进行干扰
    im          输入图像
    return      干扰后图像
    """
    draw = ImageDraw.Draw(im)    # 生成画板类
    #绘制矩形（这里仅为测试，绘制位置没有进行自适应调整）
    draw.rectangle((800, 400, 900, 430), 'black', 'black')
    #im.show()
    return im


def saveByPIL(arr, path):
    """
    将 ndarray 通过 PIL Image 保存
    """
    img = Image.fromarray(arr)
    img = img.convert("L")
    img.save(path, quality = 100)



# 读取图像直接读为灰度图像
img = cv2.imread('./test.png',cv2.IMREAD_GRAYSCALE)

# FFT变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# --------------------------加密部分start--------------------------
# 获取频域的具体信息（幅值、相位、实部、虚部）
s1 = np.abs(fshift)           #取幅值
s1_angle = np.angle(fshift)   #取相位
s1_real = s1*np.cos(s1_angle) #取实部
s1_imag = s1*np.sin(s1_angle) #取虚部

# 在FFT变换后的频域图像内加入数字水印
# 这里以在频域实部中加入水印为例
s1_real = drawText(s1_real, pwd = 'test', size = 70)

# 重新生成一个频域复数图像
s2 = np.zeros(img.shape,dtype=complex)
s2.real = np.array(s1_real) 
s2.imag = np.array(s1_imag)

# --------------------------加密部分end--------------------------


# 对加入水印的频域进行逆变换
fi = np.fft.ifftshift(s2)
img_back = np.fft.ifft2(fi)


# ----------------------将结果进行保存----------------------------
# **注** 进行本实验，一定要保存为png格式
# 因为png为无损压缩，jpg为有损压缩 
# 图像保存再读取中发生的微弱变化会完全影响FFT的频域结果
# 此外，图像保存过程中小数部分会被截断，因此需要单独保存

# 由于我们对频域进行了处理，因此逆变换后的img_back的虚部不再为0
# 我们可以将逆变换后的img_back的实部对外发布（其与原图的差别微弱，不易发现）
#cv2.imwrite("./demo2_result/image_back_real.png", img_back.real)
saveByPIL(img_back.real, "./demo2_result/image_back_real.png")

# 保存 img_back的实部 与 原图 的 差
# 为了证明与原图存在差别，将差放大
#cv2.imwrite("./demo2_result/diff_original_backreal.png", abs(img - img_back.real)*10)
saveByPIL(abs(img - img_back.real)*100, "./demo2_result/diff_original_backreal.png")

# 将作为私钥保存的逆变换后的img_back的虚部图像放大并保存
# 为了证明虚部真实存在，这里将虚部放大1000倍
#cv2.imwrite("./demo2_result/imageback_imag_zoomin.png", img_back.imag * 1000)
saveByPIL(img_back.imag * 1000, "./demo2_result/imageback_imag_zoomin.png")

# 将逆变换后的img_back的虚部单独保存作为私钥
# 同时由于保存为图像格式后将损失小数信息，实部的小数部分也要保存下来
img_back_truncated = img_back.real - np.floor(img_back.real)
img_back_imag = img_back.imag
np.save("./demo2_result/pwd.npy", [img_back_imag, img_back_truncated])


# ----------------------模拟图像解密环节----------------------------

# 读取修改后的图像（实部）
#img_modify = cv2.imread('./demo2_result/image_back_real.png',cv2.IMREAD_GRAYSCALE) 
img_modify = Image.open('./demo2_result/image_back_real.png')

# 读取密钥
[img_back_imag, img_back_truncated] = np.load("./demo2_result/pwd.npy")

# 恢复完整图像
reset_image = np.zeros(img.shape, dtype=complex)
reset_image.real = np.array(img_modify) + img_back_truncated
reset_image.imag = np.array(img_back_imag)

# 傅里叶变换，拿到频域实部图像并展示
# 观察密钥是否存在
f = np.fft.fft2(reset_image)
fshift = np.fft.fftshift(f)
reset_real = Image.fromarray(fshift.real)
reset_real.show()

# 将复现的频域结果保存
saveByPIL(np.array(reset_real), "./demo2_result/reset_real.png")


# ----------------再次模拟图像解密环节，对目标图像进行干扰----------------------------

# 读取修改后的图像（实部）
#img_modify = cv2.imread('./demo2_result/image_back_real.png',cv2.IMREAD_GRAYSCALE) 
img_modify = Image.open('./demo2_result/image_back_real.png')
img_modify = drawRectangle(img_modify)


# 读取密钥
[img_back_imag, img_back_truncated] = np.load("./demo2_result/pwd.npy")

# 恢复完整图像
reset_disturb = np.zeros(img.shape, dtype=complex)
reset_disturb.real = np.array(img_modify) + img_back_truncated
reset_disturb.imag = np.array(img_back_imag)

# 傅里叶变换，拿到频域实部图像并展示
# 观察密钥是否存在
f = np.fft.fft2(reset_disturb)
fshift = np.fft.fftshift(f)
reset_real_disturb = Image.fromarray(fshift.real)
reset_real_disturb.show()

# 将干扰的图像保存
saveByPIL(np.array(img_modify), "./demo2_result/original_disturb.png")
# 将复现的频域结果保存
saveByPIL(np.array(reset_real_disturb), "./demo2_result/reset_real_disturb.png")