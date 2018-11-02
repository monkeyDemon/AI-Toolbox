# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:59:02 2018

@author: zyb_as
"""

# -----------------------------------------------------------
# 这段代码演示如何使用 read_img_demp.py 文件读取硬盘上的图像数据
# -----------------------------------------------------------
from read_img_tool import getDataSet

datasetRootPath = "../../dataset/trainSet"  # 存放各个类别图像数据文件夹的根目录
aimSize = (100, 100, 3)       # 设置缩放大小（拿到的数据集将会是统一的这个尺寸）
categoryNum = 2               # 你需要人工确认待识别类别的数量
x_dataset, y_dataset = getDataSet(datasetRootPath, aimSize)



# -----------------------------------------------------------
# 这段代码使用读取的数据，完成一个简单的2分类demo
# 主要观察如何使用拿到的数据
# -----------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np

"""
categoryNum = 2
x_dataset = np.random.random((100, 100, 100, 3))
y_dataset = np.random.randint(2, size=(100, 1))
"""

# disrupt data set 随机打乱数据集
totalNum = x_dataset.shape[0]
permutation = np.random.permutation(totalNum)
x_dataset = x_dataset[permutation]
y_dataset = y_dataset[permutation]

# dividing data
trainsetNum = int(totalNum * 0.7)
x_train = x_dataset[:trainsetNum]
x_test = x_dataset[trainsetNum:]
y_train = y_dataset[:trainsetNum]
y_train = to_categorical(y_train, num_classes = categoryNum)
y_test = y_dataset[trainsetNum:]
y_test = to_categorical(y_test, num_classes = categoryNum)


model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape = aimSize))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(categoryNum, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.1, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
