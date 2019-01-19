# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 12:02:27 2018

CNN调参练习version1：

使用一个类似AlexNet和VGG的简单架构
不加任何预处理和调参技巧

@author: zyb_as
"""

# -----------------------------------------------------------
# 预处理
# -----------------------------------------------------------
from read_img_tool import getDataSet       # import 自制数据读取工具函数
from keras.utils import to_categorical
import numpy as np


datasetRootPath = "../../../dataset/trainSetExample"  # 存放各个类别图像数据文件夹的根目录
targetSize = (224, 224, 3)                    # 设置缩放大小（拿到的数据集将会是统一的这个尺寸）
categoryNum = 3                            # 你需要人工确认待识别类别的数量
x_dataset, y_dataset = getDataSet(datasetRootPath, targetSize, verbose=0)

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


#-----------------------------------------------------------------------------------------
#------------Set CallBack(loss history, early stopiing)----------------
#-----------------------------------------------------------------------------------------
import keras
from keras.callbacks import Callback

# 记录训练过程
# Callback 用于记录每个epoch的 loss 和 accuracy
class LossHistory(Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.acces = []
        self.val_losses = []
        self.val_acces = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acces.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acces.append(logs.get('val_acc'))

history = LossHistory()

# 在loss不降时停止迭代，减少等待时间，防止过拟合
## Callback for early stopping the training
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
min_delta=0,
patience=5,
verbose=1, mode='auto')


# -----------------------------------------------------------
# 构建模型
# -----------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from matplotlib import pyplot as plt

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape = targetSize))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))

model.add(Dense(categoryNum, activation='softmax'))

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.1, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# -----------------------------------------------------------
# 训练模型
# -----------------------------------------------------------

print('---------------------')
print('start training...')

batchSize = 32
epochNum = 20
model.fit(x_train, y_train, batch_size=batchSize, epochs=epochNum,
          validation_data=(x_test,y_test), callbacks=[early_stopping,history])

print('training finished')
print('---------------------\n')


# -----------------------------------------------------------
# 评价模型与可视化
# -----------------------------------------------------------

print('final accuracy on validation set')
score = model.evaluate(x_test, y_test, batch_size=32)
print(score)

print('\nvalidation accuracy record')
print(history.val_acces)

plt.title('Result Analysis')
plt.plot([x for x in range(1, len(history.acces) + 1)], history.acces, color='green', label='training accuracy')
plt.plot([x for x in range(1, len(history.val_acces) + 1)], history.val_acces, color='skyblue', label='validation accuracy')
plt.legend() # 显示图例
plt.xlabel('epoches')
plt.ylabel('accuracy')
plt.show()
