# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 12:02:27 2018

CNN调参练习version2：

仍然使用一个类似AlexNet和VGG的简单架构
在CNN_v1的基础上，尝试网络深度不同带来的影响
绘制不同深度网络训练集准确率和验证集准确率的学习曲线

@author: zyb_as
"""

# -----------------------------------------------------------
# 预处理
# -----------------------------------------------------------
from read_img_tool import getDataSet       # import 自制数据读取工具函数
from keras.utils import to_categorical
import numpy as np


datasetRootPath = "../../../dataset/trainSetExample"  # 存放各个类别图像数据文件夹的根目录
targetSize = (128, 128, 3)                 # 设置缩放大小（拿到的数据集将会是统一的这个尺寸）
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
#------------Set CallBack(loss history)----------------
#-----------------------------------------------------------------------------------------
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


# -----------------------------------------------------------
# 构建网络
# -----------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from matplotlib import pyplot as plt


def buildCNN(blockNum):
    """
    根据指定块数量构造CNN
    默认blockNum最小为3
    一个block为两个卷积层一个max pooling
    """
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

    for i in range(4,blockNum):
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))

    model.add(Dense(categoryNum, activation='softmax'))

    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.1, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# -----------------------------------------------------------
# 构建并训练不同深度的CNN模型
# -----------------------------------------------------------
blockNumList = range(3, 7)
accessResult = []
val_accessResult = []
batchSize = 32
epochNum = 3

for blockNum in blockNumList:
    print('---------------------')
    print('start training the ' + str(blockNum) + ' block CNN model')
    
    model = buildCNN(blockNum)
    
    history = LossHistory()
    model.fit(x_train, y_train, batch_size=batchSize, epochs=epochNum, 
              validation_data=(x_test,y_test), callbacks=[history])
    
    print('\ntraining finished')
    
    # 评价模型与可视化  
    print('final accuracy on validation set')
    score = model.evaluate(x_test, y_test, batch_size=32)
    print(score)
    
    print('\nvalidation accuracy record')
    print(history.val_acces)
    
    accessResult.append(history.acces)
    val_accessResult.append(history.val_acces)
    print('---------------------\n')


subplt1 = plt.subplot(121)
subplt2 = plt.subplot(122)
subplt1.set_title('training accuracy')
subplt2.set_title('validation accuracy')
subplt1.set_xlabel('epoches')
subplt2.set_xlabel('epoches')

for blockNum in blockNumList:
    s1 = subplt1.plot([x for x in range(1, len(accessResult[blockNum-3]) + 1)],
                       accessResult[blockNum-3],
                       label = str(blockNum) + ' blocks CNN')
plt.legend()
for blockNum in blockNumList:
    s2 = subplt2.plot([x for x in range(1, len(val_accessResult[blockNum-3]) + 1)],
                       val_accessResult[blockNum-3],
                       label = str(blockNum) + ' blocks CNN')
plt.legend()
plt.show()

