# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 12:02:27 2018

CNN调参练习version3：

使用一个类似AlexNet和VGG的简单架构
在CNN_v2的基础上，进行数据增强

@author: zyb_as
"""

# -----------------------------------------------------------
# 基本参数
# -----------------------------------------------------------

trainSetRootPath = '../../../dataset/trainSetExample'     # 训练集根目录路径，该路径下应该分布着存放各个类别图像数据的文件夹
validSetRootPath = '../../../dataset/validSetExample/'    # 验证集根目录路径，该路径下应该分布着存放各个类别图像数据的文件夹
targetSize = (224, 224, 3)                 # 设置缩放大小（拿到的数据集将会是统一的这个尺寸）
categoryNum = 3                            # 你需要人工确认待识别类别的数量
batchSize = 32
epochNum = 100



#-----------------------------------------------------------------------------------------
# image data generator
# 使用 Keras 的 ImageDataGenerator 方法读取数据，同时进行数据增强
#-----------------------------------------------------------------------------------------
from keras.preprocessing.image import ImageDataGenerator

# **根据想要尝试的图像增强方法修改这里**
train_datagen = ImageDataGenerator(rescale=1/255.,
rotation_range = 10,            #数据提升时图片随机转动的角度(整数)
width_shift_range = 0.1,        #图片水平偏移的幅度(图片宽度的某个比例,浮点数)
height_shift_range = 0.1,       #图片竖直偏移的幅度(图片高度的某个比例,浮点数)
shear_range = 0.2,              #剪切强度（逆时针方向的剪切变换角度,浮点数）
zoom_range = 0.2,               #随机缩放的幅度(缩放范围[1 - zoom_range, 1+zoom_range])
horizontal_flip = True,         #进行随机水平翻转
vertical_flip = False           #进行随机竖直翻转
)

val_datagen = ImageDataGenerator(rescale=1/255.) 

train_generator = train_datagen.flow_from_directory( 
trainSetRootPath,          #会扫描该目录下的文件，有几个文件就会默认有几类
target_size=(targetSize[0], targetSize[1]),   #生成的图片像素大小
batch_size=batchSize,      #一次生成的图片数目
class_mode='categorical') 
 
validation_generator = val_datagen.flow_from_directory( 
validSetRootPath, 
target_size=(targetSize[0], targetSize[1]), 
batch_size=batchSize, 
class_mode='categorical')


#-----------------------------------------------------------------------------------------
# Set CallBack(loss history)
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
        
history = LossHistory()

# -----------------------------------------------------------
# 构建网络
# -----------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from matplotlib import pyplot as plt


def buildCNN():   
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
    
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))

    model.add(Dense(categoryNum, activation='softmax'))

    # choose a optimizer
    #sgd = SGD(lr=0.0000001, decay=1e-6, momentum=0.1, nesterov=False)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer=adam, metrics=['accuracy'])
    return model



# -----------------------------------------------------------
# 构建并训练CNN模型
# -----------------------------------------------------------
print('---------------------')
print('start training model with the image enhancement strategy\n')

model = buildCNN()

#训练集样本总数
train_sample_count = len(train_generator.filenames)
#测试集样本总数
val_sample_count = len(validation_generator.filenames)

model.fit_generator( 
    train_generator,
    steps_per_epoch= int(train_sample_count/batchSize) + 1, # steps_per_epoch定义多少batch算作完成一次epoch 
    epochs=epochNum,
    validation_data=validation_generator, 
    validation_steps= int(val_sample_count/batchSize) + 1, # batch_size, 
    callbacks=[history])

print('\ntraining finished')
print('---------------------\n')

# 评价模型与可视化 
print('best accuracy on training set:' + str(max(history.acces))) 
print('best accuracy on validation set:' + str(max(history.val_acces)))

print('\nvalidation accuracy record on each epoches:')
print(history.val_acces)

plt.title('Result Analysis')
plt.plot([x for x in range(1, len(history.acces) + 1)], history.acces, color='green', label='training accuracy')
plt.plot([x for x in range(1, len(history.val_acces) + 1)], history.val_acces, color='skyblue', label='validation accuracy')
plt.legend() # 显示图例
plt.xlabel('epoches')
plt.ylabel('accuracy')
plt.show()
    


