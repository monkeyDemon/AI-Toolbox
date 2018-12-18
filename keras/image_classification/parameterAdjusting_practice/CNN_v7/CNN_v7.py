# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:05:19 2018

CNN调参练习version7：

到目前,在CNN_v1 - CNN_v6的过程中
我们使用各种方法,完成了一个CNN网络从头到尾的训练和优化

现在,我们尝试另一种思路：迁移学习
使用在更大的数据集ImageNet上训练的成熟网络框架VGG16网络作为预训练权重进行迁移学习

@author: zyb_as
"""

import keras
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten 
from keras.layers import BatchNormalization, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator 
#from keras.callbacks import ModelCheckpoint 
from keras.callbacks import Callback 
from keras.models import load_model
from matplotlib import pyplot as plt
import os
import gc


"""
# pretrainWeightfile    预训练权重文件路径及文件名（不存在将会自动下载）
# weightfileSavePath    最终网络权重训练结果的存储路径及文件名
# trainSetRootPath      训练集根目录路径，该路径下应该分布着存放各个类别图像数据的文件夹
# validSetRootPath      验证集根目录路径，该路径下应该分布着存放各个类别图像数据的文件夹
# trainingRecordPath    记录训练过程文件的存放路径（用于TensorBoard可视化训练过程）
# targetSize            设置样本尺寸（注意要与迁移的模型一致）
# categoryNum           多分类问题的类别数
# epochNum_toplayer     全连接层微调过程的迭代次数
# epochNum_totallayer   整体微调过程的迭代次数
# traning_state         *指定当前训练阶段*,transfer learning or fine tuning or complete training

# 各种经典网络的权重下载地址: 'https://github.com/fchollet/deep-learning-models/releases/'
"""

pretrainWeightfile = './vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
weightfileSavePath = 'vgg16_fineturning_weights.h5'
trainSetRootPath = '../../../dataset/trainSetExample'
validSetRootPath = '../../../dataset/validSetExample'
trainingRecordPath = './trainingRecord/'
targetSize = (224, 224, 3)
categoryNum = 3
batch_size = 32
epochNum_toplayer = 8
epochNum_totallayer = 20
#lrate = 0.01
#traning_state = 'transfer learning'
#traning_state = 'fine tuning'
traning_state = 'complete training'

#-----------------------------------------------------------------------------------------
#------------Set CallBack(loss history, early stopiing, model check point)----------------
#-----------------------------------------------------------------------------------------

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


"""
## Callback for early stopping the training 
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
min_delta=0, 
patience=15, 
verbose=1, mode='auto') 
"""

#-----------------------------------------------------------------------------------------
#---------------------------image data generator------------------------------------------
#-----------------------------------------------------------------------------------------

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
batch_size=batch_size,    #生成的图片数目
class_mode='categorical') 
 
validation_generator = val_datagen.flow_from_directory( 
validSetRootPath, 
target_size=(targetSize[0], targetSize[1]), 
batch_size=batch_size, 
class_mode='categorical') 


#-----------------------------------------------------------------------------------------
#-------------------------------------build model-----------------------------------------
#-----------------------------------------------------------------------------------------

def add_new_last_layer(base_model,nb_classes):
    x = base_model.output
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


#-----------------------------------------------------------------------------------------
#-------------------------------------training-----------------------------------------
#-----------------------------------------------------------------------------------------

if traning_state == 'transfer learning':
    """
    ---------------------------------transfer learning---------------------------------------
    """
    # load preTrain weights
    # 判断指定的预训练权重是否存在，存在则加载权重，否则从默认网址下载
    if os.path.exists(pretrainWeightfile) == False:
        pretrainWeightfile = 'imagenet'
    
    basemodel = VGG16(weights=pretrainWeightfile, include_top=False, 
	              pooling=None, input_shape=targetSize)
    # add new layer
    model = add_new_last_layer(basemodel, categoryNum)

    # first: freeze all convolutional pretrained layers(only train the top layers, which were randomly initialized)
    for layer in basemodel.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable) 
    # ！！！！！！！！！！choose a optimizer**********************
    #sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.1, nesterov=False)
    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    train_sample_count = len(train_generator.filenames)     #训练集样本总数
    val_sample_count = len(validation_generator.filenames)  #测试集样本总数
    
    # train the model on the new data for a few epochs
    fitted_model = model.fit_generator( 
    train_generator, 
    steps_per_epoch= int(train_sample_count/batch_size) + 1, #多少步完成一次epoch 
    epochs=epochNum_toplayer,
    validation_data=validation_generator, 
    validation_steps= int(val_sample_count/batch_size) + 1, 
    callbacks=[TensorBoard(log_dir=trainingRecordPath),history])
    
    model.save(weightfileSavePath)
    
    # 评价模型与可视化 
    print('best accuracy on training set:' + str(max(history.acces))) 
    print('best accuracy on validation set:' + str(max(history.val_acces)))
    
    plt.title('Learning Curve')
    plt.plot([x for x in range(1, len(history.acces) + 1)], history.acces, color='green', label='training accuracy')
    plt.plot([x for x in range(1, len(history.val_acces) + 1)], history.val_acces, color='skyblue', label='validation accuracy')
    plt.legend() # 显示图例
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.show()
elif traning_state == 'fine tuning':
    """
    -------------------------fine tuning-------------------------------
    """

    model = load_model(weightfileSavePath)

    for layer in model.layers:
        if layer.trainable == False:
            layer.trainable = True

    # compile the model (should be done *after* resetting layers to trainable)
    #ldecay = lrate/200 
    #sgd = SGD(lr=lrate, momentum=0.9, decay=ldecay, nesterov=False)     
    adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)   
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    
    train_sample_count = len(train_generator.filenames)     #训练集样本总数
    val_sample_count = len(validation_generator.filenames)  #测试集样本总数

    fitted_model = model.fit_generator( 
    train_generator, 
    steps_per_epoch= int(train_sample_count/batch_size) + 1, #多少步完成一次epoch 
    epochs=epochNum_totallayer,
    validation_data=validation_generator, 
    validation_steps= int(val_sample_count/batch_size) + 1, # batch_size, 
    callbacks=[TensorBoard(log_dir=trainingRecordPath),history])
    
    model.save(weightfileSavePath)
    
    # 评价模型与可视化 
    print('best accuracy on training set:' + str(max(history.acces))) 
    print('best accuracy on validation set:' + str(max(history.val_acces)))
    
    plt.title('Learning Curve')
    plt.plot([x for x in range(1, len(history.acces) + 1)], history.acces, color='green', label='training accuracy')
    plt.plot([x for x in range(1, len(history.val_acces) + 1)], history.val_acces, color='skyblue', label='validation accuracy')
    plt.legend() # 显示图例
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.show()
elif traning_state == 'complete training':
    """
    ---------------------------------complete training---------------------------------------
    """

    # load preTrain weights
    # 判断指定的预训练权重是否存在，存在则加载权重，否则从默认网址下载
    if os.path.exists(pretrainWeightfile) == False:
        pretrainWeightfile = 'imagenet'
    
    basemodel = VGG16(weights=pretrainWeightfile, include_top=False, 
	              pooling=None, input_shape=targetSize)
    # add new layer
    model = add_new_last_layer(basemodel, categoryNum)

    # first: freeze all convolutional pretrained layers(only train the top layers, which were randomly initialized)
    for layer in basemodel.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable) 
    # ！！！！！！！！！！choose a optimizer**********************
    #sgd = SGD(lr=0.0000001, decay=1e-6, momentum=0.1, nesterov=False)
    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam,   
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    
    train_sample_count = len(train_generator.filenames)     #训练集样本总数
    val_sample_count = len(validation_generator.filenames)  #测试集样本总数
    
    # train the model on the new data for a few epochs
    model.fit_generator( 
    train_generator, 
    steps_per_epoch= int(train_sample_count/batch_size) + 1, #多少步完成一次epoch 
    epochs=epochNum_toplayer,
    validation_data=validation_generator, 
    validation_steps= int(val_sample_count/batch_size) + 1, 
    callbacks=[TensorBoard(log_dir=trainingRecordPath),history])
    
    model.save(weightfileSavePath)
    train_learn_record = history.acces
    valid_learn_record = history.val_acces
    
    # fine tuning
    # 回收model占用的内存空间
    del model, basemodel
    gc.collect()  

    model = load_model(weightfileSavePath)

    for layer in model.layers:
        if layer.trainable == False:
            layer.trainable = True
        
    # compile the model (should be done *after* resetting layers to trainable)
    #ldecay = lrate/200 
    #sgd = SGD(lr=lrate, momentum=0.9, decay=ldecay, nesterov=False)  
    adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)   

    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    
    model.fit_generator( 
    train_generator, 
    steps_per_epoch= int(train_sample_count/batch_size) + 1, #多少步完成一次epoch 
    epochs=epochNum_totallayer,
    validation_data=validation_generator, 
    validation_steps= int(val_sample_count/batch_size) + 1, 
    callbacks=[TensorBoard(log_dir=trainingRecordPath),history])
    
    model.save(weightfileSavePath)
    
    train_learn_record.append(history.acces)
    valid_learn_record.append(history.val_acces)
    
    #评价模型与可视化
    print('best accuracy on training set:' + str(max(train_learn_record[1]))) 
    print('best accuracy on validation set:' + str(max(valid_learn_record[1])))
    
    plt.title('Learning Curve')
    plt.plot([x for x in range(1, len(train_learn_record) + 1)], train_learn_record, color='green', label='training accuracy')
    plt.plot([x for x in range(1, len(valid_learn_record) + 1)], valid_learn_record, color='skyblue', label='validation accuracy')
    plt.legend() # 显示图例
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.show()
else:
    print('traning_state error!')
