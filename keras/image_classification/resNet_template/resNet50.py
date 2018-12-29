# -*- coding: utf-8 -*-
"""

ResNet50 models base on Keras.

This procedure is suitable for the situation that you need to quickly train 
a ResNet50 network as a classification.

The program will start training from scratch on the specified data set
If you want to do fine turning on the weights pretrained by imageNet,
please use resNet50_transfer_learning.py

In the program, we manually construct the ResNet50 network structure,
this allow us to modify the structure of CNN more flexibly

# Reference paper
- [Deep Residual Learning for Image Recognition] (https://arxiv.org/abs/1512.03385) 

Adapted from code contributed by BigMoyan.
"""

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
 
import os

import keras 
from keras.models import Model

from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Add
from keras.layers import Dense
from keras.layers import Flatten 
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D

from keras import initializers
from keras.constraints import max_norm
from keras.preprocessing.image import ImageDataGenerator 

from keras.optimizers import SGD 
from keras.optimizers import Adam

from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint 
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import Callback 


#-----------------------------------------------------------------------------------------
#--------------------------------parameter defination-------------------------------------
#-----------------------------------------------------------------------------------------

"""
# Before applying this demo to your new classification problem
# please check the settings of the following parameters.
# Besides, don't forget to check the hyperparameters

# weight_load_path   load the specified weights to continue training
		     if the specified file doesn't exist, the program will start training from the beginning
# weight_save_path   the path to save the weights after training
                     you can set the same with weight_load_path to cover the weight file
# train_set_path     the root path of the training set. Each category should correspond to a folder
# valid_set_path     the root path of the validation set. Each category should correspond to a folder
# record_save_path   the path to save the training record file
# category_num       the category num of the classification problem
"""

# TODO: set basic configuration parameters
weight_load_path = './weights/resnet50_weights_tf.h5' 
weight_save_path = './weights/resnet50_weights_tf.h5'  
train_set_path = 'train_root_path/'
valid_set_path = 'validation_root_path/'
record_save_path = './records'
category_num = 2
batch_size = 32  


#-----------------------------------------------------------------------------------------
#--------------------------------model defination-----------------------------------------
#-----------------------------------------------------------------------------------------
 
def Conv2d_BN(x, nb_filter, kernel_size, strides=(1,1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name,
               kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
               kernel_constraint=max_norm(2.))(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x

def Conv_Block(inpt, nb_filter, kernel_size, strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter[0], kernel_size=(1,1), strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3,3), padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1,1), padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
        x = Add()([x,shortcut])
        return x
    else:
        x = Add()([x,inpt])
        return x

def ResNet50():
    inpt = Input(shape=(224,224,3))
    x = ZeroPadding2D((3,3))(inpt)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7,7), strides=(2,2), padding='valid')
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    
    x = Conv_Block(x, nb_filter=[64,64,256], kernel_size=(3,3), strides=(1,1), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[64,64,256], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[64,64,256], kernel_size=(3,3))
    
    x = Conv_Block(x, nb_filter=[128,128,512], kernel_size=(3,3), strides=(2,2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[128,128,512], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[128,128,512], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[128,128,512], kernel_size=(3,3))
    
    x = Conv_Block(x, nb_filter=[256,256,1024], kernel_size=(3,3), strides=(2,2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[256,256,1024], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[256,256,1024], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[256,256,1024], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[256,256,1024], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[256,256,1024],kernel_size=(3,3))
    
    x = Conv_Block(x, nb_filter=[512,512,2048], kernel_size=(3,3), strides=(2,2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[512,512,2048], kernel_size=(3,3))
    x = Conv_Block(x, nb_filter=[512,512,2048], kernel_size=(3,3))
    x = AveragePooling2D(pool_size=(7,7))(x)
    x = Flatten()(x)
    x = Dense(category_num, activation='softmax',
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
            kernel_constraint=max_norm(3.))(x)
    
    model = Model(inputs=inpt,outputs=x)
    return model



#-----------------------------------------------------------------------------------------
#--Set CallBack(loss history, early stopiing, model check point, reduce learning rate)----
#-----------------------------------------------------------------------------------------

class LossHistory(Callback): 
    
    def on_train_begin(self, logs={}): 
        self.losses = [] 
        self.val_losses = [] 
        self.acc = []
        self.val_acc = []
 
    def on_epoch_end(self, batch, logs={}): 
        self.losses.append(logs.get('loss')) 
        self.val_losses.append(logs.get('val_loss')) 
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        
# record loss history callback
history = LossHistory() 

# Callback for early stopping the training
early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=14, verbose=1, mode='auto') 

# set model checkpoint callback (model weights will auto save in weight_save_path)
checkpoint = ModelCheckpoint(weight_save_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)  

# monitor a learning indicator(reduce learning rate when learning effect is stagnant)
reduceLRcallback = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, 
verbose=1, mode='auto', cooldown=0, min_lr=0)



#-----------------------------------------------------------------------------------------
#---------------------------image data generator------------------------------------------
#-----------------------------------------------------------------------------------------

# TODO: try the data augmentation method you want
train_datagen = ImageDataGenerator(rescale=1/255.,
rotation_range = 45,            
width_shift_range = 0.2,        # degree of horizontal offset(a ratio relative to image width)
height_shift_range = 0.2,       # degree of vertical offset(a ratio relatice to image height)
shear_range = 0.2,              # the range of shear transformation(a ratio in 0 ~ 1)
zoom_range = 0.25,              # degree of random zoom(the zoom range will be [1 - zoom_range, 1 + zoom_range])
horizontal_flip = True,         # whether to perform horizontal flip
vertical_flip = True,           # whether to perform vertical flip 
fill_mode = 'nearest'           # mode list: nearest, constant, reflect, wrap
)

val_datagen = ImageDataGenerator(rescale=1/255.) 

train_generator = train_datagen.flow_from_directory( 
train_set_path, # the path of traning set(each directory should correspond to one category)
target_size=(224, 224),         
batch_size=batch_size,         
class_mode='categorical') 
 
validation_generator = val_datagen.flow_from_directory( 
valid_set_path, 
target_size=(224, 224), 
batch_size=batch_size, 
class_mode='categorical') 



#-----------------------------------------------------------------------------------------
#---------------------------------training model------------------------------------------
#-----------------------------------------------------------------------------------------

# build basic model
model = ResNet50()

# load specified weights to continue training(if exist)
if os.path.exists(weight_load_path):
    model.load_weights(weight_load_path)

# TODO: choose training parameters
epochs = 200

# TODO: choose a optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#lrate = 0.01 
#decay = lrate/epochs 
#optimizer = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

# compile the model
model.compile(optimizer = optimizer, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# number of traning set
train_sample_count = len(train_generator.filenames)
# number of validation set
val_sample_count = len(validation_generator.filenames)

print(train_sample_count, val_sample_count)

# start training(the best model will be automatically save)
model.fit_generator( 
train_generator,
steps_per_epoch= int(train_sample_count/batch_size) + 1, # steps_per_epoch defines how many batch in one epoch
epochs=epochs,
validation_data=validation_generator, 
validation_steps= int(val_sample_count/batch_size) + 1, 
callbacks=[TensorBoard(log_dir=record_save_path), early_stopping, history, checkpoint, reduceLRcallback]
)
