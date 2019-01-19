#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
ResNet50 models base on Keras.

This procedure is suitable for the situation that you need to quickly train 
a ResNet50 network as a classification.

The program will do fine turning on the weights pretrained by imageNet.
If you want to start training from scratch on the specified data set,
please use resNet50.py

# Reference paper
- [Deep Residual Learning for Image Recognition] (https://arxiv.org/abs/1512.03385) 

# weights download address for classic CNN: 'https://github.com/fchollet/deep-learning-models/releases/'

@author: zyb_as
"""

import os
import sys

import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten 

from keras.optimizers import SGD
from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator 

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

# weight_load_path   load the weights pretrained on imageNet
                     if the specified file doesn't exist, the program will start download weights from Internet
# weight_save_path   the path to save the weights after training
# train_set_path     the root path of the training set. Each category should correspond to a folder
# valid_set_path     the root path of the validation set. Each category should correspond to a folder
# record_save_path   the path to save the training record file
# category_num       the category num of the classification problem
"""
# TODO: set basic configuration parameters
weight_load_path = './weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
weight_save_path = './weights/resnet50_weights_tf_transfor.h5'
train_set_path = 'train_root_path/'
valid_set_path = 'validation_root_path/'
record_save_path = './records'
category_num = 2
batch_size = 32



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
early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=1, mode='auto') 

# set model checkpoint callback (model weights will auto save in weight_save_path)
checkpoint = ModelCheckpoint(weight_save_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)

# monitor a learning indicator(reduce learning rate when learning effect is stagnant)
reduceLRcallback = ReduceLROnPlateau(monitor='val_acc', factor=0.7, patience=5,
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
train_set_path,           
target_size=(224, 224),   
batch_size=batch_size,    
class_mode='categorical') 
 
validation_generator = val_datagen.flow_from_directory( 
valid_set_path, 
target_size=(224, 224), 
batch_size=batch_size, 
class_mode='categorical') 



#-----------------------------------------------------------------------------------------
#--------------------------------continue training----------------------------------------
#-----------------------------------------------------------------------------------------
# load specified weights to continue training(if exist)
if os.path.exists(weight_save_path):
    model = load_model(weight_save_path)
    
    # TODO: choose training parameters
    epochs = 200

    # TODO: choose a optimizer
    optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
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
    sys.exit()



#-----------------------------------------------------------------------------------------
#-------------------------------------build model-----------------------------------------
#-----------------------------------------------------------------------------------------

def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    return model

# load preTrained ResNet50 model without top fully connection layer
if os.path.exists(weight_load_path) == False:
	weight_load_path = 'imagenet'
baseModel = ResNet50(weights = weight_load_path, include_top=False, pooling=None, input_shape=(224, 224, 3))

# add new layer
model = add_new_last_layer(baseModel, category_num)

# check model
model.summary()



#-----------------------------------------------------------------------------------------
#---------------------------------transfer learning---------------------------------------
#-----------------------------------------------------------------------------------------
# TODO: choose training parameters
epoch_num = 200

# TODO: choose a optimizer
optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#lrate = 0.01 
#decay = lrate/epochs 
#optimizer = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) 
    
# compile the model (should be done after recover layers to trainable)    
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# number of training set
train_sample_count = len(train_generator.filenames)
# number of validation set
val_sample_count = len(validation_generator.filenames)

print(train_sample_count, val_sample_count)

fitted_model = model.fit_generator( 
train_generator, 
steps_per_epoch = int(train_sample_count/batch_size), 
epochs = epoch_num,
validation_data = validation_generator, 
validation_steps = int(val_sample_count/batch_size), 
callbacks = [TensorBoard(log_dir=record_save_path), early_stopping, history, checkpoint, reduceLRcallback]
)
