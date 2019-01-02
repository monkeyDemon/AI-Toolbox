# -*- coding: utf-8 -*-
"""

DenseNet models for Keras.

This procedure is suitable for the situation that you need to quickly train 
a DenseNet network as a classification.

The program will start training from scratch on the specified data set
If you want to do fine turning on the weights pretrained by imageNet,
please use denseNet_transfer_learning.py

In the program, we manually construct the DenseNet network structure,
this allow us to modify the structure of CNN more flexibly

# Reference paper
- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

# Reference implementation
- [Torch DenseNets]
  (https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua)
- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py)
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import os 

import keras 
from keras import backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten 
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D

from keras import initializers
from keras import regularizers
from keras.constraints import max_norm
#from keras.utils.data_utils import get_file
#from keras.engine.topology import get_source_inputs
#from keras.applications import imagenet_utils
#from keras.applications.imagenet_utils import decode_predictions
#from keras.applications.imagenet_utils import _obtain_input_shape

from keras.optimizers import Adam 
from keras.preprocessing.image import ImageDataGenerator 
from keras.callbacks import Callback 
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint 
from keras.callbacks import ReduceLROnPlateau



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
weight_load_path = './weights/denseNet121_weights_tf.h5' 
weight_save_path = './weights/denseNet121_weights_tf.h5'  
train_set_path = 'train_root_path/'
valid_set_path = 'validation_root_path/'
record_save_path = './records'
category_num = 2
batch_size = 32 



#-----------------------------------------------------------------------------------------
#--------------------------------model defination-----------------------------------------
#-----------------------------------------------------------------------------------------

def dense_block(x, blocks, name):
    """A dense block.
    
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.      
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x
 
def transition_block(x, reduction, name):
    """A transition block.
    
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), (1,1), 
               use_bias = False, name = name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x
 
 
def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, (1,1), use_bias = False,
                name=name + '_1_conv')(x1)    
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, (3,3), padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(blocks,
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000):

    if input_tensor is None:
        img_input = Input(shape=(224,224,3))
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
 
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    #print bn_axis
    #print K.image_data_format()
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)
 
    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')
 
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn')(x)
 
    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg': 
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

 
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    '''
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    '''
    inputs = img_input
    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(inputs, x, name='densenet201')
    else:
        model = Model(inputs, x, name='densenet')
 
    #x = Dense(1024, activation='relu')(x)
    x = model.output
    #x = Flatten()(x)
    predictions = Dense(classes, activation='softmax',use_bias=False,
                        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                        #bias_initializer=initializers.Zeros(),
                        kernel_constraint=max_norm(5.),
                        #bias_constraints=max_norm(5.),
                        #kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l2(0.01)
                        )(x)
    
    denseNet_model = Model(inputs=model.input, outputs=predictions)
    return denseNet_model
 
 
def DenseNet121(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    return DenseNet([6, 12, 24, 16],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)
 
def DenseNet169(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    return DenseNet([6, 12, 32, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)

def DenseNet201(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    return DenseNet([6, 12, 48, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)



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

model = DenseNet121(include_top = False, input_shape = (224, 224, 3),
                    pooling = 'avg', classes = category_num)

# load specified weights to continue training(if exist)
if os.path.exists(weight_load_path):
    model.load_weights(weight_load_path)

# TODO: choose training parameters
epochs = 200

# TODO: choose a optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#lrate = 0.001 
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
validation_steps= int(val_sample_count/batch_size) + 1, # batch_size, 
callbacks=[TensorBoard(log_dir=record_save_path), early_stopping, history, checkpoint, reduceLRcallback]
)
