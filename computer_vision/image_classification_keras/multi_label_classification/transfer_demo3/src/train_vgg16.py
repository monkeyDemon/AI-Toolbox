"""
Created on Thu Apr 25 14:42:28 2019

A multi-label classification demo with Keras

This code comes from a popular blog post, I did a slight modification on
it's data processing module to make it more easier to migrate to other issues.

To run this demo, you only need to prepare a file record the map between
image path and its multi label, just like labels.txt shows.

For more informations, see 
https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/

USAGE
python train.py --model fashion.model --label_file labels.txt

@author: Adrian
@modified: zyb_as
"""

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import keras
from keras.models import load_model
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from progressbar import *
import numpy as np
import pandas as pd
import argparse
import random
import copy
import os

# TODO: modify here if your want to use other backend
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--weight_load_path", required=True,
#	help="path to pretrained model")
ap.add_argument("-s", "--weight_save_path", required=True,
	help="path to save model")
ap.add_argument("-w", "--weight_load_path", required=True,
	help="path to save model")
ap.add_argument("-l", "--label_file", required=True,
	help="path to label map file that record image path and correspond label")
ap.add_argument("-n", "--label_num", required=True,
	help="path to category number of multi labels")
ap.add_argument("-g", "--gpu_devices", required=True,
	help="the gpu devices to be use")
args = vars(ap.parse_args())


# TODO: initialize the number of epochs to train for, 
#       initialize learning rate, batch size, image dimensions
#       initialize tag number
EPOCHS = 30
INIT_LR = 0.00001
BS = 32
IMAGE_DIMS = (96, 96, 3)
label_num = int(args['label_num'])
gpu_device = args['gpu_devices']
train_ratio = 0.9
seed = random.randint(0,100)


# Specify which gpu to be used
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

# allocate memory on demand(not fully occupied)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


def parse_multi_label(label_str, label_num):
    label_str = label_str[1:-1].split(' ')
    label_list = [0] * label_num
    for i, label in enumerate(label_str):
        label = int(label)
        label_list[i] = label
    return label_list


# load all the imgs path and corredpond multi label
print("\n[INFO] loading dataset (image_path - label map pairs)...")
dataset = []
with open(args['label_file'], 'r') as f:
    while True:
        line = f.readline().rstrip()
        if line == "":
            break
        items = line.split('\t')
        sample = [0] * (label_num + 1)
        sample[0] = items[0]
        label = parse_multi_label(items[1], label_num)
        sample[1:] = label
        dataset.append(sample)

random.shuffle(dataset)
cut_position = int(len(dataset) * train_ratio)
train_set = dataset[:cut_position]
valid_set = dataset[cut_position:]


#-----------------------------------------------------------------------------------------
#---------------------------image data generator------------------------------------------
#-----------------------------------------------------------------------------------------

labels_list = ['black', 'blue', 'dress', 'jeans', 'red', 'shirt'] 
columns_name_list = copy.copy(labels_list)
columns_name_list.insert(0, 'img_path')
print(labels_list)
print(columns_name_list)

train_df = pd.DataFrame(train_set, columns=columns_name_list)
valid_df = pd.DataFrame(valid_set, columns=columns_name_list)
print(train_df.head())
print(valid_df.head())

# construct the image generator for data augmentation
train_datagen = ImageDataGenerator(rescale=1/255.,
    rotation_range=25, 
    width_shift_range=0.1,
    height_shift_range=0.1, 
    shear_range=0.2, 
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip = True,
    fill_mode="nearest")
valid_datagen = ImageDataGenerator(rescale=1/255.)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="img_path",
    y_col=labels_list,
    batch_size=32,
    seed=seed,
    shuffle=True,
    class_mode="other",
    target_size=(IMAGE_DIMS[0],IMAGE_DIMS[1]))

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory=None,
    x_col="img_path",
    y_col=labels_list,
    batch_size=32,
    seed=seed,
    shuffle=True,
    class_mode="other",
    target_size=(IMAGE_DIMS[0],IMAGE_DIMS[1]))


#-----------------------------------------------------------------------------------------
#--Set CallBack(loss history, early stopiing, model check point, reduce learning rate)----
#-----------------------------------------------------------------------------------------

# Callback for early stopping the training 
early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=1, mode='auto')

# set model checkpoint callback (model weights will auto save in weight_save_path)
checkpoint = ModelCheckpoint(args["weight_save_path"], monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)

# monitor a learning indicator(reduce learning rate when learning effect is stagnant)
reduceLRcallback = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=7,
verbose=1, mode='auto', cooldown=0, min_lr=0)


#-----------------------------------------------------------------------------------------
#-------------------------------------training--------------------------------------------
#-----------------------------------------------------------------------------------------
def add_fully_connection_layer(base_model, nb_classes):
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(nb_classes, activation='sigmoid')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    return model

def add_convolution_layer(base_model, nb_classes):
    x = base_model.output
    x = Conv2D(nb_classes, (1, 1), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    # GAP
    x = GlobalAveragePooling2D()(x)
    predictions = Activation('sigmoid')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    return model

# load specified weights to continue training(if exist)
if os.path.exists(args["weight_save_path"]):
    # continue training 
    print("\n[INFO] loading model...")
    model = load_model(args["weight_save_path"])
else:
    print("\n[INFO] loading pretrained model...")
    # load preTrained ResNet50 model without top fully connection layer
    if os.path.exists(args["weight_load_path"]) == False:
        args["weight_load_path"] = 'imagenet'
    baseModel = VGG16(weights = args["weight_load_path"], include_top=False, input_shape=IMAGE_DIMS)
    # add new layers
    #model = add_fully_connection_layer(baseModel, label_num)
    model = add_convolution_layer(baseModel, label_num)

    
print("\n[INFO] print model structure...")
model.summary()

# TODO: choose a optimizer
#opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
opt = Adam(lr=INIT_LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# compile the model 
# using binary cross-entropy rather than categorical cross-entropy
# keep in mind that the goal here is to treat each output label as an independent Bernoulli distribution
# should be done after recover layers to trainable
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("\n[INFO] training network...")
H = model.fit_generator(
    generator = train_generator,
	steps_per_epoch = len(train_set) // BS,
	validation_data = valid_generator,
    validation_steps = len(valid_set) // BS,
	epochs=EPOCHS, 
    callbacks = [early_stopping, checkpoint, reduceLRcallback],
    verbose=1)


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = len(H.history["loss"])
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("log/learning_curve.jpg")
