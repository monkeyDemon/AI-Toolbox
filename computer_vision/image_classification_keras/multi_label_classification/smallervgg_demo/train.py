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
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from progressbar import *
import numpy as np
import argparse
import random
import cv2
import os

# TODO: modify here if your want to use other backend
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--label_file", required=True,
	help="path to label map file")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())


# TODO: initialize the number of epochs to train for, 
#       initialize learning rate, batch size, image dimensions
#       initialize tag number
EPOCHS = 75
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)
tag_num = 6
gpu_device = '5'


# Specify which gpu to be used
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

# allocate memory on demand(not fully occupied)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


def parse_multi_label(label_str, tag_num):
    label_str = label_str[1:-1].split(' ')
    label_list = [0] * tag_num
    for i, label in enumerate(label_str):
        label = int(label)
        label_list[i] = label
    return label_list


# load all the imgs path and corredpond label
print("\n[INFO] loading image_path - label map file...")
data_label_map_list = []
with open(args['label_file'], 'r') as f:
    while True:
        line = f.readline().rstrip()
        if line == "":
            break
        data_label_map_list.append(line)

# randomly shuffle 
random.seed(42)
random.shuffle(data_label_map_list)


print("\n[INFO] loading images...")
pro = ProgressBar()
total_num = len(data_label_map_list)
cnt = 0
data = []
labels = []
for idx in pro(range(total_num)):
    line = data_label_map_list[idx]
    items = line.split('\t')
    # get img data
    img_path = items[0]
    image = cv2.imread(img_path)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)
    # get correspond label
    label = parse_multi_label(items[1], tag_num)
    labels.append(label)
    

print("\n[INFO] preparing datasets...")
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(labels), data.nbytes / (1024 * 1000.0)))

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("\n[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=tag_num,
	finalAct="sigmoid")

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("\n[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("\n[INFO] serializing network...")
model.save(args["model"])


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
