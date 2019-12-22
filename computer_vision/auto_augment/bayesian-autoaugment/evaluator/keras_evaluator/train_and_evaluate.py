# -*- coding: utf-8 -*-
# @Date    : 2019/9/15 19:08
# @Version : 1.0.0
# @Brief   : mobilenetv1 classify
import os
import argparse
from keras import Model
from keras.applications import MobileNet, ResNet50
from keras.callbacks import *
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from augmentation import *

parser = argparse.ArgumentParser(description='indecent classify', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, help='classify model.')
parser.add_argument('--pretrain_weight', type=str, help='the pretrain weight. None or imagenet.')
parser.add_argument('--train_dir', type=str, help='training dataset.')
parser.add_argument('--val_dir', type=str, help='val dataset.', default='val_person')
parser.add_argument('--cmd_id', type=int, help='command id.') 
parser.add_argument('--gpu_num', type=str, help='Number of point.')
parser.add_argument('--polices_str', type=str, help='data augment polices in string format.')
parser.add_argument('--class_num', type=int, help='number of the category to predict')
parser.add_argument('--max_sample_per_class', type=int, help='max sample number per category.')
parser.add_argument('--checkpoint_filepath', type=str, help='checkpoint_filepath')
parser.add_argument('--cost_filepath', type=str, help='cost_filepath') 
parser.add_argument('--class_weight', type=str, default='1,1', help='class weight, use \, to split.')
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

TRAIN_DIR = args.train_dir
VALID_DIR = args.val_dir
SIZE = (160, 160)
BATCH_SIZE = 64 
CLASS_NUM = args.class_num
MAX_SAMPLE = args.max_sample_per_class
PRETRAIN = 'imagenet' if args.pretrain_weight == 'imagenet' else None
if not os.path.exists(args.checkpoint_filepath):
    os.makedirs(args.checkpoint_filepath)

if args.model == "mobilenet":
    base_model = MobileNet(weights=PRETRAIN,
                           include_top=False,
                           input_shape=(SIZE[0], SIZE[1], 3))
elif args.model == "resnet50":
    base_model = ResNet50(weights=PRETRAIN,
                          include_top=False,
                          input_shape=(SIZE[0], SIZE[1], 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)  # dense layer 3
preds = Dense(CLASS_NUM, activation='softmax')(x)  # final layer with softmax activation
model = Model(inputs=base_model.input, outputs=preds)


train_datagen = data_generator(TRAIN_DIR, SIZE, CLASS_NUM, BATCH_SIZE, args.polices_str, MAX_SAMPLE)
valid_datagen = data_generator(VALID_DIR, SIZE, CLASS_NUM, BATCH_SIZE, args.polices_str, -1)


model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# set callbacks
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

checkpoint_best = ModelCheckpoint("%s/best_model.h5" % args.checkpoint_filepath,
                                   monitor='val_acc',
                                   verbose=1,
                                   mode='max',
                                   save_best_only=True)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', 
                                         factor=0.1,
                                         patience=5, 
                                         verbose=1)

early_stopping = EarlyStopping(monitor='val_acc', 
                               min_delta=0, 
                               patience=12, 
                               verbose=1, 
                               mode='auto')
callbacks = [checkpoint_best, reduce_learning_rate, early_stopping, history]


step_size_train = train_datagen.num_of_examples // train_datagen.batch_size
step_size_val = valid_datagen.num_of_examples // valid_datagen.batch_size

class_weight = {}
if args.class_weight == "":
    for index in range(CLASS_NUM):
        class_weight[index] = 1 
else:
    weights = args.class_weight.split(',')
    for index in range(CLASS_NUM):
        class_weight[index] = weights[index]
print(class_weight)


history_record = model.fit_generator(train_datagen.get_mini_batch(use_aug=True), 
                    steps_per_epoch=step_size_train, epochs=50,  
                    callbacks=callbacks, 
                    validation_data=valid_datagen.get_mini_batch(),
                    validation_steps=step_size_val,
                    class_weight=class_weight)
                    #use_multiprocessing=True)

# compute cost and save
record = history_record.history
val_acc_list = record['val_acc']
best_val_acc = max(val_acc_list)
cost = 1 - best_val_acc

cmd_id = args.cmd_id
cost_file_path = args.cost_filepath
with open(cost_file_path, 'a+') as writer:
    line = str(cmd_id) + '|' + str(cost) + '\n'
    writer.write(line)

sys.exit(0)
