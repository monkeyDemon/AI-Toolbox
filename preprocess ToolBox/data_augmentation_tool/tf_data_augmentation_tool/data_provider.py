"""
Created on Thu Oct 11 21:43:47 2018


@author: zyb_as 
"""
import tensorflow as tf
import numpy as np

import preprocessing as preproc




def datset_input_fn(filename, batch_size, epochs_num):
    dataset=tf.data.TFRecordDataset(filename)

    def parser_train(record):
        keys_to_features={
            'image/class/label':tf.FixedLenFeature(shape=(), dtype=tf.int64),
            'image/encoded':tf.FixedLenFeature(shape=(), dtype=tf.string),  
            'image/shape':tf.FixedLenFeature(shape=(3,), dtype=tf.int64)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        # get image raw data
        image = tf.decode_raw(parsed["image/encoded"], tf.uint8) ###### img_raw
        # The type is now uint8 but we need it to be float.
        image = tf.cast(image, tf.float32)
        # get image shape
        shape = parsed["image/shape"]
        # reshape
        image = tf.reshape(image, shape=shape)
        # do data augmentation
        preprocessed_image = preproc.preprocess(image, img_shape=shape, is_training=True)
        # get image class label
        label = parsed["image/class/label"]
        return preprocessed_image, label

    dataset = dataset.map(parser_train)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs_num)
    iterator = dataset.make_one_shot_iterator()
    return iterator


def get_batch_data_op(tfrecord_path, batch_size, epochs_num):    
    # get batch data iterator
    iterator = datset_input_fn(filename=tfrecord_path, 
        batch_size=batch_size, epochs_num=epochs_num)
    feature, label = iterator.get_next()
    return feature, label