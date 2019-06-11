"""
Created on Mon Feb 11 14:34:02 2019

Converts images data to TFRecords of TF-Example protos.

This module reads the files and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

this script is modified on 
https://github.com/tensorflow/models/tree/master/research/slim/datasets

@author: The TensorFlow Authors
@modified by: zyb_as
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import tensorflow as tf
from PIL import Image
import numpy as np

import dataset_utils

flags = tf.app.flags
# TODO: set parameters
flags.DEFINE_string('dataset_root_path',
					'/home/ansheng/cv_strategy/porn_detect/dataset/DATASET_zyb_v1/dataset_mini',
					'The dataset root directory containing a set of subdirectories representing' +
					'class names. Each subdirectory should contain PNG or JPG encoded images.')
flags.DEFINE_string('tfrecord_save_path',
					'/home/ansheng/cv_strategy/porn_detect/cnn_tf/classification_by_slim/tfrecord',
					'The directory to save the tfrecord files')
flags.DEFINE_string('dataset_base_name',
					'zyb_v1_mini',
					'Give a base name for the dataset')
flags.DEFINE_integer('train_num_shards', 5,
					'The number of shards per dataset split, ' +
					'also determine the ratio of train and validation set.')
flags.DEFINE_integer('valid_num_shards', 1,
					'The number of shards per dataset split, ' +
					'also determine the ratio of train and validation set.')
flags.DEFINE_integer('random_seed', 0, 'Seed for repeatability.')
flags.DEFINE_integer('zoom_size', -1, 'Specify uniform image zoom size.')
flags.DEFINE_string('cuda_visible_devices', '0', 'Specify which gpu to be used.')

FLAGS = flags.FLAGS

_DATASET_ROOT_PATH = FLAGS.dataset_root_path
_TFRECORD_SAVE_PATH = FLAGS.tfrecord_save_path
_DATASET_BASE_NAME = FLAGS.dataset_base_name
_TRAIN_NUM_SHARDS = FLAGS.train_num_shards
_VALID_NUM_SHARDS = FLAGS.valid_num_shards
_RANDOM_SEED = FLAGS.random_seed
_ZOOM_SIZE = FLAGS.zoom_size
_CUDA_VISIBLE_DEVICES = FLAGS.cuda_visible_devices



class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def read_image_shape(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image



def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  directories = []
  class_names = []
  for sub_dir_name in os.listdir(dataset_dir):
    path = os.path.join(dataset_dir, sub_dir_name)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(sub_dir_name)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  assert split_name in ['train', 'validation']
  num_shards = _TRAIN_NUM_SHARDS if split_name == 'train' else _VALID_NUM_SHARDS

  output_filename = _DATASET_BASE_NAME
  output_filename += '_%s_%05d-of-%05d.tfrecord' % (split_name, 
                    shard_id, num_shards)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
  """
  assert split_name in ['train', 'validation']
  num_shards = _TRAIN_NUM_SHARDS if split_name == 'train' else _VALID_NUM_SHARDS

  num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))
  error_count = 0    # record the number of exceptions
  with tf.Graph().as_default():
    image_reader = ImageReader()

    # setting not fully occupied memory, allocated on demand
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #sess = tf.Session(config = config)
    with tf.Session(config = config) as sess:

      for shard_id in range(num_shards):
        output_filename = _get_dataset_filename(
            _TFRECORD_SAVE_PATH, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            try:
                #tf_example = dataset_utils.get_tf_example_RGB(filenames[i], class_id)
                #tf_example = dataset_utils.get_tf_example_RGB_RESIZE2(filenames[i], class_id, _ZOOM_SIZE)
                tf_example = dataset_utils.get_tf_example_RGB_RESIZE(filenames[i], class_id, _ZOOM_SIZE)
            except Exception, e:
                print("error occur! img {} with error: {}".format(filenames[i], repr(e)))
                error_count += 1
                continue

            tfrecord_writer.write(tf_example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.write('the number of exceptions occured: %d \n' % error_count)
  sys.stdout.flush()


def build_tfrecord(dataset_root_dir, tfrecord_save_path):
  """build the TF record files.

  Args:
    dataset_root_dir: The dataset directory where the dataset is stored.
    tfrecord_save_path: The directory to save the tfrecord files
  """
  print('\nStart...')
  
  if not tf.gfile.Exists(tfrecord_save_path):
    tf.gfile.MakeDirs(tfrecord_save_path)
  else:
	print("tfrecord_save_path has exist, please check!")
	print("stop")
	return
	

  print('\nloading all images\' filename list...')
  photo_filenames, class_names = _get_filenames_and_classes(dataset_root_dir)
  
  # Shuffle and divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  train_ratio = _TRAIN_NUM_SHARDS / (_TRAIN_NUM_SHARDS + _VALID_NUM_SHARDS)
  valid_num = int((1 - train_ratio) * len(photo_filenames))
  training_filenames = photo_filenames[valid_num:]
  validation_filenames = photo_filenames[:valid_num]
  print('the total size of training dataset: %d' % len(training_filenames))
  print('the total size of validation dataset: %d' % len(validation_filenames))

  class_names_to_ids = dict(zip(class_names, range(len(class_names))))
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))

  # First, convert the training and validation sets.
  print('\nStart converting the training dataset...')
  _convert_dataset('train', training_filenames, class_names_to_ids)
  print('\nStart converting the validation dataset...')
  _convert_dataset('validation', validation_filenames, class_names_to_ids)

  # Finally, write the labels file:
  dataset_utils.write_label_file(labels_to_class_names, tfrecord_save_path)

  print('\nFinished converting the dataset!')
  
  

if __name__ == "__main__":
  # Specify which gpu to be used
  os.environ["CUDA_VISIBLE_DEVICES"] = _CUDA_VISIBLE_DEVICES
  # build tf record files
  build_tfrecord(_DATASET_ROOT_PATH, _TFRECORD_SAVE_PATH)
