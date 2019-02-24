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

import dataset_utils


# TODO: set parameters
# The number of images in the validation set.
_TRAIN_RATIO = 0.9

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5

# The dataset root directory containing a set of subdirectories representing
# class names. Each subdirectory should contain PNG or JPG encoded images.
_DATASET_ROOT_PATH = './dataset'

# The directory to save the tfrecord files
_TFRECORD_SAVE_PATH = './dataset_tfrecord'

# Give a base name for the dataset
_DATASET_BASE_NAME = 'flowers'



class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

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
  output_filename = _DATASET_BASE_NAME
  output_filename += '_%s_%05d-of-%05d.tfrecord' % (split_name, 
                    shard_id, _NUM_SHARDS)
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

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            _TFRECORD_SAVE_PATH, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def build_tfrecord(dataset_root_dir, tfrecord_save_path):
  """build the TF record files.

  Args:
    dataset_root_dir: The dataset directory where the dataset is stored.
    tfrecord_save_path: The directory to save the tfrecord files
  """
  if not tf.gfile.Exists(tfrecord_save_path):
    tf.gfile.MakeDirs(tfrecord_save_path)

  photo_filenames, class_names = _get_filenames_and_classes(dataset_root_dir)
  
  # Shuffle and divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  valid_num = int((1 - _TRAIN_RATIO) * len(photo_filenames))
  training_filenames = photo_filenames[valid_num:]
  validation_filenames = photo_filenames[:valid_num]

  class_names_to_ids = dict(zip(class_names, range(len(class_names))))
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, class_names_to_ids)
  _convert_dataset('validation', validation_filenames, class_names_to_ids)

  # Finally, write the labels file:
  dataset_utils.write_label_file(labels_to_class_names, tfrecord_save_path)

  print('\nFinished converting the dataset!')
  
  

if __name__ == "__main__":
    build_tfrecord(_DATASET_ROOT_PATH, _TFRECORD_SAVE_PATH)