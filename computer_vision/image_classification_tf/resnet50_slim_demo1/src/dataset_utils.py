# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# modified by: zyb_as 2019/6/10
# ==============================================================================
"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import sys
from PIL import Image
import tensorflow as tf



def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(values):
  """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def process_image_channels(image):
    process_flag = False
    if image.mode == 'RGBA':
        # process the 4 channels .png
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r,g,b))
        process_flag = True
    elif image.mode != 'RGB':
        # process the one channel image
        image = image.convert("RGB")
        process_flag = True
    return image, process_flag


def process_image_resize(image, resize):
    if resize is not None:
        image = image.resize((resize, resize), Image.ANTIALIAS)
    return image


def process_image_resize2(image, resize):
    width, height = image.size
    if resize is not None:
        if width > height:
            width = resize
            height = int(height * resize / width) 
        else:
            height = resize
            width = int(width * resize / height)
        image = image.resize((width, height), Image.ANTIALIAS)
    return image


def get_tf_example_RGB(image_path, label):
    image = Image.open(image_path)
    # process pic to three channels(png has four channels and gray has one)
    image, process_flag = process_image_channels(image)
    # get image size: width and height
    width, height = image.size
    # get image raw data
    bytes_io = io.BytesIO()
    image.save(bytes_io, format='JPEG')
    image_raw_data = bytes_io.getvalue()
    # build tf_example
    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': bytes_feature(image_raw_data),
                'image/label': int64_feature(label),
                'image/width': int64_feature(width),
                'image/height': int64_feature(height)
            }
        ))
    return tf_example 


def get_tf_example_RGB_RESIZE(image_path, label, resize=None):
    image = Image.open(image_path)
    # process pic to three channels(png has four channels and gray has one)
    image, process_flag = process_image_channels(image)
    # reshape image
    image = process_image_resize(image, resize)
    # get image raw data
    bytes_io = io.BytesIO()
    image.save(bytes_io, format='JPEG')
    image_raw_data = bytes_io.getvalue()
    # get image size: width and height
    width, height = image.size
    # build tf_example
    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': bytes_feature(image_raw_data),
                'image/label': int64_feature(label),
                'image/width': int64_feature(width),
                'image/height': int64_feature(height)
            }
        ))
    return tf_example 


def get_tf_example_RGB_RESIZE2(image_path, label, resize=None):
    image = Image.open(image_path)
    # process pic to three channels(png has four channels and gray has one)
    image, process_flag = process_image_channels(image)
    # reshape image
    image = process_image_resize2(image, resize)
    # get image raw data
    bytes_io = io.BytesIO()
    image.save(bytes_io, format='JPEG')
    image_raw_data = bytes_io.getvalue()
    # get image size: width and height
    width, height = image.size
    # build tf_example
    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': bytes_feature(image_raw_data),
                'image/label': int64_feature(label),
                'image/width': int64_feature(width),
                'image/height': int64_feature(height)
            }
        ))
    return tf_example 


def write_label_file(labels_to_class_names, dataset_dir,
                     filename='labels.txt'):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))




def read_label_file(label_file_path):
  """Reads the labels file and returns a mapping from ID to class name.

  Args:
    label_file_path: The path of the file where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
  labels_filename = label_file_path
  with tf.gfile.Open(labels_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names
