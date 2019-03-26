# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:44:31 2018

Preprocessing images.

Copied and Modified from:
    https://github.com/tensorflow/models/blob/master/research/slim/
    preprocessing/vgg_preprocessing.py
    
@author: modified by zyb_as
"""

import math
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.

  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:

    image, depths, normals = _random_crop([image, depths, normals], 120, 150)

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.

  Returns:
    the image_list with cropped images.

  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong rank for tensor  %s [expected] [actual]',
         image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  with tf.control_dependencies([rank_assertions[0]]):
    image_shape = tf.shape(image_list[0])
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    with tf.control_dependencies([rank_assertions[i]]):
      shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height),
        ['Wrong height for tensor %s [expected][actual]',
         image.name, height, image_height])
    width_assert = tf.Assert(
        tf.equal(width, image_width),
        ['Wrong width for tensor %s [expected][actual]',
         image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  with tf.control_dependencies(asserts):
    max_offset_height = tf.reshape(image_height - crop_height + 1, [])
  with tf.control_dependencies(asserts):
    max_offset_width = tf.reshape(image_width - crop_width + 1, [])
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  return [_crop(image, offset_height, offset_width,
                crop_height, crop_width) for image in image_list]


def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs


def _normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalizes an image."""
    image = tf.to_float(image)
    return tf.div(tf.div(image, 255.) - mean, std)


def _normalize2(image):
    """Normalizes an image."""
    image = tf.subtract(image, 128)
    image = tf.div(image, 128) 
    return image



def _random_rotate(image, rotate_prob=0.5, rotate_angle_max=30, 
                   interpolation='BILINEAR'):
    """Rotates the given image using the provided angle.
    
    Args:
        image: An image of shape [height, width, channels].
        rotate_prob: The probability to roate.
        rotate_angle_angle: The upper bound of angle to ratoted.
        interpolation: One of 'BILINEAR' or 'NEAREST'.
        
    Returns:
        The rotated image.
    """
    def _rotate():
        rotate_angle = tf.random_uniform([], minval=-rotate_angle_max,
                                         maxval=rotate_angle_max, 
                                         dtype=tf.float32)
        rotate_angle = tf.div(tf.multiply(rotate_angle, math.pi), 180.)
        rotated_image = tf.contrib.image.rotate([image], [rotate_angle],
                                                interpolation=interpolation)
        return tf.squeeze(rotated_image)
    
    rand = tf.random_uniform([], minval=0, maxval=1)
    return tf.cond(tf.greater(rand, rotate_prob), lambda: image, _rotate)


def _border_expand(image, mode='CONSTANT', constant_values=255):
    """Expands the given image.
    
    Args:
        Args:
        image: A 3-D image `Tensor`.
        output_height: The height of the image after Expanding.
        output_width: The width of the image after Expanding.
        resize: A boolean indicating whether to resize the expanded image
            to [output_height, output_width, channels] or not.

    Returns:
        expanded_image: A 3-D tensor containing the resized image.
    """
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    
    def _pad_left_right():
        pad_left = tf.floordiv(height - width, 2)
        pad_right = height - width - pad_left
        return [[0, 0], [pad_left, pad_right], [0, 0]]
        
    def _pad_top_bottom():
        pad_top = tf.floordiv(width - height, 2)
        pad_bottom = width - height - pad_top
        return [[pad_top, pad_bottom], [0, 0], [0, 0]]
    
    paddings = tf.cond(tf.greater(height, width),
                       _pad_left_right,
                       _pad_top_bottom)
    expanded_image = tf.pad(image, paddings, mode=mode, 
                          constant_values=constant_values)
    return expanded_image


def border_expand(image, mode='CONSTANT', constant_values=255,
                  resize=False, output_height=None, output_width=None,
                  channels=3):
    """Expands (and resize) the given image."""
    expanded_image = _border_expand(image, mode, constant_values)
    if resize:
        if output_height is None or output_width is None:
            raise ValueError('`output_height` and `output_width` must be '
                             'specified in the resize case.')
        expanded_image = _fixed_sides_resize(expanded_image, output_height,
                                             output_width)
        expanded_image.set_shape([output_height, output_width, channels])
    return expanded_image
        

def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)


def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(tf.rint(height * scale))
  new_width = tf.to_int32(tf.rint(width * scale))
  return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image


def _fixed_sides_resize(image, output_height, output_width):
    """Resize images by fixed sides.
    
    Args:
        image: A 3-D image `Tensor`.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.

    Returns:
        resized_image: A 3-D tensor containing the resized image.
    """
    output_height = tf.convert_to_tensor(output_height, dtype=tf.int32)
    output_width = tf.convert_to_tensor(output_width, dtype=tf.int32)

    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_nearest_neighbor(
        image, [output_height, output_width], align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image



def _random_brightness(image, proportion):
    """random brightness
    
    Args:
    image: A image tensor,
    proportion: the random proportion to do brightness transformation

    Returns:
        A preprocessed image.
    """
    theshold_const = tf.constant(proportion, dtype=tf.float32)
    rand = tf.random_uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32) 
    image = tf.cond(rand[0] < theshold_const, 
                    lambda: tf.image.random_brightness(image, max_delta=30), 
                    lambda: tf.identity(image)) 
    
    # Another method of implementation(with bug)
    #def true_proc():
    #    image = tf.image.random_brightness(image, max_delta=30)
    #    return image
    #
    #def false_proc():
    #    return image
    #
    #theshold_const = tf.constant(proportion, dtype=tf.float32)
    #rand = tf.random_uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32) 
    #image = tf.cond(tf.less(rand[0], theshold_const), true_fn = true_proc, false_fn = false_proc)
    
    return image
    

def _random_flip(image, left_right_proportion, up_down_proportion):
    """random flip
    
    Args:
    image: A image tensor,
    left_right_proportion: the random proportion to do left right flip
    up_down_proportion: the random proportion to do up down flip

    Returns:
        A preprocessed image.
    """
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_flip_up_down(image)
    
    left_right_theshold = tf.constant(left_right_proportion, dtype=tf.float32)
    up_down_theshold = tf.constant(up_down_proportion, dtype=tf.float32)
    rand = tf.random_uniform([2], minval=0.0, maxval=1.0, dtype=tf.float32) 
    image = tf.cond(rand[0] < left_right_theshold, 
                    lambda: tf.image.flip_left_right(image), 
                    lambda: tf.identity(image)) 
    image = tf.cond(rand[0] < up_down_theshold, 
                    lambda: tf.image.flip_up_down(image), 
                    lambda: tf.identity(image)) 
    return image



# when do conv, make sure:
#shape of input = [batch, in_height, in_width, in_channels]
#shape of filter = [filter_height, filter_width, in_channels, out_channels]
_gauss_kernel = np.zeros((3,3,3,3))
_gauss_kernel[0,0,0,0] = 0.0751136
_gauss_kernel[0,1,0,0] = 0.123841
_gauss_kernel[0,2,0,0] = 0.0751136
_gauss_kernel[1,0,0,0] = 0.123841
_gauss_kernel[1,1,0,0] = 0.20418
_gauss_kernel[1,2,0,0] = 0.123841
_gauss_kernel[2,0,0,0] = 0.0751136
_gauss_kernel[2,1,0,0] = 0.123841
_gauss_kernel[2,2,0,0] = 0.0751136

_gauss_kernel[0,0,1,1] = 0.0751136
_gauss_kernel[0,1,1,1] = 0.123841
_gauss_kernel[0,2,1,1] = 0.0751136
_gauss_kernel[1,0,1,1] = 0.123841
_gauss_kernel[1,1,1,1] = 0.20418
_gauss_kernel[1,2,1,1] = 0.123841
_gauss_kernel[2,0,1,1] = 0.0751136
_gauss_kernel[2,1,1,1] = 0.123841
_gauss_kernel[2,2,1,1] = 0.0751136

_gauss_kernel[0,0,2,2] = 0.0751136
_gauss_kernel[0,1,2,2] = 0.123841
_gauss_kernel[0,2,2,2] = 0.0751136
_gauss_kernel[1,0,2,2] = 0.123841
_gauss_kernel[1,1,2,2] = 0.20418
_gauss_kernel[1,2,2,2] = 0.123841
_gauss_kernel[2,0,2,2] = 0.0751136
_gauss_kernel[2,1,2,2] = 0.123841
_gauss_kernel[2,2,2,2] = 0.0751136

def _gauss(image, shape):
    """random gauss fuzzy
    
    Args:
    image: A image tensor

    Returns:
        A preprocessed image.
    """
    image = tf.reshape(image, shape=(1,shape[0],shape[1],3))
    gauss_kernel = tf.constant(_gauss_kernel, dtype=tf.float32)
    image = tf.nn.conv2d(image, gauss_kernel,strides=[1,1,1,1], padding='SAME')
    image = tf.reshape(image, shape=shape)
    return image



def preprocess_for_train(image, img_shape):
    """Preprocesses the given image for training.

    Args:
    image: A image tensor,
    img_shape: shape of image

    Returns:
        A preprocessed image.
    """
    # ==========================
    # TODO: do data augmentation
    # ==========================

    # Border expand and resize
    #_fixed_resize_side = 336
    #image = border_expand(image, resize=True, 
    #        output_height=_fixed_resize_side, output_width=_fixed_resize_side)

    # random crop
    #image = tf.random_crop(image, [224, 224, 3])
    
    # random flip
    #image = _random_flip(image, left_right_proportion=0.5, up_down_proportion=0.3)
        
    # danger, pixel value may out 0-255 and return a image looks strange
    #image = _random_brightness(image, proportion=0.5)
    #tf.image.adjust_brightness
    
    # danger, pixel value may out 0-255 and return a image looks strange
    #image = tf.image.random_contrast(image, lower=0.6, upper=1)  # > 1 danger
    
    #image = tf.image.random_hue(image, max_delta=0.1)
    
    #image = tf.image.random_saturation(image, lower=0.2, upper=2)

    # gauss fuzzy
    #image = _gauss(image, img_shape)
    
    # normalize 
    image = _normalize2(image)
    return image



def preprocess_for_eval(image, img_shape):
    """Preprocesses the given image for evaluation.

    Args:
    image: A image tensor,
    img_shape: shape of image

    Returns:
        A preprocessed image.
    """
    # normalize
    image = _normalize2(image)
    return image




def preprocess(image, img_shape, is_training):
    """preprocessing.
    
    Outputs of this function can be passed to loss or postprocess functions.
    
    Args:
        image: A image tensor,
			img_shape: shape of image
        is_training: specify the preprocess operation for train or validation
                    function will not do data augmentation while validation    
    Returns:
        preprocessed inputs:
    """
    if is_training:
        return preprocess_for_train(image, img_shape)
    else:
        return preprocess_for_eval(image, img_shape)

