# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:44:31 2018

Preprocessing images and do data augmentation

@author: zyb_as

Copied and Modified from:
    https://github.com/tensorflow/models/blob/master/research/slim/
    preprocessing/vgg_preprocessing.py
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


def border_expand(image, mode='CONSTANT', constant_values=0,
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




'''
def _normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalizes an image.
    do the same normalize operation of vgg
    input pixel value range [0,255]
    """
    image = tf.to_float(image)
    return tf.div(tf.div(image, 255.) - mean, std)


def _normalize_2(image):
    """Normalizes an image."""
    image = tf.to_float(image)
    image = tf.subtract(image, 128)
    image = tf.div(image, 128) 
    return image


def _normalize_3(image):
    """Normalizes an image.
    input pixel value range [0,255] 
    result pixel value range [0, 1]
    """
    image = tf.to_float(image)
    image = tf.div(image, 255) 
    return image
'''

    

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


def _random_crop(image, crop_size, crop_probability):
    theshold_const = tf.constant(crop_probability, dtype=tf.float32)
    rand = tf.random_uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32) 
    image = tf.cond(rand[0] < theshold_const,   
                    lambda: tf.random_crop(image, [crop_size, crop_size, 3]),
                    lambda: tf.cast(tf.image.resize_images(image, [crop_size, crop_size]), dtype=tf.uint8))
    return image


def _random_flip(image, left_right_probability, up_down_probability):
    """random flip
    
    Args:
    image: A image tensor,
    left_right_probability: the random probability to do left right flip
    up_down_probability: the random probability to do up down flip

    Returns:
        A preprocessed image.
    """
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_flip_up_down(image)
    
    left_right_theshold = tf.constant(left_right_probability, dtype=tf.float32)
    up_down_theshold = tf.constant(up_down_probability, dtype=tf.float32)
    rand = tf.random_uniform([2], minval=0.0, maxval=1.0, dtype=tf.float32) 
    image = tf.cond(rand[0] < left_right_theshold, 
                    lambda: tf.image.flip_left_right(image), 
                    lambda: tf.identity(image)) 
    image = tf.cond(rand[1] < up_down_theshold, 
                    lambda: tf.image.flip_up_down(image), 
                    lambda: tf.identity(image)) 
    return image


def _transpose_image(image, probability):
    theshold_const = tf.constant(probability, dtype=tf.float32)
    rand = tf.random_uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32) 
    image = tf.cond(rand[0] < theshold_const,   
                    lambda: tf.image.transpose_image(image),
                    lambda: tf.identity(image))
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




# ==========================
# color space transformation
# ==========================
    
def _random_adjust_brightness(image, probability):
    theshold_const = tf.constant(probability, dtype=tf.float32)
    rand = tf.random_uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32) 
    # tf.image.random_brightness, 在[-max_delta, max_delta)的范围随机调整图片的亮度。
    image = tf.cond(rand[0] < theshold_const, 
                    lambda: tf.image.random_brightness(image, max_delta=0.3), 
                    lambda: tf.identity(image)) 
    return image

   
def _random_adjust_contrast(image, probability):   
    theshold_const = tf.constant(probability, dtype=tf.float32)
    rand = tf.random_uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32) 
    image = tf.cond(rand[0] < theshold_const, 
                    lambda: tf.image.random_contrast(image, lower=0.5, upper=1.5),
                    lambda: tf.identity(image))
    return image

  
def _random_adjust_hue(image, probability):
    theshold_const = tf.constant(probability, dtype=tf.float32)
    rand = tf.random_uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32)
    # 在[-max_delta, max_delta]的范围随机调整图片的色相。max_delta的取值在[0, 0.5]之间。
    image = tf.cond(rand[0] < theshold_const,   # 0.01，>1
                    lambda: tf.image.random_hue(image, 0.15), 
                    lambda: tf.identity(image)) 
    return image


def _random_adjust_saturation(image, probability):
    theshold_const = tf.constant(probability, dtype=tf.float32)
    rand = tf.random_uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32) 
    image = tf.cond(rand[0] < theshold_const,   
                    lambda: tf.image.random_saturation(image, lower=0.5, upper=3),
                    lambda: tf.identity(image))
    return image

  
def _random_uniform_noise(image, probability, img_shape):
    def add_uniform_noise(image, img_shape):
        minval=-20  # uniform noise range [minval, maxval]
        maxval=20
        # change to float data type
        image = tf.to_float(image)
        # add random noise
        image = image + tf.random_uniform(shape=img_shape, minval=minval, maxval=maxval)
        # clip the value out of range
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=255) 
        image = tf.cast(image, dtype=tf.uint8)
        return image
    theshold_const = tf.constant(probability, dtype=tf.float32)
    rand = tf.random_uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32) 
    image = tf.cond(rand[0] < theshold_const,   
                    lambda: add_uniform_noise(image, img_shape),
                    lambda: tf.identity(image))
    return image



def _gauss_kernel(image, shape):
    '''
    gauss filtering
    
    here we use convolution opertation to do gauss filtering 
    when do conv, make sure:
    shape of input = [batch, in_height, in_width, in_channels]
    shape of filter = [filter_height, filter_width, in_channels, out_channels]
    see for more https://blog.csdn.net/dcrmg/article/details/52304446
    if you compute correct, a 3*3 kernel will same as bellow
    #_gauss_kernel = np.zeros((3,3,3,3))
    #_gauss_kernel[0,0,0,0] = 0.0751136
    #_gauss_kernel[0,1,0,0] = 0.123841
    #_gauss_kernel[0,2,0,0] = 0.0751136
    #_gauss_kernel[1,0,0,0] = 0.123841
    #_gauss_kernel[1,1,0,0] = 0.20418
    #_gauss_kernel[1,2,0,0] = 0.123841
    #_gauss_kernel[2,0,0,0] = 0.0751136
    #_gauss_kernel[2,1,0,0] = 0.123841
    #_gauss_kernel[2,2,0,0] = 0.0751136
    #
    #_gauss_kernel[0,0,1,1] = 0.0751136
    #_gauss_kernel[0,1,1,1] = 0.123841
    #_gauss_kernel[0,2,1,1] = 0.0751136
    #_gauss_kernel[1,0,1,1] = 0.123841
    #_gauss_kernel[1,1,1,1] = 0.20418
    #_gauss_kernel[1,2,1,1] = 0.123841
    #_gauss_kernel[2,0,1,1] = 0.0751136
    #_gauss_kernel[2,1,1,1] = 0.123841
    #_gauss_kernel[2,2,1,1] = 0.0751136
    #
    #_gauss_kernel[0,0,2,2] = 0.0751136
    #_gauss_kernel[0,1,2,2] = 0.123841
    #_gauss_kernel[0,2,2,2] = 0.0751136
    #_gauss_kernel[1,0,2,2] = 0.123841
    #_gauss_kernel[1,1,2,2] = 0.20418
    #_gauss_kernel[1,2,2,2] = 0.123841
    #_gauss_kernel[2,0,2,2] = 0.0751136
    #_gauss_kernel[2,1,2,2] = 0.123841
    #_gauss_kernel[2,2,2,2] = 0.0751136
    '''
    r = 5#random.randint(1,3)
    s = 1#random.uniform(1, 1.5)  
    summat = 0
    #PI = 3.14159265358979323846
    PI = math.pi
    _gauss_kernel = np.zeros((2*r+1,2*r+1,3,3))
    for i in range(0,2*r+1):
        for j in range(0,2*r+1):
            gaussp = (1/(2*PI*(s**2))) * math.e**(-((i-r)**2+(j-r)**2)/(2*(s**2))) 
            _gauss_kernel[i,j,0,0] = gaussp
            _gauss_kernel[i,j,1,1] = gaussp
            _gauss_kernel[i,j,2,2] = gaussp
            summat += gaussp
    for i in range(0,2*r+1):
        for j in range(0,2*r+1):
            _gauss_kernel[i,j,0,0] = _gauss_kernel[i,j,0,0]/summat
            _gauss_kernel[i,j,1,1] = _gauss_kernel[i,j,1,1]/summat
            _gauss_kernel[i,j,2,2] = _gauss_kernel[i,j,2,2]/summat
    
    #image = tf.reshape(image, shape=(1,shape[0],shape[1],3))
    image = tf.expand_dims(image, 0)
    image = tf.cast(image, tf.float32)
    gauss_kernel = tf.constant(_gauss_kernel, dtype=tf.float32)
    image = tf.nn.conv2d(image, gauss_kernel,strides=[1,1,1,1], padding='SAME')
    image = tf.reshape(image, shape=shape)
    image = tf.cast(image, dtype=tf.uint8)
    return image


def _random_gauss_filtering(image, probability, img_shape):
    """random gauss fuzzy
    
    random do gauss filtering
    
    warning: 高斯滤波结果会产生黑边，暂时还没有想到优雅的解决方案，使用请谨慎
    
    Args:
    image: A image tensor
    probability: the probability to do gauss filtering
    img_shape: shape of image
    
    Returns:
        A preprocessed image.
    """
    theshold_const = tf.constant(probability, dtype=tf.float32)
    rand = tf.random_uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32) 
    image = tf.cond(rand[0] < theshold_const, 
                    lambda: _gauss_kernel(image, img_shape), 
                    lambda: tf.identity(image)) 
    return image




# ==========================
# interface function to call
# ==========================


def augmentation_for_train(image, img_shape):
    """Preprocesses the given image for training.

    First, resize the image to uniform size 256 * 256
    Second, do all the color space transformations(adjust brightness, contrast and so on...)
    Third, do random crop to uniform size 224 * 224
    At last, do all the position transformations

    Args:
    image: A image tensor,
    img_shape: shape of image

    Returns:
        A preprocessed image.
    """
    # resize
    #resize_size = 256
    #image = tf.cast(tf.image.resize_images(image, [resize_size, resize_size]), dtype=tf.uint8)
    #img_shape = tf.shape(image)
    
    # ==========================
    # TODO: do data augmentation
    # ==========================

    # -----color space transformation-----
    
    # 随机调整亮度 
    image = _random_adjust_brightness(image, probability=0.3)
    
    # 随机调整对比度
    image = _random_adjust_contrast(image, 0.3)  
    
    # 随机调整色相
    image = _random_adjust_hue(image, 0.3)
    
    # 图像饱和度，
    image = _random_adjust_saturation(image, 0.3)    
    
    # 随机添加均匀分布噪声  
    #image = _random_uniform_noise(image, 0.3, img_shape)

    # 随机进行高斯滤波
    #image = _random_gauss_filtering(image, 0.3, img_shape)   
    
    
    # -----position transformation-----

    # random crop
    # if img_shape > network input size, need to do random crop
    # this is also a effective data augmentation method
    crop_size = 224
    image = _random_crop(image, crop_size, crop_probability=0.5)

    # random flip 
    image = _random_flip(image, left_right_probability=0.5, up_down_probability=0.3)
    
    # 随机转置
    image = _transpose_image(image, 0.2)
    
    # 随机旋转
    image = _random_rotate(image, rotate_prob=0.3, rotate_angle_max=15)
    
    
    # Border expand and resize
    # 以最长边扩展为最大正方形, 和其他方法综合使用可能会造成一些奇怪的生成结果
    #_fixed_resize_side = 224
    #image = border_expand(image, resize=True, 
    #        output_height=_fixed_resize_side, output_width=_fixed_resize_side)
    
    # normalize  
    #image = _normalize_2(image)
    return image


def augmentation_for_eval(image, img_shape):
    """Preprocesses the given image for evaluation.

    Args:
    image: A image tensor,
    img_shape: shape of image

    Returns:
        A preprocessed image.
    """
    # resize
    resize_size = 224
    image = tf.cast(tf.image.resize_images(image, [resize_size, resize_size]), dtype=tf.uint8)
    
    ## normalize
    #image = _normalize_2(image)
    return image


def augmentation_for_train2(image, img_shape):
    """Preprocesses the given image for training.

    make sure the input image's long edge has been resized to 256 (maintain aspect ratio)
    First, do all the color space transformations(adjust brightness, contrast and so on...)
    Second, padding the image to uniform size 256 * 256
    Third, do random crop to uniform size 224 * 224
    At last, do all the position transformations
    
    Args:
    image: A image tensor,
    img_shape: shape of image

    Returns:
        A preprocessed image.
    """
    # ==========================
    # TODO: do data augmentation
    # ==========================

    # -----color space transformation-----
    
    # 随机调整亮度 
    image = _random_adjust_brightness(image, probability=0.3)
    
    # 随机调整对比度
    image = _random_adjust_contrast(image, 0.3)  
    
    # 随机调整色相
    image = _random_adjust_hue(image, 0.3)
    
    # 图像饱和度，
    image = _random_adjust_saturation(image, 0.3)    
    
    # 随机添加均匀分布噪声  
    #image = _random_uniform_noise(image, 0.3, img_shape)

    # 随机进行高斯滤波
    #image = _random_gauss_filtering(image, 0.3, img_shape)   
    
    
    # -----position transformation-----

    # padding
    resize_size = 256
    image = tf.image.resize_image_with_crop_or_pad(image, resize_size, resize_size)
    
    # random crop
    # if img_shape > network input size, need to do random crop
    # this is also a effective data augmentation method
    crop_size = 224
    image = _random_crop(image, crop_size, crop_probability=0.5)

    # random flip 
    image = _random_flip(image, left_right_probability=0.5, up_down_probability=0.3)
    
    # 随机转置
    image = _transpose_image(image, 0.2)
    
    # 随机旋转
    image = _random_rotate(image, rotate_prob=0.3, rotate_angle_max=15)
    
    # normalize  
    #image = _normalize_2(image)
    return image


def augmentation_for_eval2(image, img_shape):
    """Preprocesses the given image for evaluation.

    make sure the input image's long edge has been resized to 256 (maintain aspect ratio)
    padding to uniform size 256 * 256
    resize the image to uniform size 224 * 224
    
    Args:
    image: A image tensor,
    img_shape: shape of image

    Returns:
        A preprocessed image.
    """
    # padding
    resize_size = 256
    image = tf.image.resize_image_with_crop_or_pad(image, resize_size, resize_size)

    # resize
    resize_size = 224
    image = tf.cast(tf.image.resize_images(image, [resize_size, resize_size]), dtype=tf.uint8)
    
    # normalize
    #image = _normalize_2(image)
    return image


def augmentation(image, img_shape, is_training):
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
        return augmentation_for_train(image, img_shape)
    else:
        return augmentation_for_eval(image, img_shape)
