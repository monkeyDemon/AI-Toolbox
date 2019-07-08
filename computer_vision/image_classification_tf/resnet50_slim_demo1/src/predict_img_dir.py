# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 17:58:14 2018

@author: zyb_as
"""

import os
import cv2
import glob
import json
import time
import shutil
import numpy as np
import tensorflow as tf

import predictor_pb
import predictor_ckpt

flags = tf.app.flags

flags.DEFINE_string('weight_path',
                    './log/frozen/frozen_inference_graph.pb',
                    'Path to frozen inference graph.')
flags.DEFINE_string('images_dir', 
                    'testset_dir', 
                    'Path to images (directory).')
flags.DEFINE_string('output_dir', 
                    ' ', 
                    'Path to output file.')
flags.DEFINE_string('gpu_device', '1', 'Specify which gpu to be used')

FLAGS = flags.FLAGS


def detect_porn(model, image_path):
    image_src = cv2.imread(image_path)
    shape = image_src.shape
    width = shape[1]
    height = shape[0]

    # time preprocess use time
    start_time = time.time()
    if len(shape) == 2:
        image = cv2.cvtColor(image_src, cv2.COLOR_GRAY2RGB)
    image = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)

    # resize(maintain aspect ratio) 
    long_edge_size = 224
    if width > height:
        height = int(height * long_edge_size / width)
        width = long_edge_size
    else:
        width = int(width * long_edge_size / height)
        height = long_edge_size
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    # padding
    preprocess_image = np.zeros((long_edge_size, long_edge_size, 3))
    if width > height:
        st = int((long_edge_size - height)/2)
        ed = st + height
        preprocess_image[st:ed,:,:] = image
    else:
        st = int((long_edge_size - width)/2)
        ed = st + width
        preprocess_image[:,st:ed,:] = image
    end_time = time.time()
    preprocess_time = end_time - start_time

    # time inference use time
    start_time = time.time()
    pred_label = model.predict([preprocess_image])[0]
    end_time = time.time()
    inference_time = end_time - start_time

    return pred_label, preprocess_time, inference_time



if __name__ == '__main__':
    weight_path = FLAGS.weight_path
    images_dir = FLAGS.images_dir
    output_dir = FLAGS.output_dir
    gpu_device = FLAGS.gpu_device
    threshold = 0.9
    
    print('loading model...')
    weight_type = weight_path.split('.')[-1]
    if weight_type == 'pb':
        model = predictor_pb.Predictor(weight_path, gpu_index=gpu_device)
    else:
        # ckpt
        model = predictor_ckpt.Predictor(weight_path, gpu_index=gpu_device)
    
    print("\nstart predicting...")
    
    image_files = glob.glob(os.path.join(images_dir, '*.*'))

    cnt = 0
    num_samples = len(image_files)
    total_inference_time = 0
    total_preprocess_time = 0
    for image_path in image_files:
        cnt += 1
        if cnt % 100 == 0:
            print('Predict {}/{}.'.format(cnt, num_samples))
        
        pred_label, preprocess_time, inference_time = detect_porn(model, image_path)
        if cnt > 1:
            total_inference_time += inference_time
            total_preprocess_time += preprocess_time 

        if pred_label[1] > threshold:
            print(pred_label)
            image_name = image_path.split('/')[-1] 
            dst_path = os.path.join(output_dir, str(pred_label[1]) +'_' + image_name)
            shutil.copy(image_path, dst_path)
            
    print("------------------------------------")
    cnt -= 1
    preprocess_avg_time = total_preprocess_time * 1000 /cnt
    inference_avg_time = total_inference_time * 1000 /cnt
    print("test count: {}".format(cnt))
    print("average preprocess time: {} ms".format(preprocess_avg_time))
    print("average inference time: {} ms".format(inference_avg_time))
    print('finish')
