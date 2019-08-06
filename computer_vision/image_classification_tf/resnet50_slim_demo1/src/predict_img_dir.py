# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 17:58:14 2018

对指定目录下的图片进行预测

你可以指定ckpt，pb，pb_savemodel三种格式的模型文件来完成预测

@author: zyb_as
"""
import os
import cv2
import sys
import glob
import json
import time
import base64
import shutil
import numpy as np
import tensorflow as tf

import predictor_pb
import predictor_ckpt
import predictor_savemodel

flags = tf.app.flags

flags.DEFINE_string('weight_path',
                    './log/frozen/frozen_inference_graph.pb',
                    'Path to frozen inference graph.')
flags.DEFINE_string('weight_format',
                    'ckpt',
                    'type of weight file format. use ckpt, pb or savemodel')
flags.DEFINE_string('images_dir', 
                    'testset_dir', 
                    'Path to images (directory).')
flags.DEFINE_string('output_dir', 
                    ' ', 
                    'Path to output file.')
flags.DEFINE_string('gpu_device', '1', 'Specify which gpu to be used')

FLAGS = flags.FLAGS


def detect(model, image_path):
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


def load_img2base64_str(img_path):
    with open(img_path, "rb") as image_file:
        #encoded_string = str(base64.urlsafe_b64encode(image_file.read()), "utf-8") 
        #encoded_string = str(base64.urlsafe_b64encode(image_file.read())) 

        #img_b64 = base64.b64encode(image_file.read())
        #encoded_string = str(img_b64, encoding='utf-8')

        encoded_string = image_file.read()

        #base64_data = base64.b64encode(image_file.read())
        #encoded_string = base64_data.decode()
    return encoded_string
    

if __name__ == '__main__':
    weight_path = FLAGS.weight_path
    weight_format = FLAGS.weight_format
    images_dir = FLAGS.images_dir
    output_dir = FLAGS.output_dir
    gpu_device = FLAGS.gpu_device
    threshold = 0.85
    
    print('loading model...')
    weight_type = weight_path.split('.')[-1]
    if weight_format == 'pb':
        # pb
        model = predictor_pb.Predictor(weight_path, gpu_index=gpu_device)
    elif weight_format == 'ckpt':
        # ckpt
        model = predictor_ckpt.Predictor(weight_path, gpu_index=gpu_device)
    elif weight_format == 'savemodel':
        # savemodel pb
        model = predictor_savemodel.Predictor(weight_path, gpu_index=gpu_device)
    else:
        sys.exit()
    
    if weight_format == 'pb' or weight_format == 'ckpt':
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
            
            pred_label, preprocess_time, inference_time = detect(model, image_path)
            if cnt > 1:
                total_inference_time += inference_time
                total_preprocess_time += preprocess_time 

            if pred_label[1] > threshold: # TODO: need to be modify according to actual situation
                print(pred_label)
                image_name = image_path.split('/')[-1] 
                #dst_path = os.path.join(output_dir, str(pred_label[1]) +'_' + image_name)
                dst_path = os.path.join(output_dir, image_name)
                shutil.copy(image_path, dst_path)
                
        print("------------------------------------")
        cnt -= 1
        preprocess_avg_time = total_preprocess_time * 1000 /cnt
        inference_avg_time = total_inference_time * 1000 /cnt
        print("test count: {}".format(cnt))
        print("average preprocess time: {} ms".format(preprocess_avg_time))
        print("average inference time: {} ms".format(inference_avg_time))
        print('finish')
    else:
        print("\nstart predicting...")
        cnt = 0
        image_files = glob.glob(os.path.join(images_dir, '*.*'))
        num_samples = len(image_files)
        total_inference_time = 0
        for image_path in image_files:
            cnt += 1
            if cnt % 100 == 0:
                print('Predict {}/{}.'.format(cnt, num_samples))
            
            # load img and convert to base64 str
            base64_str = load_img2base64_str(image_path)

            # predict and time
            start_time = time.time()
            pred_label = model.predict([base64_str])[0]
            end_time = time.time()
            inference_time = end_time - start_time

            if cnt > 1:
                total_inference_time += inference_time

            if pred_label[1] > threshold:   # TODO: need to be modify according to actual situation
                print(pred_label)
                image_name = image_path.split('/')[-1] 
                #dst_path = os.path.join(output_dir, str(pred_label[1]) +'_' + image_name)
                dst_path = os.path.join(output_dir, image_name)
                shutil.copy(image_path, dst_path)
                
        print("------------------------------------")
        cnt -= 1
        inference_avg_time = total_inference_time * 1000 /cnt
        print("test count: {}".format(cnt))
        print("average inference time: {} ms".format(inference_avg_time))
        print('finish')
        
