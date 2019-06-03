# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 17:58:14 2018

@author: zyb_as
"""

import cv2
import glob
import json
import os
import shutil
import tensorflow as tf

import predictor
import predictor_ckpt

flags = tf.app.flags

flags.DEFINE_string('weight_path',
                    './training/frozen_inference_graph_pb/'+
                    'frozen_inference_graph.pb',
                    'Path to frozen inference graph.')
flags.DEFINE_string('images_dir', 
                    '/data2/raycloud/jingxiong_datasets/AIChanllenger/' +
                     'AgriculturalDisease_testA/images', 
                    'Path to images (directory).')
flags.DEFINE_string('output_dir', 
                    ' ', 
                    'Path to output file.')
flags.DEFINE_string('gpu_device', '1', 'Specify which gpu to be used')

FLAGS = flags.FLAGS


if __name__ == '__main__':
    weight_path = FLAGS.weight_path
    images_dir = FLAGS.images_dir
    output_dir = FLAGS.output_dir
    gpu_device = FLAGS.gpu_device
    threshold = 0.9
    
    print('loading model...')
    weight_type = weight_path.split('.')[-1]
    if weight_type == 'pb':
        model = predictor.Predictor(weight_path, gpu_index=gpu_device)
    else:
        # ckpt
        model = predictor_ckpt.Predictor(weight_path, gpu_index=gpu_device)
    
    print("\nstart predicting...")
    
    image_files = glob.glob(os.path.join(images_dir, '*.*'))

    val_results = []
    predicted_count = 0
    num_samples = len(image_files)
    for image_path in image_files:
        predicted_count += 1
        if predicted_count % 100 == 0:
            print('Predict {}/{}.'.format(predicted_count, num_samples))
        
        image_name = image_path.split('/')[-1]
        image = cv2.imread(image_path)
        if image is None:
            print('image %s does not exist.' % image_name)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)
            
        #pred_label = int(model.predict([image])[0])
        pred_label = model.predict([image])[0]

        if pred_label[1] > threshold:
            print(pred_label)
            dst_path = os.path.join(output_dir, str(pred_label[1]) +'_' + image_name)
            shutil.copy(image_path, dst_path)
            
    print('finish')
