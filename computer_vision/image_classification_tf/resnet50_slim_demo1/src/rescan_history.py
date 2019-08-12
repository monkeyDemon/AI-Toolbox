# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:35:27 2018

url数据大批量回扫

@author: zyb_as
"""
from __future__ import print_function, division
import cv2
import shutil
import glob
import os
import sys
import requests
import numpy as np
import tensorflow as tf

import predictor_ckpt
import predictor_pb

flags = tf.app.flags

flags.DEFINE_string('weight_path', ' ',
                    'Path to model weight file, .pb or .ckpt is both ok.')
flags.DEFINE_string('url_dir', ' ',
                    'directory of the url files')
flags.DEFINE_string('url_save_path', ' ',
                    'path to save url hit record')
flags.DEFINE_string('gpu_device', '0', 'Specify which gpu to be used')

FLAGS = flags.FLAGS


def get_img_by_url(url):
    response = requests.get(url, timeout=1)
    imgDataNp = np.fromstring(response.content, dtype='uint8')
    img = cv2.imdecode(imgDataNp, cv2.IMREAD_UNCHANGED)   # here the img is RGB three dimensional data range from 0-255
    return img, response.content


def detect(model, image, threshold):
    # use different strategy for different size
    shape = image.shape
    width = shape[1]
    height = shape[0]
    if len(shape) == 2: 
        # gray
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif shape[2] == 4:
        # argb
        image = image[:,:,1:4]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    pred_label = model.predict([preprocess_image])[0]
    is_hit = True if pred_label[1] > threshold else False  # TODO: need to be modify according to actual situation
    return is_hit, pred_label



if __name__ == "__main__":    
    weight_path = FLAGS.weight_path
    url_dir = FLAGS.url_dir
    url_save_path = FLAGS.url_save_path
    gpu_device = FLAGS.gpu_device
    threshold = 0.5
    

    print('loading model...')
    weight_type = weight_path.split('.')[-1]
    if weight_type == 'pb':
        model = predictor_pb.Predictor(weight_path, gpu_index=gpu_device)
    else:
        # ckpt
        model = predictor_ckpt.Predictor(weight_path, gpu_index=gpu_device)
    
    with open(url_save_path, 'w') as f_write:
        for url_file in os.listdir(url_dir):    
            print("\n\nstart rescaning {}...".format(url_file))
            sys.stdout.flush()
            cnt = 0
            # read one url record file
            url_file = os.path.join(url_dir, url_file)
            for line in open(url_file).readlines():
                cnt += 1
                if cnt % 1000 == 0:
                    print(cnt)
                    sys.stdout.flush()
                line = line.rstrip()
                url = line.split('\t')[0]
                url = url.replace('\\', '')
                try:
                    img, _ = get_img_by_url(url)
                    is_hit, pred_label = detect(model, img, threshold)
                    if is_hit:
                        line = line + '\t' + str(pred_label[1]) + '\n' # TODO: need to be modify according to actual situation
                        f_write.write(line)
                        f_write.flush() # 刷新，否则不会立刻显示
                except Exception,e:
                    print("error occur: {}, url: {}".format(repr(e), url))
                    continue
    print("finish")
