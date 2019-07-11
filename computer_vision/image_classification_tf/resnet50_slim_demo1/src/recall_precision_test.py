# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:35:27 2018

测试程序示例：
对模型进行准召测试
注：不同问题需要进行相应的简单修改

@author: zyb_as
"""
from __future__ import print_function, division
import cv2
import shutil
import glob
import os
import tensorflow as tf

import predictor_pb
import predictor_ckpt

flags = tf.app.flags

flags.DEFINE_string('weight_path', ' ',
                    'Path to model weight file, .pb or .ckpt is both ok.')
flags.DEFINE_string('positive_img_dir', ' ',
                    'Path to positive images (directory).')
flags.DEFINE_string('negative_img_dir', ' ',
                    'Path to negative images (directory).')
flags.DEFINE_string('output_dir',
                    './head_rp_test',
                    'Directory to recall precision test result output file.')
flags.DEFINE_string('gpu_device', '0', 'Specify which gpu to be used')

FLAGS = flags.FLAGS


# TODO: 针对不同问题，预测函数需要进行简单修改
def detect(model, image_path, threshold):
    image_src = cv2.imread(image_path)
    # use different strategy for different size
    shape = image_src.shape
    if shape[0] <= 60 or shape[1] <= 60:
        is_hit = False  # ignore small image
        pred_label = [1, 0]
    else:
        image = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)
        pred_label = model.predict([image])[0]
        is_hit = True if pred_label[1] > threshold else False
    return is_hit, pred_label



if __name__ == "__main__":    
    weight_path = FLAGS.weight_path
    positive_img_dir = FLAGS.positive_img_dir
    negative_img_dir = FLAGS.negative_img_dir
    output_dir = FLAGS.output_dir
    gpu_device = FLAGS.gpu_device
    threshold = 0.95
    

    print('loading model...')
    weight_type = weight_path.split('.')[-1]
    if weight_type == 'pb':
        model = predictor_pb.Predictor(weight_path, gpu_index=gpu_device)
    else:
        # ckpt
        model = predictor_ckpt.Predictor(weight_path, gpu_index=gpu_device)
    
    
    # 为了分析模型，将预测错误的样本保存出来
    # 将所有测试结果保存在'output_dir'中
    if os.path.exists(output_dir): # 确保不会和以前的结果冲突
        raise RuntimeError('{} has exist, please check'.format(output_dir))
    else:
        os.mkdir(output_dir)
    
    # compute recall & precision
    print("\n\n-------------------evaluate recall & precision--------------------")
    # save False Negative sample in recall_mis_save_path
    recall_mis_save_path = os.path.join(output_dir, 'recall_mis/')
    os.mkdir(recall_mis_save_path)
    # save False Positive sample in false_detect_save_path
    false_detect_save_path = os.path.join(output_dir, 'false_detect/')
    os.mkdir(false_detect_save_path)
    # save True Positive sample in true_detect_save_path
    true_detect_save_path = os.path.join(output_dir, 'true_detect/')
    os.mkdir(true_detect_save_path)
    
    TP_count = 0  
    FP_count = 0  
    FN_count = 0  
    TN_count = 0 
    
    cnt = 0
    for root, dirs, files in os.walk(positive_img_dir):
        for filename in files:
            cnt += 1
            if cnt % 1000 == 0:
                print(cnt)
            #print('*', end='')
            image_path = os.path.join(root, filename)
            try:
                is_hit, pred_label = detect(model, image_path, threshold)
            except Exception,e:
                print("error occur: {}".format(repr(e)))
                continue
            if(is_hit == False):
                #print("\n" + imgpath + " : " + str(pred_label[1]))
                shutil.copy(image_path, recall_mis_save_path + str(pred_label[1]) + '_' + filename)
                #shutil.move(image_path, recall_mis_save_path + filename)
                FN_count += 1
            else:
                shutil.copy(image_path, true_detect_save_path + str(pred_label[1]) + '_' + filename)
                TP_count += 1
    
    print('\n------------------------------------')
    cnt = 0
    for root, dirs, files in os.walk(negative_img_dir):
        for filename in files:
            cnt += 1
            if cnt % 1000 == 0:
                print(cnt)
            #print('*', end='')
            image_path = os.path.join(root, filename)
            try:
                is_hit, pred_label = detect(model, image_path, threshold)
            except Exception,e:
                print("error occur: {}".format(repr(e)))
                continue
            if(is_hit == True):
                #print("\n" + imgpath + " : " + str(pred_label[1]))
                shutil.copy(image_path, false_detect_save_path + str(pred_label[1]) + '_' + filename)
                #shutil.move(image_path, false_detect_save_path + filename)
                FP_count += 1
            else:
                TN_count += 1    
    
    precision = TP_count / (TP_count + FP_count) * 100
    print('precision: %f' % precision)
    recall = TP_count / (TP_count + FN_count) * 100
    print('recall: %f' % recall)
