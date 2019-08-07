# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:35:27 2018

一个多功能性能测试程序demo

与predict_img_dir.py仅仅对指定目录下图像进行预测不同
recall_precision_test.py会对模型进行准召测试
绘制准召变化曲线，计算曲线下面积
从多个角度为模型比较和选择提供参考

@author: zyb_as
"""
from __future__ import print_function, division
import os
import cv2
import glob
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import predictor_pb
import predictor_ckpt

flags = tf.app.flags

flags.DEFINE_string('weight_path', ' ',
                    'Path to model weight file, .pb or .ckpt is both ok.')
flags.DEFINE_string('positive_img_dir', ' ',
                    'Path to positive images (directory).')
flags.DEFINE_string('negative_img_dir', ' ',
                    'Path to negative images (directory).')
flags.DEFINE_string('output_dir', './rp_test',
                    'Directory to recall precision test result output file.')
flags.DEFINE_string('gpu_device', '0', 'Specify which gpu to be used')

FLAGS = flags.FLAGS


# TODO: 针对不同问题，预测函数需要进行简单修改，下面是两个示例
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
    score = pred_label[1]
    return is_hit, score


def detect2(model, image_path, threshold):
    image_src = cv2.imread(image_path)
    # use different strategy for different size
    shape = image_src.shape
    width = shape[1]
    height = shape[0]
    if width <= 60 or height <= 60:
        is_hit = False  # ignore small image
        pred_label = [1, 0]
    else:
        if len(shape) == 2:
            image = cv2.cvtColor(image_src, cv2.COLOR_GRAY2RGB)
        else:
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

        # predict
        pred_label = model.predict([preprocess_image])[0]
        is_hit = True if pred_label[1] > threshold else False
    score = pred_label[1]
    return is_hit, score



if __name__ == "__main__":    
    weight_path = FLAGS.weight_path
    positive_img_dir = FLAGS.positive_img_dir
    negative_img_dir = FLAGS.negative_img_dir
    output_dir = FLAGS.output_dir
    gpu_device = FLAGS.gpu_device
    threshold = 0.8
    fig_save_name = "precision_recall_curve.jpg" 
    

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
    
    # test positive images
    cnt = 0
    pos_evaluation_list = []
    for root, dirs, files in os.walk(positive_img_dir):
        for filename in files:
            cnt += 1
            if cnt % 1000 == 0:
                print(cnt)
            #print('*', end='')
            image_path = os.path.join(root, filename)
            try:
                is_hit, score = detect2(model, image_path, threshold)
                pos_evaluation_list.append(score)
            except Exception,e:
                print("error occur: {}".format(repr(e)))
                continue
            if(is_hit == False):
                shutil.copy(image_path, recall_mis_save_path + str(score) + '_' + filename)
                FN_count += 1
            else:
                #shutil.copy(image_path, true_detect_save_path + str(score) + '_' + filename)
                TP_count += 1
    
    print('\n------------------------------------')
    # test negative images  
    cnt = 0
    neg_evaluation_list = []
    for root, dirs, files in os.walk(negative_img_dir):
        for filename in files:
            cnt += 1
            if cnt % 1000 == 0:
                print(cnt)
            #print('*', end='')
            image_path = os.path.join(root, filename)
            try:
                is_hit, score = detect2(model, image_path, threshold)
                neg_evaluation_list.append(score)
            except Exception,e:
                print("error occur: {}".format(repr(e)))
                continue
            if(is_hit == True):
                shutil.copy(image_path, false_detect_save_path + str(score) + '_' + filename)
                FP_count += 1
            else:
                TN_count += 1    
    
    precision = TP_count / (TP_count + FP_count) * 100
    print('precision: %f' % precision)
    recall = TP_count / (TP_count + FN_count) * 100
    print('recall: %f' % recall)


    print("\n\n---------------不同置信度下准召计算----------------")
    interval = 0.001
    confidence_list = [c for c in np.arange(0, 1, interval)] # the confidence when model discrimination
    fig_save_path = os.path.join(output_dir, fig_save_name)

    precision_list = [0 for p in range(len(confidence_list))]
    recall_list = [0 for p in range(len(confidence_list))]
    for idx, conf in enumerate(confidence_list):
        threshold = conf
        TP_count = 0.0000001   # 实际为色情，预测为色情的个数
        FP_count = 0.0000001   # 实际为非色情，预测为色情的个数
        FN_count = 0.0000001   # 实际为色情，预测为非色情的个数
        TN_count = 0.0000001   # 实际为非色情，预测为非色情的个数
        for score in pos_evaluation_list:
            if score < threshold:
                FN_count += 1
            else:
                TP_count += 1
        for score in neg_evaluation_list:
            if score >= threshold:
                FP_count += 1
            else:
                TN_count += 1
        precision = TP_count / (TP_count + FP_count) * 100
        recall = TP_count / (TP_count + FN_count) * 100
        precision_list[idx] = precision
        recall_list[idx] = recall

    # save the visualization of the confidence's impact
    plt.plot(precision_list, recall_list, 'b')
    plt.xlabel("precision")
    plt.ylabel("recall")
    plt.savefig(fig_save_path)

    # save precision recall list
    pr_list_path = os.path.join(output_dir, 'pr_list.txt')
    np.savetxt(pr_list_path, (confidence_list, precision_list, recall_list))

    # compute area
    area = 0
    stop_idx = len(precision_list) - 1 
    for idx, precision in enumerate(precision_list):
        if precision < 80:
            continue  # TODO: 可以通过设置筛选条件来观察某一阶段的曲线面积，从而帮助筛选模型
        if idx < stop_idx: 
            dx = precision_list[idx+1] - precision_list[idx] 
            recall = (recall_list[idx+1] + recall_list[idx])/2
            if dx > 0:
                area += recall * dx
    print("area of precision-recall curve: {}".format(area))
