"""
Created on Sun Feb 24 14:49:10 2019

A demo that use tf.image to do data augmentation

@author: zyb_as
"""
from __future__ import division

import numpy as np
from PIL import Image
import tensorflow as tf

import data_provider

    

def main(_):
    # Specify which gpu to be used
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # our demo will do data augmentation on batch_size * batch_num number of imgs
    # images load from tfrecord file tf_record_path
    # the result will be save in result_save_dir
    tf_record_path = './tfrecord/zyb_v1_mini_train_00000-of-00005.tfrecord'
    result_save_dir = 'data_augmentation_result'
    batch_size = 16
    batch_num = 10
    
    # check directory 
    if not tf.gfile.Exists(result_save_dir):
        tf.gfile.MakeDirs(result_save_dir)
    else:
        tf.gfile.DeleteRecursively(result_save_dir)
        tf.gfile.MakeDirs(result_save_dir)



    # config and start session
    config = tf.ConfigProto() 
    #config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        # get next batch data op
        feature, label = data_provider.get_batch_data_op(tf_record_path, batch_size, 1) 
        for batch_idx in range(batch_num):
            # get a new batch data
            images, groundtruth_lists = sess.run([feature, label]) 
            for i in range(len(images)):
                img = images[i]
                img = img + 1
                img = img * 128
                #img = (img * 128) + 128
                img =img.astype(np.uint8) 
                img = Image.fromarray(img)
                img.save(result_save_dir + '/' + str(batch_idx) 
                    + "_" + str(i) + ".jpg")



if __name__ == '__main__':
    tf.app.run()

