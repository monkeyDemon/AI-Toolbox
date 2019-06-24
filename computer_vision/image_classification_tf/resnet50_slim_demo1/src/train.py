# -*- coding: utf-8 -*-
from __future__ import division
"""
Created on Thu Oct 11 17:21:35 2018

Train a CNN classification model via pretrained ResNet-50 model.

@author: shirhe-lyh
@modified: zyb_as
"""
import os
import sys
import math
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.contrib.slim import nets
from tensorflow.python.ops import control_flow_ops

import resnet_v1_50_model
import data_provider


slim = tf.contrib.slim
flags = tf.app.flags

# TODO: Modify parameter defaults here or specify them directly at call time
flags.DEFINE_string('tf_record_dir', 
                    '/home/ansheng/cv_strategy/porn_detect/cnn_tf/' +
                    'resnet50_slim/tfrecord',
                    'Directory to tfrecord files.')
flags.DEFINE_string('checkpoint_path', 
                    '/home/ansheng/cv_strategy/model_zoo/' +
                    'resnet_v1_50.ckpt', 
                    'Path to pretrained ResNet-50 model.')
flags.DEFINE_boolean('train_from_scratch', True, 
    'train from scratch on imagenet pretrained model or continue training on previous model')
flags.DEFINE_string('label_path',
                    '/home/ansheng/cv_strategy/porn_detect/cnn_tf/' +
                    'classification_by_slim/tfrecord/labels.txt',
                    'Path to label file.')
flags.DEFINE_string('log_dir', './log/train_log', 'Path to log directory.')
flags.DEFINE_string('gpu_device', '0', 'Specify which gpu to be used')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float(
    'learning_rate_decay_factor', 0.1, 'Learning rate decay factor.')  # TODO: not use
flags.DEFINE_integer(
    'num_epochs_per_decay', 3,
    'Number of epochs after which learning rate decays. Note: this flag counts '  # TODO: not use
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')
flags.DEFINE_integer('num_classes', 2, 'Number of classes')
flags.DEFINE_integer('epoch_num', 10, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 48, 'Batch size')

FLAGS = flags.FLAGS

    
'''    
def configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.
    
    Modified from:
        https://github.com/tensorflow/models/blob/master/research/slim/
        train_image_classifier.py
    
    Args:
        num_samples_per_epoch: he number of samples in each epoch of training.
        global_step: The global_step tensor.
        
    Returns:
        A `Tensor` representing the learning rate.
    """
    decay_steps = int(num_samples_per_epoch * FLAGS.num_epochs_per_decay /
                      FLAGS.batch_size)
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
    
    
def get_init_fn():
    """Returns a function run by che chief worker to warm-start the training.
    
    Modified from:
        https://github.com/tensorflow/models/blob/master/research/slim/
        train_image_classifier.py
    
    Note that the init_fn is only run when initializing the model during the 
    very first global step.
    
    Returns:
        An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None
    
    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.logdir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists ' +
            'in %s' % FLAGS.logdir)
        return None
    
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)
    
    variables_to_restore = slim.get_variables_to_restore()
    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=True)
'''






def get_learning_rate(epoch_step, cur_learning_rate, lr_decay_factor, num_epochs_per_decay):
    """get the learning rate.
    """
    if epoch_step == 0:
        return cur_learning_rate

    lr = cur_learning_rate
    if epoch_step % num_epochs_per_decay == 0:
        lr *= lr_decay_factor
        print("learning rate adjustment from {} to {}".format(cur_learning_rate, lr))
    return lr
    

def main(_):
    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_device

    model_ckpt_path = FLAGS.checkpoint_path # Path to the pretrained model
    model_save_dir = FLAGS.log_dir  # Path to the model.ckpt-(num_steps) will be saved
    train_from_scratch = FLAGS.train_from_scratch # train from scratch on imagenet pretrained model or continue training on previous model
    tensorboard_summary_dir = os.path.join(model_save_dir, 'tensorboard_summary')
    tf_record_dir = FLAGS.tf_record_dir
    batch_size = FLAGS.batch_size
    num_classes = FLAGS.num_classes
    epoch_num = FLAGS.epoch_num
    init_learning_rate = FLAGS.learning_rate
    lr_decay_factor = FLAGS.learning_rate_decay_factor
    num_epochs_per_decay = FLAGS.num_epochs_per_decay
    
    # check directory 
    if not tf.gfile.Exists(model_save_dir):
        tf.gfile.MakeDirs(model_save_dir)
    else:
        print("warning! log_dir has exist!")
    tf.gfile.MakeDirs(tensorboard_summary_dir)

    # create placeholders
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')
    is_training = tf.placeholder(tf.bool, name='is_training')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    # build model correlation op: logits, classed, loss, acc
    classification_model = resnet_v1_50_model.Model(num_classes=num_classes)

    #inputs_dict = { 'inputs': inputs,
    #                'is_training': is_training}
    inputs_dict = classification_model.preprocess(inputs, is_training)
    predict_dict = classification_model.predict(inputs_dict)

    #loss_dict = classification_model.loss(predict_dict, labels)
    loss_dict = classification_model.focal_loss(predict_dict, labels)
    loss = loss_dict['loss']

    postprocessed_dict = classification_model.postprocess(predict_dict)
    accuracy = classification_model.accuracy(postprocessed_dict, labels)


    # set training correlation parameters 
    global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #train_step = optimizer.minimize(loss)

    # these three line can fix the low valid accuarcy bug when set is_training=False
    # this bug is cause by use of BN, see for more: https://blog.csdn.net/jiruiYang/article/details/77202674
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([tf.group(*update_ops)]):
        train_step = optimizer.minimize(loss, global_step)
    
    # init Saver to restore model
    if train_from_scratch:
        # if train from imagenet pretrained model, exclude the last classification layer
        checkpoint_exclude_scopes = 'Logits'
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    else:   
        # if continue training, just restore the whole model
        exclusions = []
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
        if not excluded:
            variables_to_restore.append(var)
    saver_restore = tf.train.Saver(var_list=variables_to_restore)

    # init Saver to save model
    saver = tf.train.Saver(tf.global_variables())
    
    init = tf.global_variables_initializer()

    # config and start session
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:
        sess.run(init)
        
        # Load the pretrained checkpoint file xxx.ckpt
        saver_restore.restore(sess, model_ckpt_path)
        
        total_batch_num = 0
        total_best_acc = 0
        cur_lr = init_learning_rate
        for epoch in range(epoch_num):
            ####################
            # training one epoch
            ####################

            print("start training epoch {0}...".format(epoch+1))
            sys.stdout.flush()
            epoch_start_time = time.time()

            # get next train batch op
            train_feature, train_label = data_provider.get_train_data_op(tf_record_dir, batch_size, 1) 

            # get current epoch's learning rate
            cur_lr = get_learning_rate(epoch, cur_lr, lr_decay_factor, num_epochs_per_decay)

            # training batch by batch until one epoch finish
            batch_num = 0
            loss_sum = 0
            acc_sum = 0
            while True: 
                # get a new batch data
                try:
                    images, groundtruth_lists = sess.run([train_feature, train_label]) 
                    #for i in range(len(images)):
                    #    img = images[i]
                    #    img = img + 1
                    #    img = img * 128
                    #    #img = (img * 128) + 128
                    #    img =img.astype(np.uint8) 
                    #    img = Image.fromarray(img)
                    #    img.save("./tmp/" + str(total_batch_num) + "_" + str(i) + ".jpg")
                        
                    total_batch_num += 1
                    batch_num += 1
                except tf.errors.OutOfRangeError:
                    print("epoch {0} training finished.".format(epoch + 1)) 
                    sys.stdout.flush()
                    break

                train_dict = {inputs: images, 
                                labels: groundtruth_lists,
                                is_training: True,
                                learning_rate: cur_lr}
                loss_, acc_, _ = sess.run([loss, accuracy, train_step], feed_dict=train_dict)

                loss_sum += loss_
                loss_ = loss_sum / batch_num
                acc_sum += acc_
                acc_ = acc_sum / batch_num
                
                train_text = 'Step: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(
                    batch_num, loss_, acc_)
                print(train_text)
                sys.stdout.flush()

                #loss_summary.value.add(tag="train_loss", simple_value = loss_)
                #acc_summary.value.add(tag="train_accuary", simple_value = acc_)
                #train_writer.add_summary(loss_summary, total_batch_num)
                #train_writer.add_summary(acc_summary, total_batch_num)

            epoch_end_time = time.time()
            print("total use time: {}s\n".format(int(epoch_end_time - epoch_start_time)))

            ####################
            # validation one epoch
            ####################

            print("start validation, please wait...")
            sys.stdout.flush()

            # get next valid batch op
            valid_feature, valid_label = data_provider.get_valid_data_op(tf_record_dir, batch_size, 1)
            #sess.run(valid_iterator.initializer) # we use make_initializable_iterator, so should be init before use

            # valid batch by batch until validation dataset finish
            batch_num = 0
            loss_sum, loss_mean = 0, 0
            acc_sum, acc_mean = 0, 0
            while True: 
                # get a new batch data
                try:
                    valid_images, valid_groundtruth_lists = sess.run([valid_feature, valid_label]) 
                    batch_num += 1
                except tf.errors.OutOfRangeError:
                    # compute mean accuracy
                    loss_mean = loss_sum / batch_num
                    acc_mean = acc_sum / batch_num
                    print("validation finished. Valid loss:{:.5f}, Valid accuracy:{:.5f}".format(
                        loss_mean, acc_mean)) 
                    sys.stdout.flush()
                    
                    # summary validation accuracy
                    #valid_acc_summary.value.add(tag="valid_accuary", simple_value = acc_mean)
                    #train_writer.add_summary(valid_acc_summary, epoch)
                    break

                valid_dict = {inputs: valid_images, 
                              labels: valid_groundtruth_lists,
                              is_training: False}
                
                valid_loss_, valid_acc_ = sess.run([loss, accuracy], feed_dict=valid_dict)
                loss_sum += valid_loss_
                acc_sum += valid_acc_
                

            if acc_mean > total_best_acc:
                print("epoch {}: val_acc improved from {:.5f} to {:.5f}".format(epoch+1, total_best_acc, acc_mean))
                sys.stdout.flush()
                total_best_acc = acc_mean

                ckpt_name = "resnet50-zyb_v1-epoch{0}.ckpt".format(epoch+1)
                model_save_path = os.path.join(model_save_dir, ckpt_name)
                #saver.save(sess, model_save_path, global_step = total_batch_num) # TODO: global_step?
                saver.save(sess, model_save_path, global_step=global_step) # TODO: global_step?
                print('save mode to {}'.format(model_save_path))
                sys.stdout.flush()
            else:
                print("epoch {}: val_acc did not improve from {}".format(epoch+1, total_best_acc))
                sys.stdout.flush()

            time.sleep(120) # let gpu take a breath
            print("\n\n")
            sys.stdout.flush()



    #dataset = get_record_dataset(FLAGS.train_record_path, num_samples=FLAGS.num_samples, 
    #                             num_classes=FLAGS.num_classes)
    #data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    #image, label = data_provider.get(['image', 'label'])
	# 
    ## get preprocessed batch data
    #preprocessed_inputs, labels = preprocessing.preprocess(image, label, is_training=True, batch_size=FLAGS.batch_size)
    #    
    #cls_model = model.Model(is_training=True, num_classes=FLAGS.num_classes)
    #prediction_dict = cls_model.predict(preprocessed_inputs)
    #loss_dict = cls_model.loss(prediction_dict, labels)
    #loss = loss_dict['loss']
    #postprocessed_dict = cls_model.postprocess(prediction_dict)
    #acc = cls_model.accuracy(postprocessed_dict, labels)
    #tf.summary.scalar('loss', loss)
    #tf.summary.scalar('accuracy', acc)

    #global_step = slim.create_global_step()
    #learning_rate = configure_learning_rate(FLAGS.num_samples, global_step)
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, 
    #                                       momentum=0.9)
#   # optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
    #train_op = slim.learning.create_train_op(loss, optimizer,
    #                                         summarize_gradients=True)
    #tf.summary.scalar('learning_rate', learning_rate)
    #
    #init_fn = get_init_fn()
    #
    #sys.stdout.flush()
    #slim.learning.train(train_op=train_op, logdir=FLAGS.logdir, 
    #                    init_fn=init_fn, number_of_steps=FLAGS.num_steps,
    #                    save_summaries_secs=20,
    #                    save_interval_secs=3600)
    

if __name__ == '__main__':
    tf.app.run()

