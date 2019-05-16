# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:21:12 2018

resnet-50 模型定义函数

@author: zyb_as 
"""

import tensorflow as tf

from tensorflow.contrib.slim import nets


slim = tf.contrib.slim
    
        
class Model(object):
    """xxx definition."""
    
    def __init__(self, num_classes):
        """Constructor.
        
        Args:
            num_classes: Number of classes.
        """
        self._num_classes = num_classes
        
    @property
    def num_classes(self):
        return self._num_classes
    
    def preprocess(self, inputs, is_training):
        ''' preprocess
        here, we assuming that the size of the input image is correct 
        '''
        inputs = tf.to_float(inputs)
        inputs = tf.subtract(inputs, 128)
        inputs = tf.div(inputs, 128)
        preprocessed_inputs_dict = {}
        preprocessed_inputs_dict['inputs'] = inputs
        preprocessed_inputs_dict['is_training'] = is_training
        return preprocessed_inputs_dict
        

    def predict(self, inputs_dict):
        """Predict prediction tensors from inputs tensor.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            inputs_dict: a dictionary of inputs, include: inputs, is_training
            inputs: A float32 placeholder or tensor with shape [batch_size, height, width, num_channels] 
                    representing a batch of images.
                    tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='inputs')
            is_training: tf.placeholder(tf.bool, name='is_training')
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                    passed to the Loss or Postprocess functions.
        """
        input_images = inputs_dict['inputs']
        is_training = inputs_dict['is_training'] 
        with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
            net, endpoints = nets.resnet_v1.resnet_v1_50(
                input_images,
                num_classes=None,
                is_training=is_training)

        with tf.variable_scope('Logits'):
            # the last average pooling layer makes the resnet50 ouput tensor with shape [None, 1, 1, 2048]
            # use tf.squeeze to flatten it into [None, 2048]
            net = tf.squeeze(net, axis=[1, 2])
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='scope')
            logits = slim.fully_connected(net, num_outputs=self.num_classes,
                                      activation_fn=None, scope='fc')

        prediction_dict = {'logits': logits}
        return prediction_dict
    

    def postprocess(self, prediction_dict):
        """Convert predicted output tensors to final forms.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations
                of specified models.
                
        Returns:
            A dictionary containing the postprocessed results.
        """
        logits = prediction_dict['logits']
        logits = tf.nn.softmax(logits, name='score_list')
        classes = tf.argmax(logits, axis=1, name='classes')

        postprocessed_dict = {'logits': logits,
                              'classes': classes}
        return postprocessed_dict
    

    def loss(self, prediction_dict, groundtruth_lists):
        """Compute scalar loss tensors with respect to provided groundtruth.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            groundtruth_lists_dict: A dict of tensors holding groundtruth
                information, with one entry for each image in the batch.
                
        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.
        """
        logits = prediction_dict['logits']

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = groundtruth_lists, 
                logits = logits)
        loss = tf.reduce_mean(losses)
        loss_dict = {'loss': loss}
        return loss_dict


    def focal_loss(self, prediction_dict, groundtruth_lists):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Keyword Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]
            gamma {float} -- (default: {2.0})
            alpha {constant tensor} -- each value respect to each category's weight

        Returns:
            A dictionary mapping strings (loss names) to loss values.
        """
        epsilon = 1.e-9
        gamma = 2.0
        alpha = tf.constant([[1], [1]], dtype=tf.float32)
        #alpha = tf.constant([1, 1], dtype=tf.float32)

        #y_true = tf.convert_to_tensor(groundtruth_lists, tf.float32)
        y_true = self._convert_one_hot(groundtruth_lists)
        y_true = tf.cast(y_true, tf.float32)

        logits = prediction_dict['logits']
        y_pred = tf.nn.softmax(logits, name='softmax_for_focalloss')
        #y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        #fl = tf.multiply(alpha, tf.multiply(weight, ce))
        #reduced_fl = tf.reduce_max(fl, axis=1)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        loss_dict = {'loss': loss}
        return loss_dict


    def accuracy(self, postprocessed_dict, groundtruth_lists):
        """Calculate accuracy.
        
        Args:
            postprocessed_dict: A dictionary containing the postprocessed 
                results
            groundtruth_lists: A dict of tensors holding groundtruth
                information, with one entry for each image in the batch.
                
        Returns:
            accuracy: The scalar accuracy.
        """
        classes = postprocessed_dict['classes']
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.cast(classes, dtype=tf.int32), groundtruth_lists), dtype=tf.float32))
        return accuracy
    
    def _convert_one_hot(self, y_label):
        """ convert y_label to ont hot encode
        for example convert [0, 2, 1, 3] to
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]]
        """
        batch_size = tf.size(y_label)
        labels_1 = tf.expand_dims(y_label, 1)
        indices = tf.expand_dims(tf.range(0,batch_size,1), 1)
        concated = tf.concat([indices, labels_1], 1)
        onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, self.num_classes]), 1.0, 0.0)
        return onehot_labels


