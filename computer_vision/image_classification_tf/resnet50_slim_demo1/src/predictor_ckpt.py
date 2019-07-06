# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:49:09 2018

@author: shirhe-lyh
"""

import os
import tensorflow as tf

# Note: We need to import addditional module to fix the following bug:
# tensorflow.python.framework.errors_impl.NotFoundError: Op type not 
# registered 'ImageProjectiveTransform' in binary running on BJGS-SF-81. 
# Make sure the Op and Kernel are registered in the binary running in this 
# process. Note that if you are loading a saved graph which used ops from 
# tf.contrib, accessing (e.g.) `tf.contrib.resampler` should be done before 
# importing the graph, as contrib ops are lazily registered when the module 
# is first accessed.
import tensorflow.contrib.image

#from timeit import default_timer as timer


class Predictor(object):
    """Classify images to predifined classes."""
    
    def __init__(self,
                 checkpoint_path,
                 gpu_index=None):
        """Constructor.
        
        Args:
            frozen_inference_graph_path: Path to frozen inference graph.
            gpu_index: The GPU index to be used. Default None.
        """
        self._gpu_index = gpu_index
        # Specify which gpu to be used.
        if self._gpu_index is not None:
            if not isinstance(self._gpu_index, str):
                self._gpu_index = str(self._gpu_index)
            os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu_index
        
        #self._graph, self._sess = self._load_model(frozen_inference_graph_path)
        #self._inputs = self._graph.get_tensor_by_name('image_tensor:0')
        #self._logits = self._graph.get_tensor_by_name('logits:0')
        #self._classes = self._graph.get_tensor_by_name('classes:0')

        print('Creating session and loading parameters')
        with tf.Graph().as_default():
            # setting not fully occupied memory, allocated on demand
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            #config.gpu_options.per_process_gpu_memory_fraction = 0.50
            sess = tf.Session(config = config)
            with sess.as_default():
                meta_path = checkpoint_path + '.meta'
                saver = tf.train.import_meta_graph(meta_path)
                #saver.restore(sess, './xxx/xxx.ckpt')
                saver.restore(sess, checkpoint_path)
            
                self._inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
                self._is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
                self._prediction = tf.get_default_graph().get_tensor_by_name('score_list:0')
                
                self._sess = sess
                #pred = sess.run(prediction, feed_dict={inputs: xxx}
        
        
    def predict(self, inputs):
        """Predict prediction tensors from inputs tensor.
        
        Args:
            preprocessed_inputs: A 4D float32 tensor with shape [batch_size, 
                height, width, channels] representing a batch of images.
            
        Returns:
            classes: A 1D integer tensor with shape [batch_size].
        """
        feed_dict = {self._inputs: inputs, self._is_training: False}
        classes = self._sess.run(self._prediction, feed_dict=feed_dict)
        return classes
    
        
