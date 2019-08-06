# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:49:09 2018

@author: shirhe-lyh
@modified by: zyb_as
"""

import os
import tensorflow as tf

# if your graph file *.pb is transformed by tensorRT, import tensorRT
# otherwise you will see the error: Op type not registered 'TRTEngineOp' in binary running on ...
import tensorflow.contrib.tensorrt as trt
import tensorflow.contrib.image



class Predictor(object):
    """Classify images to predifined classes."""
    
    def __init__(self,
                 frozen_inference_graph_dir,
                 gpu_index=None):
        """Constructor.
        
        Args:
            frozen_inference_graph_dir: directory of savemodel format inference graph.
            gpu_index: The GPU index to be used. Default None.
        """
        self._gpu_index = gpu_index
        
        self._graph, self._sess = self._load_model(frozen_inference_graph_dir)
        # TODO: need to modify
        self._inputs = self._graph.get_tensor_by_name('images:0')
        #self._is_training = self._graph.get_tensor_by_name('is_training:0')
        self._prediction = self._graph.get_tensor_by_name('score_list:0')


    def _load_model(self, frozen_inference_graph_dir):
        """
            Load a (frozen) Tensorflow model into memory.
        """
        if not tf.gfile.Exists(frozen_inference_graph_dir):
            raise ValueError('`frozen_inference_graph_dir` does not exist.')
            
        # Specify which gpu to be used.
        if self._gpu_index is not None:
            if not isinstance(self._gpu_index, str):
                self._gpu_index = str(self._gpu_index)
            os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu_index

        graph=tf.Graph()

        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.50
        sess = tf.Session(graph=graph, config=config)

        tf.saved_model.loader.load(sess, ["serve"], frozen_inference_graph_dir)
        return graph, sess
            

    def predict(self, inputs):
        """Predict prediction tensors from inputs tensor.
        
        Args:
            preprocessed_inputs: A 4D float32 tensor with shape [batch_size, 
                height, width, channels] representing a batch of images.
            
        Returns:
            classes: A 1D integer tensor with shape [batch_size].
        """
        #feed_dict = {self._inputs: inputs, self._is_training: False}
        feed_dict = {self._inputs: inputs}
        classes = self._sess.run(self._prediction, feed_dict=feed_dict)
        return classes
