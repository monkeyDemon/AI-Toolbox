# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 22:05:47 2019

convert freeze pb model to tensorrt version

tensorRT can speed up the time of model inference

@author: zyb_as
"""
from __future__ import division
import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile

flags = tf.app.flags
flags.DEFINE_string('data_type', None, 'data type of the transformed model')
flags.DEFINE_string('pb_path', None, 'Path to the frozen pb model')
flags.DEFINE_string('tensorrt_pb_save_path', None, 'Path to tensorrt speed up pb model outputs')
flags.DEFINE_string('calibrate_img_dir', None, 'directory of the images used to calibrate the INT8 model, need when data_type=INT8.')
flags.DEFINE_string('cuda_visible_devices', '0', 'Specify which gpu to be used.')
FLAGS = flags.FLAGS


def preprocess_and_inference(image_path, sess, inputs, is_training, prediction):
    image_src = cv2.imread(image_path)
    shape = image_src.shape
    width = shape[1]
    height = shape[0]

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

    inputs_image = preprocess_image.reshape((1, long_edge_size, long_edge_size, 3))
    feed_dict = {inputs: inputs_image, is_training: False}
    classes = sess.run(prediction, feed_dict=feed_dict)
    return classes[0]


def get_graph_definition(graph_path):
  with gfile.FastGFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def convert_tensorrt_speedup_graph(input_graph_path, output_graph_path, data_type="FP32", calibrate_img_dir=""):
    output_node_names = ["score_list"]
    batch_size = 1
    workspace_size = 1<<30
    precision = data_type
 
    trt_graph = trt.create_inference_graph(
       input_graph_def = get_graph_definition(input_graph_path),
       outputs = output_node_names,
       max_batch_size=batch_size,
       max_workspace_size_bytes=workspace_size,
       precision_mode=precision,
       minimum_segment_size=3)

    if data_type == "FP32" or data_type == "FP16":
        # save the new graph transformed by tensorRT
        with gfile.FastGFile(output_graph_path, "wb") as f: 
            f.write(trt_graph.SerializeToString()) #序列化输出
        print("convert tensorrt {} speed up graph finished".format(data_type))
    elif data_type == "INT8":
        calib_graph = tf.Graph()
        with calib_graph.as_default():
            tf.import_graph_def(trt_graph, name='')

        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.50
        sess = tf.Session(graph=calib_graph, config=config)

        inputs = calib_graph.get_tensor_by_name('inputs:0')
        is_training = calib_graph.get_tensor_by_name('is_training:0')
        prediction = calib_graph.get_tensor_by_name('score_list:0')
       
        # calibrate graph
        image_files = glob.glob(os.path.join(calibrate_img_dir, '*.*'))
        for image_path in image_files:
            preprocess_and_inference(image_path, sess, inputs, is_training, prediction)

        infer_graph=trt.calib_graph_to_infer_graph(trt_graph)
        with gfile.FastGFile(output_graph_path, 'wb') as f:
            f.write(infer_graph.SerializeToString())
        print("convert tensorrt {} speed up graph finished".format(data_type))
    else:
        print("data_type error, return")
 




def main(_):
    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_visible_devices
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)

    if FLAGS.data_type not in ["FP32", "FP16", "INT8"]:
        raise RuntimeError("parameter not valid, use: FP32 FP16 or INT8")
    convert_tensorrt_speedup_graph(FLAGS.pb_path, FLAGS.tensorrt_pb_save_path, FLAGS.data_type, FLAGS.calibrate_img_dir)


if __name__ == '__main__':
    tf.app.run()
