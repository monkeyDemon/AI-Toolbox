# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 22:05:47 2019

convert freeze pb model to tensorrt version

tensorRT can speed up the time of model inference

@author: zyb_as
"""
import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile

flags = tf.app.flags
flags.DEFINE_string('data_type', None, 'data type of the transformed model')
flags.DEFINE_string('pb_path', None, 'Path to the frozen pb model')
flags.DEFINE_string('tensorrt_pb_save_path', None, 'Path to tensorrt speed up pb model outputs')
flags.DEFINE_string('cuda_visible_devices', '0', 'Specify which gpu to be used.')
FLAGS = flags.FLAGS


def get_graph_definition(graph_path):
  #with tf.gfile.GFile(graph_path, 'rb') as f:
  with gfile.FastGFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def convert_tensorrt_speedup_graph(input_graph_path, output_graph_path, data_type="FP32"):
 
    output_node_names = ["score_list"]
    batch_size = 1
    workspace_size = 1<<30
    precision = data_type
 
    #with tf.Session() as sess:
    #    # get input graph path

    trt_graph = trt.create_inference_graph(
       input_graph_def = get_graph_definition(input_graph_path),
       outputs = output_node_names,
       max_batch_size=batch_size,
       max_workspace_size_bytes=workspace_size,
       precision_mode=precision,
       minimum_segment_size=3)

    # save the new graph transformed by tensorRT
    with gfile.FastGFile(output_graph_path, "wb") as f: 
        f.write(trt_graph.SerializeToString()) #序列化输出
 
 


def main(_):
    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_visible_devices
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)

    if FLAGS.data_type not in ["FP32", "FP16", "INT8"]:
        raise RuntimeError("parameter not valid, use: FP32 FP16 or INT8")
    convert_tensorrt_speedup_graph(FLAGS.pb_path, FLAGS.tensorrt_pb_save_path, FLAGS.data_type)


if __name__ == '__main__':
    tf.app.run()
