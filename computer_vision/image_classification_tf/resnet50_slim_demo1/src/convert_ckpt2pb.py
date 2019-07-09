# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 22:05:47 2019

freeze model
convert ckpt to pb

转换步骤：
-通过传入 CKPT 模型的路径得到模型的图和变量数据
-通过 import_meta_graph 导入模型中的图
-通过 saver.restore 从模型中恢复图中各个变量的数据
-通过 graph_util.convert_variables_to_constants 将模型持久化

@author: zyb_as
"""
import os
import tensorflow as tf
from tensorflow.python.framework import graph_util

flags = tf.app.flags
flags.DEFINE_string('action', None, 'command action: print or convert')
flags.DEFINE_string('checkpoint_path', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')
flags.DEFINE_string('pb_model_save_path', None, 'Path to write pb model outputs')
flags.DEFINE_string('cuda_visible_devices', '0', 'Specify which gpu to be used.')
FLAGS = flags.FLAGS



def freeze_graph(input_checkpoint, output_graph_path):
    '''
    :param input_checkpoint:
    :param output_graph_path: PB模型保存路径
    :return:
    '''
 
    # TODO: modify output op name
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    # use op name not tensor name(use "Logits/SpatialSqueeze" not "Logits/SpatialSqueeze:0")
    output_node_names = "score_list"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph_path, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点
 
 

def print_op_names(checkpoint_path):
    '''
    print all the operation names in the model
    
    convert ckpt to pb need you know the op name of input and output op
    so you can use this function to find out
    '''
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        sess = tf.Session(config = config)
        with sess.as_default():
            meta_path = checkpoint_path + '.meta'
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, checkpoint_path)
    
            op_list = sess.graph.get_operations()
            for op in op_list:
                print(op.name)
                print(op.values())



def main(_):
    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_visible_devices

    if FLAGS.action == 'print':
        print_op_names(FLAGS.checkpoint_path)
    elif FLAGS.action == 'convert':
        freeze_graph(FLAGS.checkpoint_path, FLAGS.pb_model_save_path)
    else:
        raise RuntimeError("parameter action error! use print or convert")


if __name__ == '__main__':
    tf.app.run()
