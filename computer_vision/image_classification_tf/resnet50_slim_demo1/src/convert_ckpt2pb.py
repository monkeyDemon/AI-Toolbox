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

import tensorflow as tf
from tensorflow.python.framework import graph_util
 

'''
def freeze_graph_test(pb_path, image_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
 
            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
            input_image_tensor = sess.graph.get_tensor_by_name("input:0")
            input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
            input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")
 
            # 定义输出的张量名称
            output_tensor_name = sess.graph.get_tensor_by_name("InceptionV3/Logits/SpatialSqueeze:0")
 
            # 读取测试图片
            im=read_image(image_path,resize_height,resize_width,normalization=True)
            im=im[np.newaxis,:]
            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
            # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})
            out=sess.run(output_tensor_name, feed_dict={input_image_tensor: im,
                                                        input_keep_prob_tensor:1.0,
                                                        input_is_training_tensor:False})
            print("out:{}".format(out))
            score = tf.nn.softmax(out, name='pre')
            class_id = tf.argmax(score, 1)
            print("pre class_id:{}".format(sess.run(class_id)))
 '''


def freeze_graph(input_checkpoint, output_graph_path):
    '''
    :param input_checkpoint:
    :param output_graph_path: PB模型保存路径
    :return:
    '''
 
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    # TODO: modify output op name
    output_node_names = "InceptionV3/Logits/SpatialSqueeze"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,# 等于:sess.graph_def
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
