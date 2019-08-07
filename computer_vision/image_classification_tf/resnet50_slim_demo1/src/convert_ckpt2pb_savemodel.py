# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 22:05:47 2019

convert ckpt to savemodel format pb

使用Tensorflow Serving server来部署模型，必须选择SavedModel格式

在转换权重文件格式的同时，本脚本同时改造模型的输入节点使得模型支持base64的string输入
这样便于模型部署后的http请求

@author: zyb_as
"""
import os
import tensorflow as tf
import resnet_v1_50_model

# TODO: import the model you want to convert
from net.resnet import resnet_v1_50_model as model
#from net.resnet import resnet_v1_20_model as model

flags = tf.app.flags
flags.DEFINE_string('action', None, 'command action: print or convert')
flags.DEFINE_string('checkpoint_path', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')
flags.DEFINE_string('pb_model_save_path', None, 'Path to write pb model outputs')
flags.DEFINE_string('cuda_visible_devices', '0', 'Specify which gpu to be used.')
FLAGS = flags.FLAGS


def saveModel(model_path, graph, sess):
    freezing_graph = graph
    images = tf.saved_model.utils.build_tensor_info(freezing_graph.get_tensor_by_name("images:0"))
    scores = tf.saved_model.utils.build_tensor_info(freezing_graph.get_tensor_by_name("score_list:0"))
    classes = tf.saved_model.utils.build_tensor_info(freezing_graph.get_tensor_by_name("classes:0")) 
    
    builder = tf.saved_model.builder.SavedModelBuilder(model_path)
    freezing_graph = graph
    builder.add_meta_graph_and_variables(
      sess,
      ['serve'], # tag
      signature_def_map={
          'serving_default': tf.saved_model.signature_def_utils.build_signature_def(
                      inputs = {'inputs':images},
                      outputs = {'scores':scores, 'classes':classes},
                      method_name = tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
                     )
      },
      clear_devices=True
    )
    builder.save()
    print("Exported SavedModel into %s" % model_path)



def prepareImage(image):
    # 定义预处理方法
    # 这样部署后，只需直接传入base64的string即可直接得到结果
    #img_decoded = tf.image.decode_png(image, channels=3)
    img_decoded = tf.image.decode_jpeg(image, channels=3)

    # 与本demo一致的tensor版本预处理，保持长宽比将长边resize到224，然后padding到224*224
    shape = tf.shape(img_decoded)
    height = tf.to_float(shape[0])
    width = tf.to_float(shape[1])
    scale = tf.cond(tf.greater(height, width),
                    lambda: 224 / height,
                    lambda: 224 / width)
    new_height = tf.to_int32(tf.rint(height * scale))
    new_width = tf.to_int32(tf.rint(width * scale))
    resized_image = tf.image.resize_images(img_decoded, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
    padd_image = tf.image.resize_image_with_crop_or_pad(resized_image, 224, 224)
    padd_image = tf.cast(padd_image, tf.uint8)
    return padd_image

    # 另一个预处理示例，与本demo无关
    #g_img_mean = np.array([[[116.79, 116.28, 103.53]] * 57] * 57)
    #img_centered = tf.subtract(img_decoded, g_img_mean)
    #img_bgr = img_centered[:, :, ::-1]
    #img_bgr = tf.cast(img_bgr,dtype=tf.float32) 


def freeze_graph(input_checkpoint, output_graph_path):
    '''
    :param input_checkpoint:
    :param output_graph_path: savemodel PB模型保存路径
    :return:
    '''
    graph = tf.Graph()
    with graph.as_default():
        train_layers = []

        images = tf.placeholder(tf.string, name="images")
        images_rank = tf.cond(tf.less(tf.rank(images), 1), lambda: tf.expand_dims(images, 0), lambda: images)
        #orignal_inputs = tf.map_fn(prepareImage, images_rank, dtype=tf.float32)
        orignal_inputs = tf.map_fn(prepareImage, images_rank, dtype=tf.uint8)

        is_training = False 
        num_classes = 2
    
        # build model correlation op: logits, classed, loss, acc
        classification_model = model.Model(num_classes=num_classes)
        inputs_dict = classification_model.preprocess(orignal_inputs, is_training)
        predict_dict = classification_model.predict(inputs_dict)
        postprocessed_dict = classification_model.postprocess(predict_dict)
        # the ouput op is postprocesed_dict["logits"], op_name='score_list'

        saver_all = tf.train.Saver(tf.all_variables())
        with tf.Session() as sess:
            saver_all.restore(sess, input_checkpoint)
            saveModel(output_graph_path, graph, sess)
 
 

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
