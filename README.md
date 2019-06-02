# AI-Toolbox

## Introduction

Algorithm Engineer Toolbox, for the purpose of quickly iterating new ideas.

This repository is a flexible toolbox that contains many independent algorithm demos. It is designed to improve the efficiency of the work, choose the algorithm demo you interested in, and slightly modify it to quickly verify your ideas.

At present, the project is still in the initial stage, mainly covering algorithms
 in the field of image processing and computer vision, later will cover algorithms in other fields such as natural language processing and machine learning. continue to update...

## AI-Toolbox visualization

Here is a visualization of some of the algorithms involved in the toolbox.

Quickly browse through the toolbox for the algorithm that you interested in.

All sample images used are for research purposes only and will be deleted immediately if infringement occurs. Have fun~

 | digital image processing |  |  |  |  |
  | ------ | ------ | ------ | ------ | ------ |
 | FFT for image encryption | SIFT for match | selective search |None |None |
 | ![fft](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/algorithm_image/fft.png) | ![sift](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/algorithm_image/sift.png) | ![selectivesearch](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/algorithm_image/selectivesearch.png) |![Alt](https://avatar.csdn.net/7/7/B/1_ralf_hx163com.jpg) |![Alt](https://avatar.csdn.net/7/7/B/1_ralf_hx163com.jpg) |
| **classification CNN** |  |  |  |  |
| VGG | GoogLeNet | ResNet | DenseNet |  |
| ![VGG](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/readme_image/classification_Net/VGG.png) | ![GoogLeNet](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/readme_image/classification_Net/inception_v1_all.png) | ![ResNet](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/readme_image/classification_Net/resnet_struct2.png) | ![DenseNet](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/readme_image/classification_Net/denset_1.png) | ![Alt](https://avatar.csdn.net/7/7/B/1_ralf_hx163com.jpg) |
| **Applications of CNN** |  |  |  |  
| Face Recognition | None | None | None | None |
| ![face_recognize](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/readme_image/face_recognize/chenduxiu.png) | ![Alt](https://avatar.csdn.net/7/7/B/1_ralf_hx163com.jpg) | ![Alt](https://avatar.csdn.net/7/7/B/1_ralf_hx163com.jpg) | ![Alt](https://avatar.csdn.net/7/7/B/1_ralf_hx163com.jpg) | ![Alt](https://avatar.csdn.net/7/7/B/1_ralf_hx163com.jpg) |
| **preprocess Tools** |  |  |  |  
| KeyFrame Extract | Data Augmentation | Data Annotation | None | None |
| ![KeyFrame Extract](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/algorithm_image/resize_keyframe_tool.png) | ![Data Augmentation](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/algorithm_image/resize_data_augmentation.png) | ![Data Annotation](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/algorithm_image/resize_image_select_tool_2.png) | ![Alt](https://avatar.csdn.net/7/7/B/1_ralf_hx163com.jpg) | ![Alt](https://avatar.csdn.net/7/7/B/1_ralf_hx163com.jpg) |


 ## Algorithm checklist

Is this AI-Toolbox useful for you?

Here is a list shows which algorithms are included in our toolbox, you can check.

- CNN
  * basic exercise
    + [parameter adjustment exercise(keras)](https://github.com/monkeyDemon/AI-Toolbox/tree/master/keras/image_classification/parameterAdjusting_practice)
  * classification
    + VGG
    + GoogleNet
    + [ResNet(Keras)](https://github.com/monkeyDemon/AI-Toolbox/tree/master/keras/image_classification/resNet_template),  [ResNet(Tensorflow)](https://github.com/monkeyDemon/AI-Toolbox/tree/master/computer_vision/image_classification_tf/resnet50_slim_demo1)
    + [DenseNet](https://github.com/monkeyDemon/AI-Toolbox/tree/master/keras/image_classification/denseNet_template)
    + [multi label classification](https://github.com/monkeyDemon/AI-Toolbox/tree/master/computer_vision/image_classification_keras/multi_label_classification)
  * object detection
    + RCNN
    + YOLO
  * face recognition
    + face detection(base on Dlab)
    + [face recognition](https://github.com/monkeyDemon/AI-Toolbox/tree/master/computer_vision/face_detection_and_recognition/face_recognize_by_facenet)(base on FaceNet)
- digital image processing
  * feature point detection
    + [SIFT](https://github.com/monkeyDemon/AI-Toolbox/tree/master/image%20processing%20ToolBox/feature_point_detection/sift)
  * frequency domain method
    + [FFT](https://github.com/monkeyDemon/AI-Toolbox/tree/master/image%20processing%20ToolBox/frequency_domain_method)
  * image segmentation
    + [selective search](https://github.com/monkeyDemon/AI-Toolbox/tree/master/image%20processing%20ToolBox/image_segmentation/selective_search)
- preprocess tools
  * [keyFrames extract tool](https://github.com/monkeyDemon/AI-Toolbox/tree/master/preprocess%20ToolBox/keyframes_extract_tool)
  * [data augmentation tool](https://github.com/monkeyDemon/AI-Toolbox/tree/master/preprocess%20ToolBox/data_augmentation_tool/tf_data_augmentation_tool)
  * [data annotation tool](https://github.com/monkeyDemon/AI-Toolbox/tree/master/preprocess%20ToolBox/data_annotation_tool)
