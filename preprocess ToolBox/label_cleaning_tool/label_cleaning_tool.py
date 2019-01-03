# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 10:54:07 2018

Image label cleaning tool

I think you often suffer such a situation too:
    
You have a large amount of rough-labeled image data, and you don't have
enough labor to complete the labeling of these data.

This is really helpless!

We often use the following workflow to make the data labeled more accurate:

step1: Use the rough-labeled data to train a convolutional neural network.
You can pick a CNN template directly from here.
https://github.com/monkeyDemon/AI-Toolbox/tree/master/keras/image_classification

step2: Use the trained model as an error annotation detector.
If the label of an image is labeled as A category and the model considers it not
belong to A category, recorded this image.
[Our 'Image label cleaning tool' is just a demo to simplify this step]

step3: Manual review
Manually review images of all labeled categories that are objectionable to the model.
You can use the data annotation tool in out AI-Toolbox
https://github.com/monkeyDemon/AI-Toolbox/tree/master/preprocess%20ToolBox/data_annotation_tool

step4: Repeate the step1 to step3
Repeat until the data labeling accuracy meets the requirements.

@author: zyb_as
"""

import os
import shutil
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

class CNN_Evaluator(object):
    """ 
    CNN model evaluator
    A packaged evaluator that makes it easier to call models for prediction
    """
    def __init__(self, weight_path, input_size):
        """init evaluator
        
        # Arguments:
            weight_path: the file of the CNN weight file
            input_size: the input size of the CNN you use
        """
        # load model
        self.model = load_model(weight_path)
        # set input size
        self.input_size = input_size
    
    
    def predictSingleImage(self, image_path):
        """ predict the specified image
        
        # Arguments:
            image_path: string, the file path of the image to predict
        """
        # preprocess
        image = self._preprocess_image(image_path)
        # predict
        y_predict = self.model.predict(image)
        return y_predict[0]

    
    def _preprocess_image(self, image_path):
        """ preprocess the specified image
        
        The preprocess operation before training a CNN model may be different.
        So, you should modify this function by your actual demand
        
        # Arguments:
            image_path: the path of the image need to do preprocess
        # Returns:
            the image after preprocess(type is ndarray)
        """
        img = image.load_img(image_path, target_size=(self.input_size[0],self.input_size[1]))

        # TODO: perform the same pre-processing operations with training CNN
        x = image.img_to_array(img)/255.

        x = np.expand_dims(x, axis = 0)
        return x 

        
    
if __name__ == "__main__":
    # TODO: set model parameters
    weight_file = "your CNN weights file trained by Keras"
    input_size = (224, 224, 3)
    evaluator = CNN_Evaluator(weight_file, input_size)
    
    # TODO: set label cleaning parameters:
    dataset_dir = 'the dataset you want to clean'
    save_dir = './doubt_imgs'  # move the doubt images to save_dir
    correct_category = 0  # specify the correct category of images in dataset_dir
    confidence = 0.5      # the confidence when model discrimination
    
    if not os.path.exists(dataset_dir):
        raise RuntimeError("The specified dataset_dir doesn't exist!")
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    for file_name in os.listdir(dataset_dir):
        src_file = os.path.join(dataset_dir, file_name)
        dst_file = os.path.join(save_dir, file_name)
        # predict current image
        predict = evaluator.predictSingleImage(src_file)
        if predict[correct_category] < confidence:
            # model think this image may not belongs to current category
            print("found suspicious image", file_name)
            shutil.move(src_file, dst_file)
    print("finish")

