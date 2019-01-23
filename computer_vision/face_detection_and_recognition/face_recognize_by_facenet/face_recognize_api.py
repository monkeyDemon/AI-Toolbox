"""
Created on Thu Jan 17 14:04:02 2019

face recognize api

@author: zyb_as
"""
import os
import numpy as np
from scipy import misc
import tensorflow as tf

import sys_config
import facenet
from align import detect_face



class face_recognizer(object):
    ''' A recognizer for performing face detection an recognition
    
    base on facenet tensorflow implementation: https://github.com/davidsandberg/facenet
    use mtcnn to do face detection
    use pretrained facenet to do face recognition
    '''
    
    def __init__(self):
        '''load mtcnn and facenet model

        load all needed parameters from sys_config.py
        preload the models and save correspond session
        '''
        self.facenet_weight_dir = sys_config.facenet_weight_dir
        self.face_minsize = sys_config.face_minsize
        self.image_size = sys_config.image_size
        self.margin = sys_config.margin
        self.three_threshold = sys_config.three_threshold 
        self.factor = sys_config.factor 
        self.distance_threshold = sys_config.distance_threshold

        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            # setting not fully occupied memory, allocated on demand
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config = config)
            with sess.as_default():
                # load mtcnn model(do face detection and align)
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)
          
                # load the facenet model(do face recognition)
                facenet.load_model(self.facenet_weight_dir)

                # Get input and output tensors
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                # save session
                self.facenet_session = sess
                
                # load target people feature embeddings
                template_embeddings_file = sys_config.template_face_embeddings_file
                if os.path.exists(template_embeddings_file):
                    self.template_face_embeddings = np.load(template_embeddings_file)
        print("init finish")


    def _compute_distance(self, embedding1, embedding2):
        #distance = np.linalg.norm(feature1 - feature2)
        distance = np.sqrt(np.sum(np.square(np.subtract(embedding1, embedding2))))
        return distance


    def detect_face_and_align(self, img):
        img_size = np.asarray(img.shape)[0:2]
        #face_minsize = int(min(img_size[0], img_size[1]) * 0.15)
        #face_minsize = max(face_minsize, self.face_minsize)
        bounding_boxes, _ = detect_face.detect_face(img, self.face_minsize, 
            self.pnet, self.rnet, self.onet, self.three_threshold, self.factor)
        if len(bounding_boxes) < 1:
            return np.array([])
        
        bb_list = []
        for i in range(len(bounding_boxes)):
            #det = np.squeeze(bounding_boxes[i, 0:4])
            det = bounding_boxes[i, 0:4]
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-self.margin/2, 0)
            bb[1] = np.maximum(det[1]-self.margin/2, 0)
            bb[2] = np.minimum(det[2]+self.margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+self.margin/2, img_size[0])
            bb_list.append(bb)
            
        face_list = []
        for i in range(len(bb_list)):
            bb = bb_list[i]
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            face_list.append(prewhitened)
        face_array = np.stack(face_list)
        return face_array


    def extract_face_embedding(self, faces):
        if len(faces) < 1: 
            print("warning: input parameter faces's length < 1")
            return None

        if isinstance(faces, list):
            faces = np.array(faces)  # convert into ndarray

        with tf.Graph().as_default():    
            with self.facenet_session.as_default():
                # Run forward pass to calculate embeddings
                feed_dict = { self.images_placeholder: faces, self.phase_train_placeholder:False }
                emb_array = self.facenet_session.run(self.embeddings, feed_dict = feed_dict)
        return emb_array


    def detect_target_face(self, img):
        """detect target face

        # Arguments:
            img: ndarray, the input image
                recommend load img by: misc.imread(img_path, mode='RGB')
        # Returns:
            sign: the identification of the detect result
                -1 means no face detected
                0  means common people(no target face detected)
                >0 means hit target face, positive number correspond to the index of target person
        """
        if not isinstance(img, np.ndarray):
            raise RuntimeError("format error! input parameter img is not a numpy.ndarray")
        if len(img.shape) != 3:
            raise RuntimeError("shape error! input parameter img's shape is illegal")
        
        # detect face and align face 
        face_array = self.detect_face_and_align(img)
        
        # can not find a face
        if len(face_array) < 1:
            return -1

        # extract face embedding 
        emb_array = self.extract_face_embedding(face_array)
                
        # compute distance between face embeddings
        for emb in emb_array:
            for idx, template_emb in enumerate(self.template_face_embeddings):
                dist = self._compute_distance(template_emb, emb)
                if dist < self.distance_threshold:
                    return idx + 1  # return the template idx(count from 1)
        return 0 # correspond to the situation: find face but miss hit


