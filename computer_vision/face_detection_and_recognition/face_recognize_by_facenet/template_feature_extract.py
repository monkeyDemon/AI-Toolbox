"""
Created on Thu Jan 17 14:04:02 2019

face template feature extract

before use face_recognize_api.py
run this script to pre-extract the template face feature embeddings

first, put all target people's template image in './template/src_template'
prepare one image for each target person, try to use clear and undisturbed photos.

this script generate the feature embeddings of target people's faces
and save it in './template/template_face_embeddings.npy'.

moreover, this script also generate the target peoples name correspond to
the face embeddings file, save in './template/target_people_names.npy'.

@author: zyb_as
"""

import os
import numpy as np
from scipy import misc
from face_recognize_api import face_recognizer


def extract_template_embedding(template_dir, embedding_save_file, name_save_file):
    print("load model")
    recognizer = face_recognizer()

    leader_name_list = ['common_people']
    embedding_list = []
    for file_name in os.listdir(template_dir):
        print("current leader: " + os.path.splitext(file_name)[0])
        src_file = os.path.join(template_dir, file_name)
        img = misc.imread(src_file, mode='RGB')

        # detect face and align 
        face = recognizer.detect_face_and_align(img)
        if len(face) < 1:
            raise RuntimeError("warning! no face detected")
        if len(face) > 1:
            raise RuntimeError("warning! more than one face ocuur in the template image")

        face_embedding = recognizer.extract_face_embedding(face)
        embedding_list.append(face_embedding[0])
        leader_name_list.append(os.path.splitext(file_name)[0])

    face_embeddings = np.array(embedding_list)
    np.save(embedding_save_file, face_embeddings)

    leader_names = np.array(leader_name_list)
    np.save(name_save_file, leader_names)
            


if __name__ == "__main__":
    src_template_dir = './template/src_template'
    embedding_save_file = './template/template_face_embeddings.npy'
    name_save_file = './template/target_people_names.npy'

    extract_template_embedding(src_template_dir, embedding_save_file, name_save_file)

