"""
Created on Thu Jan 17 14:04:02 2019

test face detect effect

@author: zyb_as
"""
import os
import cv2
import shutil
from scipy import misc
from face_recognize_api import face_recognizer

if __name__ == "__main__":
    test_img_dir = "./test_img"
    save_dir = "./temp"

    print("load model")
    recognizer = face_recognizer()

    print("start detect face")
    index = 0
    for root, dirs, files in os.walk(test_img_dir):
        for file_name in files:
            print(index)
            index += 1
            src_file = os.path.join(root, file_name)
            img = misc.imread(src_file, mode = 'RGB')

            # format error
            if len(img.shape) != 3: 
                print(file_name)
                dst_file = os.path.join('./error_img', file_name)
                shutil.move(src_file, dst_file)
                continue

            # detect face and align 
            face_array = recognizer.detect_face_and_align(img)

            # no face detected
            if len(face_array) < 1:
                print("miss")
                continue

            print("hit")
            for idx, face in enumerate(face_array):
                dst_file = os.path.join(save_dir, str(idx) + '_' + file_name) 
                misc.imsave(dst_file, face)
    print("finish")
            
