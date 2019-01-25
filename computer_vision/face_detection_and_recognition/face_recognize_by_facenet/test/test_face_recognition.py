"""
Created on Thu Jan 17 14:04:02 2019

test face recognition effect 

@author: zyb_as
"""
import os
import cv2
import shutil
import numpy as np
from scipy import misc
from face_recognize_api import face_recognizer

if __name__ == "__main__":
    test_img_dir = "./test_img"
    base_save_dir = "./temp"
    target_people_name_file = "../template/target_people_names.npy"
    target_people_names = np.load(target_people_name_file)

    # make directory
    for person in target_people_names:
        save_dir = os.path.join(base_save_dir, person)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    print("load model")
    recognizer = face_recognizer()

    print("start detect face")
    index = 0
    for root, dirs, files in os.walk(test_img_dir):
        for file_name in files:
            print("\n" + str(index))
            index += 1
            src_file = os.path.join(root, file_name)
            img = misc.imread(src_file, mode = 'RGB')

            # format error
            if len(img.shape) != 3: 
                print("format error!" + file_name)
                dst_file = os.path.join('./error_img', file_name)
                shutil.move(src_file, dst_file)
                continue

            # detect face and align 
            sign = recognizer.detect_target_face(img)

            # no face detected
            if sign == -1:
                print("no face detected")
                continue
            # no leader detected
            if sign == 0:
                print("no leader detected")
                continue

            # find leader face!
            print("hit") 
            save_dir = os.path.join(base_save_dir, target_people_names[sign])
            dst_file = os.path.join(save_dir, file_name) 
            misc.imsave(dst_file, img)
    print("finish")
            
