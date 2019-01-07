# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 20:57:56 2018

Images deduplication tool

This is a tool to remove duplicate images.

The program can handle large batches of data, and the deduplication is 
based on the dHash algorithm. Just like hash, each image will generate a 
digital fingerprint, and the fingerprints of similar images are similar
(small or even zeros for Hamming distance).

Here is a demo for evaluating image similarity: evaluate_similarity_demo.py

@author: zyb_as
"""

from PIL import Image
import os
import shutil
import warnings


def dhash(image, hash_size = 12):
    # Grayscale and shrink the image in one step.
    image = image.convert('L')
    image = image.resize((hash_size + 1, hash_size), Image.ANTIALIAS)
    
    # Compare adjacent pixels.
    difference = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = image.getpixel((col, row))
            pixel_right = image.getpixel((col + 1, row))
            difference.append(pixel_left > pixel_right)
    
    # Convert the binary array to a hexadecimal string.
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(difference):
        if value:
            decimal_value += 2**(index % 8)
        if (index % 8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
            decimal_value = 0
    
    return ''.join(hex_string)


def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        warnings.warn('string length don\'t match when calculate hamming distance')
    
    hamming_distance = 0
    for idx in range(len(str1)):
        s1 = int(str1[idx], 16) # change hex char to int
        s2 = int(str2[idx], 16)
        xor = bin(s1^s2)        # do xor operation
        if xor == '0b0':
            continue
        for i in range(2, len(xor)):
            if int(xor[i]) is 1:
                hamming_distance += 1
    return hamming_distance


# TODO: set parameters
check_dir = 'the directory to deduplicate'
duplicate_dir = './duplicate_dir' # specify the directory to save the duplicate images
mode = 'equal'          # 'equal' or 'similar'
threshold = 3           # this parameter takes effect only when mode is 'similar'

if os.path.exists(duplicate_dir) == False:
    os.mkdir(duplicate_dir)

if mode == 'equal':
    hash_set = set()
    for root, dirs, files in os.walk(check_dir):
        for file_name in files:
            src_file = os.path.join(root, file_name)
            dst_file = os.path.join(duplicate_dir, file_name)
            img = Image.open(src_file)
            try:
              	dhash_img = dhash(img)
            except:
                print('some error occur!')
                shutil.move(src_file, dst_file)
                continue
            if dhash_img in hash_set:
                print('find duplicate!', file_name)
                shutil.move(src_file, dst_file)
            else:
                hash_set.add(dhash_img)
    print("finish")
elif mode == 'similar':
    hash_list = []
    for root, dirs, files in os.walk(check_dir):
        for file_name in files:
            src_file = os.path.join(root, file_name)
            dst_file = os.path.join(duplicate_dir, file_name)
            img = Image.open(src_file)
            try:
                dhash_cur_img = dhash(img)
            except:
                print('some error occur!')
                shutil.move(src_file, dst_file)
                continue
            duplicate = False
            for tar_dhash in hash_list:
                distance = hamming_distance(dhash_cur_img, tar_dhash)
                if distance < threshold:
                    print('find duplicate!', file_name)
                    duplicate = True
                    shutil.move(src_file, dst_file)
                    break
            if duplicate == False:
                hash_list.append(dhash_cur_img)
    print("finish")
else:
    print('please reset the mode parameter!')

