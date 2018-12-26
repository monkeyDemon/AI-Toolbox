# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 16:39:26 2018

A demo for evaluating image similarity

@author: zyb_as
"""

from PIL import Image
import warnings

def dhash(image, hash_size = 8):
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

print('-----------------------------')
# from utility import dhash, hamming_distance
img1 = Image.open('./deduplicate_test_img/test1.jpg')
img2 = Image.open('./deduplicate_test_img/test2.jpg')
dhash_img1 = dhash(img1)
print(dhash_img1)
dhash_img2 = dhash(img2)
print(dhash_img2)
print(dhash_img1 == dhash_img2)
dis = hamming_distance(dhash_img1, dhash_img2)
print(dis)



# -------------------------------------------------------------------------
# A demo show that why using MD5 to remove duplication is not a good choice
# -------------------------------------------------------------------------
import hashlib
print('\n-----------------------------')

# this is a calling practice of md5 algorithm
# Calculating the hash value of a string.
test_str = 'Cherish life, I use python.'
test_str = test_str.encode('utf-8')
md5 = hashlib.md5()
md5.update(test_str)
md5_str = md5.hexdigest()
print(md5_str)

# Loading an image file into memory and calculating it's hash value.
image_file = open('./deduplicate_test_img/test1_1.jpg', 'rb').read()
md5 = hashlib.md5()
md5.update(image_file)
md5_img = md5.hexdigest()
print(md5_img)

# Loading a very similar image into memory and calculating it's hash value.
image_file2 = open('./deduplicate_test_img/test1_2.jpg', 'rb').read()
md5 = hashlib.md5()
md5.update(image_file2)
md5_img = md5.hexdigest()
print(md5_img)

# It can be seen that due to the avalanche effect of the cryptographic hash 
# algorithm, although the two pictures are very similar, the resulting md5 
# results are completely different.