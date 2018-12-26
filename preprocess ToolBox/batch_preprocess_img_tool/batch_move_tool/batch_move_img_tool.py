# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 10:13:17 2018

Batch copy/move images tool

@author: zyb_as
"""

import os
import shutil

concern_dir = './chengrenkatong_annotation_result'      
move_dir = './move'          # specify the directory to save the moved images
mode = 'copy'                # 'copy' or 'move'
concern_file_type = ['jpg', 'JPG', 'png', 'PNG', 'bmp', 'BMP', 
                    'jpeg', 'JPEG'] # the file with other types will be ignore

if os.path.exists(move_dir) == False:
    os.mkdir(move_dir)

mode_functions = {'move': shutil.move, 'copy': shutil.copy, }
operation = mode_functions[mode]

for root, dirs, files in os.walk(concern_dir):
    for file_name in files:
        suffix = file_name.split('.')[-1]
        if suffix not in concern_file_type:
            continue
        src_file = os.path.join(root, file_name)
        dst_file = os.path.join(move_dir, file_name)
        operation(src_file, dst_file)