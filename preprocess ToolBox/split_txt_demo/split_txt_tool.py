# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 21:08:37 2018

文本文件切分工具
将一个大的文本文件等分成指定个数的小文件

@author: zyb_as
"""
import time


def calculateRowNum(filename, encoding = 'utf-8'):
    """
    calculate the total number of the txt file
    filename：
    encoding
    """
    cnt = 0
    for line in open(filename, encoding=encoding).readlines():
        cnt += 1
    return cnt



def saveFile(file_lines, file_name, encoding = 'utf-8'):
    """
    save file
    file_lines: the list that record the data
    file_name: the name of the file to save
    encoding
    """
    fout = open(file_name, 'w', encoding = encoding)
    for line in file_lines:
        fout.writelines(line)
    fout.close()



def splitFile(filename, split_num, base_name):
    """
    分割指定的文件
    filename: 待分割的文件
    split_num: 等分的小文件个数
    base_name: 分割后小文件保存的文件基本名
        for example： base_name = 'record', the file after split will be save as 'record1.txt', 'record2.txt'...
    """
    
    # set the encoding of the filename to split
    encode = 'utf-8'
    
    # caculate the total lines number of filename
    print('compute total number:', end="")
    total_num = calculateRowNum(filename, encoding = encode)
    print(total_num)
    
    # the row number of a singal file after split
    singal_row_num = total_num / split_num
    print("the lines number of each sub file: %d" % int(singal_row_num))
    
    # iterate the file
    cnt = 0
    cur_file_id = 1
    cur_file_lines = []
    cur_file_end = singal_row_num
    for line in open(filename, encoding = encode).readlines():
        cnt += 1
        cur_file_lines.append(line)
        if cnt > cur_file_end - 0.1: # considering the situation that cannot be divisible, minus a tiny number
            cur_file_name = base_name + str(cur_file_id) + '.txt'
            print("save file: %s" % cur_file_name)
            saveFile(cur_file_lines, cur_file_name, encoding = encode)
            cur_file_lines = []
            cur_file_id += 1
            cur_file_end += singal_row_num
    

if __name__ == '__main__':
    
    # Example of how to use a split tool
    
    begin = time.time()
    
    # just use this api, split the specified file into 
    file_to_split = 'test.txt'
    split_num = 10
    base_name = './subfile/test'
    splitFile(file_to_split, split_num, base_name)
    
    end = time.time()
    print('finish! use time: %d seconds ' % (end - begin))

