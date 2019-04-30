# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:45:22 2019

Get the tag list with the highest frequency

@author: zyb_as
"""

from xml.dom.minidom import parse
import xml.dom.minidom


all_tag_path = 'tmp/all_tags.xml'
frequency_tag_save_path = 'tmp/high_frequency_tags_list.txt'
most_frequency_num = 2000
tags_of_interest = set(["0", "3", "4"])

# 使用minidom解析器打开 XML 文档
DOMTree = xml.dom.minidom.parse(all_tag_path)
collection = DOMTree.documentElement

tags = collection.getElementsByTagName("tag")

cnt_list = []
for tag in tags:
    tag_cnt = int(tag.getAttribute('count'))
    tag_type = tag.getAttribute('type')
    if tag_type in tags_of_interest:
        cnt_list.append(tag_cnt)
    
cnt_list.sort(reverse=True)
frequency_threshold = cnt_list[most_frequency_num]
print("highest frequency tag count: {}".format(cnt_list[0]))
print("frequency threshold: {}".format(frequency_threshold))
print("----------------------------------")

with open(frequency_tag_save_path, 'w') as writer:
    for tag in tags:
        tag_cnt = int(tag.getAttribute('count'))
        tag_type = tag.getAttribute('type')
        if tag_cnt > frequency_threshold and tag_type in tags_of_interest:
            tag_name = tag.getAttribute('name')
            line = tag_type + '\t' + tag_name + '\t' + str(tag_cnt) + '\n'
            writer.write(line)
            #print("name: {}, type: {}".format(tag_name, tag_type))
            #print(tag_name)
print("finish")