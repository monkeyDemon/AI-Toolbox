# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:45:22 2019

Get the tag list with the highest frequency

@author: zyb_as
"""

from xml.dom.minidom import parse
import xml.dom.minidom


all_tag_path = 'tmp/all_tags.xml'
most_frequency_num = 100

# 使用minidom解析器打开 XML 文档
DOMTree = xml.dom.minidom.parse(all_tag_path)
collection = DOMTree.documentElement


tags = collection.getElementsByTagName("tag")

cnt_list = []
for tag in tags:
    tag_cnt = int(tag.getAttribute('count'))
    cnt_list.append(tag_cnt)
    
cnt_list.sort(reverse=True)
frequency_threshold = cnt_list[most_frequency_num]
print("highest frequency tag count: {}".format(cnt_list[0]))
print("frequency threshold: {}".format(frequency_threshold))
print("----------------------------------")

for tag in tags:
    tag_cnt = int(tag.getAttribute('count'))
    if tag_cnt > frequency_threshold:
        tag_type = tag.getAttribute('type')
        tag_name = tag.getAttribute('name')
        #print("name: {}, type: {}".format(tag_name, tag_type))
        print(tag_name)
