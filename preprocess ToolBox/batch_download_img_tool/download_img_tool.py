# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:03:52 2018

A image download file

Read the text file which records url list of all pictures and download it in batches

In the work, in order to efficiently transfer a large amount of image data, 
we usually only transmite a text file that records image information according 
to a certain structure.

This tool shows how to download images by reading such a text file.
The structure of the file needs to be similar to the 'url_record'

@author: zyb
"""
import os
import requests

# TODO: modify these two parameter when use
download_path = "download/"        # download images save path
urlfile = "url_record"            # url record txt file


def get_img_by_url(url):
    """
    download and write a image by url
    url example: http://img.zuoyebang.cc/zyb_2142d51950cdb170b7729320eca2a545.jpg
    """
    #response = requests.get(url,timeout=0.1,proxies=proxies)
    pic_name = url.split('/')[-1]
    response = requests.get(url,timeout=1)
    #print(pic_name,'  OK')
    with open(download_path + pic_name, 'wb') as fd:
        for chunk in response.iter_content(128):  # iter_content saving while downloading
            fd.write(chunk)    


if os.path.exists(download_path) == False:
    os.mkdir(download_path)
failure_count = 0
failed_urls = []
cnt = 0
for line in open(urlfile).readlines():
    cnt += 1

    item = line.split('\t')
    url = item[-1]  # TODO: these two line may be need to modify(depend on the structure of txt file)
    if len(item) >= 3:  # TODO: ditto
        try:
            print(url)
            get_img_by_url(url.rstrip())  # rstrip() delete the specified character at the end of the string string (default is a space)
        except:
            failure_count += 1
            failed_urls.append(url.rstrip())
    
print(failure_count)
print(failed_urls)
