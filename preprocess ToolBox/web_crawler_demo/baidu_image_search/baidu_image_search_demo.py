# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:24:25 2019

这是一个借助百度识图功能获取大量相似图片的示例程序

如果你不了解百度识图，不妨尝试使用下：
https://graph.baidu.com/pcpage/index?tpl_from=pc

本程序的大致思路如下：
seed_imgs中放置想要搜索相似图的原始图片
程序将会依次获取seed_imgs中的图片作为搜索种子，
借助爬虫来模拟使用百度识图的过程，达到自动化搜索大量相似图片的目的
搜索的结果将会保存在similar_search_result中

使用方法如下：
1.准备种子图片
收集所有想要用来搜索相似图片的原始图片，放置在seed_imgs中
2.使本地图片可以被url访问
将seed_imgs中的图片做成可供外界访问的url形式，你可以使用任何可能的方法
例如我的解决办法是将这些图片上传到github上，将github作为一个临时的图床使用
根据你制作的图床的url前缀，修改变量base_url
3.安装chromedriver
教程: https://www.jb51.net/article/162903.htm
查看谷歌浏览器版本命令: chrome://version/
下载链接（需选择对应版本） http://chromedriver.storage.googleapis.com/index.html
4.运行本程序，耐心等待

python version: python3.5
@author: zyb_as
"""
import os
import cv2
import time
import requests
import numpy as np
from selenium import webdriver


# TODO: set parameters
# These two parameters needs to be modified according to your actual situation.
chrome_driver_path = 'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe'
base_url = 'https://***/***/seed_imgs/'

home_page = 'https://graph.baidu.com/pcpage/index?tpl_from=pc'
seed_imgs_dir = 'seed_imgs'
save_dir = 'similar_search_result'



def prepare_seed_imgs():
    seed_imgs_url_list = []
    save_dir_list = []
    for file_name in os.listdir(seed_imgs_dir):
        cur_url = base_url + file_name
        seed_imgs_url_list.append(cur_url)
        
        base_file_name = file_name[:-4]
        cur_save_dir = os.path.join(save_dir, base_file_name)
        if not os.path.exists(cur_save_dir):
            os.mkdir(cur_save_dir)
        save_dir_list.append(cur_save_dir)
    return seed_imgs_url_list, save_dir_list


def search_similar_images(browser, image_url, max_page):
    print("start find similar image of {}".format(image_url))
    
    search_failed = True
    try_num = 0
    while(search_failed):
        if try_num >= 3:
            break
        try:
            browser.get(home_page)
            
            # 拖拽图片到此处或粘贴图片网址
            url_upload_textbox = browser.find_element_by_css_selector('#app > div > div.page-banner > div.page-search > div > div > div.graph-search-left > input')
            url_upload_textbox.send_keys(image_url)
        
            # 识图一下
            search_image_button = browser.find_element_by_css_selector('#app > div > div.page-banner > div.page-search > div > div > div.graph-search-center')
            search_image_button.click()
        
            # 等待百度识图结果
            time.sleep(15)
        
            # 切换到当前窗口(好像可有可无)
            browser.current_window_handle
        
            # 测试是否成功
            graph_similar = browser.find_element_by_class_name('graph-similar-list')
            
            # 运行到这里说明模拟使用百度识图功能成功，页面已正常加载
            search_failed = False
        except Exception as e:
            #print("ERROR:" + traceback.format_exc())
            print("ERROR: error when request baidu image search.")
        finally:
            try_num += 1

    if search_failed:
        print("give up current image")
        return []

    # 动态加载max_page次页面
    download_page = 0
    print("dynamic loading web page...")
    while download_page < max_page:
        # 模拟向下滑动滚动条动态加载
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # 等待滚动条10s
        time.sleep(10)
        download_page += 1

    # 解析页面中的所有url
    graph_similar = browser.find_element_by_class_name('graph-similar-list')
    left_column = graph_similar.find_element_by_css_selector('div > div:nth-child(1)')
    middle_column = graph_similar.find_element_by_css_selector('div > div:nth-child(2)')
    right_column = graph_similar.find_element_by_css_selector('div > div:nth-child(3)')

    left_column_imgs = left_column.find_elements_by_tag_name('a')
    middle_column_imgs = middle_column.find_elements_by_tag_name('a')
    right_column_imgs = right_column.find_elements_by_tag_name('a')
    
    url_list = []
    for img_box in left_column_imgs:
        img_url = img_box.find_element_by_tag_name('img').get_attribute('src')
        url_list.append(img_url)
    for img_box in middle_column_imgs:
        img_url = img_box.find_element_by_tag_name('img').get_attribute('src')
        url_list.append(img_url)
    for img_box in right_column_imgs:
        img_url = img_box.find_element_by_tag_name('img').get_attribute('src')
        url_list.append(img_url)
        
    total_imgs_num = len(left_column_imgs) + len(middle_column_imgs) + len(right_column_imgs)
    print("totally fing {} images.".format(total_imgs_num))
    return url_list


def download_search_images(url_list, cur_save_dir):
    print("start downloading...")
    for img_url in url_list:
        try:
            response = requests.get(img_url, timeout=1)
        except Exception as e:
            print("ERROR: download img timeout.")
        
        try:
            #imgDataNp = np.fromstring(response.content, dtype='uint8')
            imgDataNp = np.frombuffer(response.content, dtype='uint8')
            img = cv2.imdecode(imgDataNp, cv2.IMREAD_UNCHANGED) 
            
            img_name = img_url.split('/')[-1]
            save_path = os.path.join(cur_save_dir, img_name)
            cv2.imwrite(save_path, img)
        except Exception as e:
            print("ERROR: download img corruption.")



if __name__ == "__main__":
    browser = webdriver.Chrome(executable_path=chrome_driver_path)
    browser.set_page_load_timeout(30)

    seed_imgs_url_list, save_dir_list = prepare_seed_imgs()
    
    for idx, seed_url in enumerate(seed_imgs_url_list):
        print(idx)
        
        # 获取百度识图结果
        url_list = search_similar_images(browser, seed_url, max_page=30)
        
        if len(url_list) == 0:
            continue
        
        # 下载图片
        cur_save_dir = save_dir_list[idx]
        download_search_images(url_list, cur_save_dir)
        