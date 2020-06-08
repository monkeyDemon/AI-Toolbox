# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:24:25 2019

这是一个借助Google图片，搜索指定关键字，获取大量图片的示例程序

google图片搜索页面：
https://www.google.com/imghp?hl=zh-CN&tab=ri&ogbl

本程序的大致思路如下：
程序将会依次使用预先设定的关键字进行图片搜索
借助爬虫来模拟使用Google搜图的过程，达到自动化搜索大量目标图片的目的
搜索的结果将会保存在keyword_search_result中

使用方法如下：
1.提前准备好要搜索的关键字，保存到keyword_list.txt文件中
2.安装chromedriver
教程: https://www.jb51.net/article/162903.htm
查看谷歌浏览器版本命令: chrome://version/
下载链接（需选择对应版本） http://chromedriver.storage.googleapis.com/index.html
3.运行本程序，耐心等待

python version: python3
@author: zyb_as
"""
import os
import cv2
import time
import base64
import traceback
import requests
import numpy as np
from selenium import webdriver


# TODO: needs to be modified according to your actual situation.
chrome_driver_path = 'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe'


home_page = 'https://www.google.com/imghp?hl=zh-CN&tab=ri&ogbl/'
save_dir = 'keyword_search_result'



def prepare_save_dirs(keyword_list):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir_list = []
    for keyword in keyword_list:
        cur_save_dir = os.path.join(save_dir, keyword)
        if not os.path.exists(cur_save_dir):
            os.mkdir(cur_save_dir)
        save_dir_list.append(cur_save_dir)
    return save_dir_list


def search_image_by_keyword(browser, keyword, max_page):
    print("start search images by keyword: {}...".format(keyword))
    
    search_failed = True
    try_num = 0
    while(search_failed):
        if try_num >= 3:
            break
        try:
            browser.get(home_page)
            
            # 拖拽图片到此处或粘贴图片网址
            keyword_textbox = browser.find_element_by_css_selector('#sbtc > div > div.a4bIc > input')
            keyword_textbox.send_keys(keyword)
        
            # 搜索
            search_image_button = browser.find_element_by_css_selector('#sbtc > button > div > span > svg')
            search_image_button.click()
        
            # 等待百度识图结果
            time.sleep(5)
        
            # 切换到当前窗口(好像可有可无)
            browser.current_window_handle
        
            # 测试是否成功
            img_panel = browser.find_element_by_css_selector('#islmp')
            
            # 运行到这里说明模拟使用百度识图功能成功，页面已正常加载
            search_failed = False
        except Exception as e:
            #print("ERROR:" + traceback.format_exc())
            print("ERROR: error when search by keyword.")
        finally:
            try_num += 1

    if search_failed:
        print("***give up current keyword: {}***".format(keyword))
        return []

    
    # 动态加载max_page次页面
    download_page = 0
    print("dynamic loading web page...")
    while download_page < max_page:
        # 模拟向下滑动滚动条动态加载
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # 等待滚动条5s
        time.sleep(5)
        download_page += 1

    # 解析图片面板
    img_panel = browser.find_element_by_css_selector('#islmp')
    
    continue_err_cnt = 0
    cur_img_idx = 1
    base64_str_list = []
    while(continue_err_cnt <= 10):
        try:
            css_selector_str = '#islrg > div.islrc > div:nth-child(' + str(cur_img_idx) + ') > a.wXeWr.islib.nfEiy.mM5pbd > div.bRMDJf.islir > img'
            img_info = img_panel.find_element_by_css_selector(css_selector_str)
            src_str = img_info.get_attribute('src')
            if src_str is None:
                src_str = img_info.get_attribute('data-src')
            base64_str_list.append(src_str)
        except:
            print("err")
            continue_err_cnt += 1
        cur_img_idx += 1
    
    print("totally fing {} images.".format(len(base64_str_list)))
    return base64_str_list



def download_images(base64_str_list, cur_save_dir):
    print("start downloading...")
    print(cur_save_dir)
    for idx, base64_str in enumerate(base64_str_list):
        try:    
            if base64_str.startswith("http"):
                img_url = base64_str
                response = requests.get(img_url, timeout=2)
                imgDataNp = np.frombuffer(response.content, dtype='uint8')
                img = cv2.imdecode(imgDataNp, cv2.IMREAD_UNCHANGED) 
                
                img_name = 'google_search_' + str(idx) + '.jpg'
                save_path = os.path.join(cur_save_dir, img_name)
                #cv2.imwrite(save_path, img)    # 这样保存由于中文路径会出问题
                cv2.imencode(".jpg",img)[1].tofile(save_path)
            else:
                base64_str = base64_str[23:]
                img_str = base64_str.encode('ascii')
                img_decode = base64.b64decode(img_str)
                img_np = np.frombuffer(img_decode, np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                
                img_name = 'google_search_' + str(idx) + '.jpg'
                save_path = os.path.join(cur_save_dir, img_name)
                #cv2.imwrite(save_path, img)
                cv2.imencode(".jpg",img)[1].tofile(save_path)
        except Exception as e:
            #print("ERROR: download img corruption.")
            print("ERROR:" + traceback.format_exc())



# 供外界调用的接口
# 只需提供一个关键字列表，程序便会自动使用Google图片搜索相关图片并下载保存
# keyword_list: 待搜索关键字对应的str列表，例如 ['T-shirt', 'skirt']
# max_page 每个关键字最多下载多少页
def search_imgs(keyword_list, max_page = 30):
    browser = webdriver.Chrome(executable_path=chrome_driver_path)
    browser.set_page_load_timeout(30)

    save_dir_list = prepare_save_dirs(keyword_list)

    for idx, keyword in enumerate(keyword_list):
        print(idx, keyword)
        
        # 利用Google图片搜索指定关键字，拿到url列表
        base64_str_list = search_image_by_keyword(browser, keyword, max_page)
        
        if len(base64_str_list) == 0:
            continue
        
        # 下载图片
        cur_save_dir = save_dir_list[idx]
        download_images(base64_str_list, cur_save_dir)


def load_keywords_list(keyword_path):
    keyword_list = []
    with open(keyword_path, 'r', encoding='utf-8') as reader:
        for line in reader:
            keyword = line.rstrip()
            keyword_list.append(keyword)
    return keyword_list


if __name__ == "__main__":
    # 使用示例：
    keyword_path = 'keywords_list_temp.txt'
    max_page = 10
    
    keyword_list = load_keywords_list(keyword_path)
    search_imgs(keyword_list, max_page)     
    
    