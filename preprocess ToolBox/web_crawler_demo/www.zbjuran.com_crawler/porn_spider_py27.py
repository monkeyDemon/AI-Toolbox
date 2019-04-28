# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 22:02:02 2019

A web crawler downloading sexy beauty images

This script is built to crawl the data needed for the porn detection model.


***
WARNING!
As an anti-cheating algorithmic engineer, I use this code to train pornographic
models to keep children away from adult information harassment. 

In order to comply with the relevant laws of China, only part of the code will 
be shared here.

As potential pornographic information may appear, please ensure that your 
actions are legitimate! Don't spread bad information after downloading!

作为一个反作弊算法工程师，我使用本代码来训练色情模型，让孩子远离成人信息的骚扰。
由于可能出现潜在的色情信息，请确保你的行为合法！下载后不要传播不良信息！
***

python version: Python 2.7

@author: zyb_as
"""

from __future__ import print_function

import bs4
import requests
import time
import os
import random


class PornSpiderBase(object):
    """
    色情网站爬虫基类
    封装一些网站无关的可复用函数
    """
    def __init__(self, home_page):
        self.homepage_url = home_page
        # 如果想要爬取的网站有反扒措施，需要配置UserAgent和Headers
        self.uapools = [
            "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
            "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
            "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
            "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
            "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
            "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
            "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
            "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
            "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
            "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
            "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
            "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
            "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",]
        self.headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                   'Accept-Encoding': 'gzip, deflate',
                   'Accept-Language': 'zh-CN,zh;q=0.8',
                   'Cache-Control': 'no-cache',
                   'Connection': 'keep-alive',
                   'Cookie': 'UM_distinctid=15fa02251e679e-05c01fdf7965e7-5848211c-144000-15fa02251e7800; bdshare_firstime=1510220189357; CNZZDATA1263415983=1653134122-1510216223-null%7C1510216223; CNZZDATA3866066=cnzz_eid%3D376479854-1494676185-%26ntime%3D1494676185; Hm_lvt_9a737a8572f89206db6e9c301695b55a=1510220189; Hm_lpvt_9a737a8572f89206db6e9c301695b55a=1510220990',
                   'Host': 'img1.mm131.me',
                   'Pragma': 'no-cache',
                   'Referer': home_page,
                   'User-Agent': random.choice(self.uapools)}
        self.url_save_file = "page_url_list.txt"
    
    def getDefaultHeader(self):
        # get default header
        return self.headers
    
    
    def getHeader(self, parent_url):
        # build header
        self.headers['Referer'] = parent_url
        self.headers['User-Agent'] = random.choice(self.uapools)


    def loadUrlList(self, txt_path):
        """载入url列表
        
        通用方法，从保存的txt中载入记录所有url的列表
        比如，有的章节Section的页数很多，只爬一次页面列表，这样可以避免时间的浪费
        
        Arguments:
            txt_path: 存储url列表的txt文件名+路径
        """
        file=open(txt_path, 'r')
        url_list = file.readlines()
        file.close()
        result_list = []
        for url in url_list:
            result_list.append(url.replace("\n", ""))
        return result_list
    
    
    def saveUrlList(self, txt_path, url_list):
        """保存url列表
        
        Arguments:
            txt_path: 用于保存url列表的txt文件名+路径
            url_list: 待保存的url列表
        """
        file= open(txt_path, 'w')  
        for url in url_list:
            file.write(url)
            file.write("\n")
        file.close()


    def testLoadJson(self, json_path):
        json_path = self.homepage_url + json_path
        json = requests.get(json_path).text
        json = json.split('$')[1]
        print(json)



class PornSpider3(PornSpiderBase):
    """色情爬虫
    
    目标网站："https://www.zbjuran.com/mei"
    这是个美女图像网站，类型为正常或性感
    """
    def __init__(self, home_page):
        PornSpiderBase.__init__(self, home_page)
        self.url_save_file = "spider3_page_url_list.txt"


    def getPageUrlList(self, section_url):
        """get all page's url list in the specified section
        """
        print("get all page's url list in section: %s" % section_url)
        next_url_address = section_url
        has_next_page = True
        section_url_list = []
        section_url_list.append(section_url)
        print(section_url)
        
        # get all the page url in this section
        while has_next_page:
            time.sleep(1)
            reconnect_count = 0
            response_success = False
            response = None
            while reconnect_count < 5:
                try:
                    # parse next page
                    response = requests.get(next_url_address)
                    response.encoding= 'gb2312'  
                    # If the decoding has a format error, use next line to check encode type
                    #print(requests.utils.get_encodings_from_content(response.text))   
                    response_success = True
                except:
                    print("connect error! reconnecting...")
                    time.sleep(10)
                    reconnect_count += 1
                if response_success:
                    break
            #print("****", response)
            # get next page button's corresponding link
            soup = bs4.BeautifulSoup(response.text,"html5lib")        
            temp = soup.find(attrs={"class":"pages"})
            next_page_btn = temp.findAll('a')[-2] 
            next_url_address = next_page_btn.get('href')
            next_url_address = section_url + next_url_address
            
            # check if the last page is reached
            if next_page_btn.text.encode("utf-8")  == '下一页':
                section_url_list.append(next_url_address)
                print(next_url_address)
            else:
                has_next_page = False
        return section_url_list

    
    def getPostUrlList(self, page_url):
        """ get all post's url list in the specified page
        
        获取指定页面中所有色情帖子的url列表
        """
        reconnect_count = 0
        response_success = False
        response = None
        while reconnect_count < 5:
            try:
                response = requests.get(page_url)
                response.encoding= 'gb2312'  
                # If the decoding has a format error, use next line to check encode type
                #print(requests.utils.get_encodings_from_content(response.text))   
                response_success = True
            except:
                print("connect error! reconnecting...")
                time.sleep(10)
                reconnect_count += 1
            if response_success:
                break
        # get all post info list
        soup = bs4.BeautifulSoup(response.text, "html5lib") 
        tmp = soup.find(attrs={"class":"pic-list mb20 list"})
        post_info_list = tmp.findAll('li')
        # get all post url in this page
        post_url_list = []
        for post_info in post_info_list:
            post_info = post_info.find(attrs={"class":"name"})
            post_url_address = post_info.find('a').get('href')
            post_url_address = self.homepage_url + post_url_address
            post_url_list.append(post_url_address) 
        return post_url_list


    def getPicUrlList(self, post_url):
        """get all Picture's url list in the specified post
        
        获取指定帖子中的所有色情图片的url列表
        """        
        reconnect_count = 0
        response_success = False
        response = None
        while reconnect_count < 5:
            try:
                response = requests.get(post_url)
                response.encoding= 'gb2312'  
                # If the decoding has a format error, use next line to check encode type
                #print(requests.utils.get_encodings_from_content(response.text))   
                response_success = True
            except:
                print("connect error! reconnecting...")
                time.sleep(10)
                reconnect_count += 1
            if response_success:
                break      
        # get all pic info list
        soup = bs4.BeautifulSoup(response.text,"html5lib") 
        tmp = soup.find(attrs={"class":"page"})
        tmp = tmp.findAll('li')
        pic_num = len(tmp) - 3
        
        # get all pic url in this page
        pic_page_url_list = []
        pic_page_url_list.append(post_url)
        post_url_base = post_url[:-5]
        for idx in range(2, pic_num + 1):
            cur_url_address = post_url_base + '_' + str(idx) + '.html'
            pic_page_url_list.append(cur_url_address) 
        
        pic_url_list = []
        for url in pic_page_url_list:
            try:
                response = requests.get(url)
                response.encoding= 'gb2312'  
                # If the decoding has a format error, use next line to check encode type
                #print(requests.utils.get_encodings_from_content(response.text))   
                
                # get all pic info list
                soup = bs4.BeautifulSoup(response.text, "html5lib") 
                tmp = soup.find(attrs={"class":"picbox"})
                pic_url = tmp.find('img').get('src')
                pic_url_list.append(pic_url)
            except:
                print("*", end="")
        return pic_url_list



    def downloadPicBySection(self, section_url, save_path, check_point):
        """按章节下载图片
        
        例如下载'性感美女'章节的所有图片
        函数首先调用loadUrlList函数获取指定章节的所有页面url列表
        随后调用downloadPicByPage函数依次下载每个页面中的色情图片
        
        Args:
            page_url：需要下载图片的页面链接
            save_path: 图像保存路径
            check_point: 检查点,从第check_point页开始继续爬取图片,用于快速继续之前的工作
        """
        # check if has download and save page url list
        page_url_list = []
        if os.path.exists(self.url_save_file):
            page_url_list = self.loadUrlList(self.url_save_file)
        else:
            page_url_list = self.getPageUrlList(section_url)
            self.saveUrlList(self.url_save_file, page_url_list)
        
        # download porn images by page
        for page_idx, page in enumerate(page_url_list):
            if page_idx < check_point:
                continue
            
            # get all post in current page(retry until success when error occur)
            print("\nget all posts in %d th page..." % page_idx)
            get_post_success = False
            post_url_list = []
            while get_post_success == False:
                try:
                    post_url_list = self.getPostUrlList(page)
                    get_post_success = True
                except:
                    print("%d th page occur parse error!" % page_idx)
                    time.sleep(10)
            
            print("start download %d th page..." % page_idx)
            for post_idx, post_url in enumerate(post_url_list):
                prefix = str(page_idx) + '_' + str(post_idx) + '_'
                try:
                    # download pic in this post(ignore current post if error occur)
                    self.downloadPicByPost(post_url, prefix, save_path)
                except:
                    print("%s occru parse error!" % prefix)
                    time.sleep(10)
                    

    def downloadPicByPost(self, post_url, file_prefix, save_dir):
        """按帖子下载图片
        
        函数首先调用getPicUrlList函数获取指定帖子的所有色情图片url列表
        随后依次下载并保存每一张图片
        
        Args:
            post_url：需要下载图片的页面链接
            file_prefix: 文件名前缀（调用者需通过本参数提供可以保证文件命名唯一性的前缀）
            save_dir: 图像保存路径
        """
        reconnect_count = 0
        response_success = False
        while reconnect_count < 5:
            try:
                response = requests.get(post_url)
                response.encoding= 'gb2312'  
                # If the decoding has a format error, use next line to check encode type
                #print(requests.utils.get_encodings_from_content(response.text))   
                response_success = True
            except:
                print("connect error! reconnecting...")
                time.sleep(10)
                reconnect_count += 1
            if response_success:
                break
        
        if response_success == False:
            print("error when get title of post:", post_url)
            raise("error when get port title")
        
        # get title
        soup = bs4.BeautifulSoup(response.text,"html5lib") 
        tmp = soup.find(attrs={"class":"title"})
        title = tmp.find('h2').text
        
        # make directory
        save_dir = os.path.join(save_dir, title)
        if os.path.exists(save_dir) == False:
            os.mkdir(save_dir)

        # modify file prefix
        file_prefix = title + '_' + file_prefix
        
        pic_url_list = self.getPicUrlList(post_url)

        for pic_url in pic_url_list:
            pic_name = pic_url.split('/')[-1]            # 从url中得到图片命名
            pic_name = file_prefix + pic_name    # 为图片名添加前缀
            pic_path = os.path.join(save_dir, pic_name)
            try:
                img = requests.get(pic_url, timeout=10).content
                fw = open(pic_path, 'wb')
                fw.write(img)
            except:
                print("*", end="")



if __name__ == "__main__":
    
    # PornSpider3调用示例，对应网站："https://www.zbjuran.com/mei/"
    # 爬取整个性感美女模块图片示例
    home_page = "https://www.zbjuran.com"
    spider = PornSpider3(home_page)

    section_url_address = 'https://www.zbjuran.com/mei/xinggan/'
    save_path = './download/xinggan/'
    check_point = 0
    spider.downloadPicBySection(section_url_address, save_path, check_point)
    print("finish!")