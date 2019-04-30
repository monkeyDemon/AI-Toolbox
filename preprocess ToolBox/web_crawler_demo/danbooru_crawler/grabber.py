#coding:utf-8
from __future__ import division
import os 
import cv2
import pickle
import requests
import numpy as np


# TODO: set gloabl parameters
download_dir = './danbooru'
dataset_info_path = 'dataset_info.txt'
id_set_path = 'img_id_set.pkl'
id_set = set()
tags_download_list_path = 'tmp/tags_list.txt'
status = 'not done yet'
max_page_num = 50000
checkpoint = 0
checkpoint_end = 5


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_tags_download_list(tags_list_path):
    tags_download_list = []
    with open(tags_list_path, 'r') as reader:
        for line in reader:
            tags_download_list.append(line.split('\t')[1])
    return tags_download_list


def _get_img_by_url(url, save_dir):
    """
    download and save a image by url
    """
    if save_dir[-1] != '/': # make sure save_dir end with '/'
        save_dir += '/'
    
    # request url
    #response = requests.get(url,timeout=0.1,proxies=proxies)
    pic_name = url.split('/')[-1]
    response = requests.get(url,timeout=5)
    
    # save image
    with open(save_dir + pic_name, 'wb') as fd:
        for chunk in response.iter_content(128):  # iter_content saving while downloading
            fd.write(chunk)    



def _get_img_by_url2(url, save_dir):
    #response = requests.get(url,timeout=8,proxies=proxies)
    response = requests.get(url, timeout=5)
    imgDataNp = np.frombuffer(response.content, dtype='uint8')
    img = cv2.imdecode(imgDataNp, cv2.IMREAD_UNCHANGED)   # here the img is RGB three dimensional data range from 0-255
    '''
    dim = len(img.shape)
    if dim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif dim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        print("The situation did not take into account!")
        print(dim)
        print(img.shape)
        raise("runtimeError")
    '''
    img = cv2.resize(img, (192,192))
    
    # save image
    if save_dir[-1] != '/': # make sure save_dir end with '/'
        save_dir += '/'
    pic_name = url.split('/')[-1]
    pic_save_path = save_dir + pic_name
    cv2.imwrite(pic_save_path, img)



def check_post(post):
    '''
    检查当前插图基本信息是否满足要求
    可以在当前函数中添加需要的规则，如必须包含url, 长宽比不能超过2：1
    '''
    flag = True
    # 确认包含url
    if 'file_url' not in post:
        flag = False
        print('can not find field \'file_url\'')
        return flag
    # 长宽比不能超过2：1
    wid = float(post['image_width'])
    hei = float(post['image_height'])
    ratio = wid / hei
    if ratio > 2 or ratio < 0.5:
        flag = False
        print('ratio {} out of range'.format(ratio))
        return flag
    # 确认是正常数据(后缀名正常)
    suffix = post['file_ext']
    if suffix not in ['jpg','png','bmp']:
        flag = False
        print('get wrong suffix {}'.format(suffix))
        return flag
    # 判断是否已经存在（使用记录的id字典）
    img_id = post['id']
    if img_id in id_set:
        flag = False
        print('image already exists, id: {}'.format(img_id))
        return flag
    return flag
    

def parse_post(post):
    '''
    解析post
    获取想要的信息，如url, tags
    '''
    img_id = post['id']
    #url = post['file_url']
    url = post['preview_file_url']
    tag_string = post['tag_string']
    rate = post['rating']
    tag_string = rate + ' ' + tag_string
    return img_id, url, tag_string
    
    

# request json, get urls of pictures and download them
def grabber(tag_argv, page_num):
    r = requests.get('https://danbooru.donmai.us/posts.json?tags='+tag_argv+'&page='+str(page_num))
    streams = r.json()
    # check if all pages have been visited
    if len(streams) == 0:
        print("All pictures have been downloaded!")
        global status
        status = 'done'
    else:
        info_line = []
        for post in streams:
            flag = check_post(post)
            if flag == False:
                continue
            img_id, url, tag_string = parse_post(post)
            try:
                _get_img_by_url2(url, download_dir)
            except:
                print("http error: time out.")
                continue
            # record new image id
            id_set.add(img_id)
            info = str(img_id) + '\t' + url + '\t' + tag_string
            info_line.append(info)
        # write dataset info
        with open(dataset_info_path, 'a+') as f:
            for info in info_line:
                info += '\n'
                f.write(info)


def main():
    tag_list = load_tags_download_list(tags_download_list_path)
    
    for i, tag in enumerate(tag_list):
        print("------------------------------------------")
        print("current tag: {}, checkpoint: {}".format(tag, i))
        if checkpoint > i:
            print("skip")
            continue
        if i >= checkpoint_end:
            print("come to checkpoint end, return")
            break
        # start downloading current tag
        status = 'not done yet'
    	n = 1
        while n <= max_page_num and status == 'not done yet':
            print("\nstart downloading page {}...".format(n))
            grabber(tag, n)
            n = n + 1
            save_obj(id_set, id_set_path)
	print('Download successful!')


if __name__ == '__main__':
    # check if directory already exists
    if (os.path.exists(download_dir) == False):
        os.mkdir(download_dir)
        
    # load image id dictionary
    if os.path.exists(id_set_path):
        id_set = load_obj(id_set_path)

    main()