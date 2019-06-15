# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:03:52 2018

A image download file

Read the text file which records url list of all pictures and download

In the work, in order to efficiently transfer a large amount of image data,
we usually only transmite a text file that records image information according
to a certain structure.

This tool shows how to download images by reading such a text file.
The structure of the file needs to be similar to the 'url_record'

python version: python 3.5

@author: zyb_as
"""

import os
import sys
import argparse, textwrap
import requests
import numpy as np

# set options
parser = argparse.ArgumentParser(description = 'manual to this script',
        usage = textwrap.dedent('''\
        command example:
        python %(prog)s --file_type='npy' --file_path='test.npy' --save_dir='./download'
        python %(prog)s --file_type='txt' --file_path='test.txt' --save_dir='./download' --splitter='\\t' --fields_num=1 --url_field_idx=0
        use "python %(prog)s --help" for more information '''),
        formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument('--file_type', type = str, default = None,
        help = textwrap.dedent('''\
        value: \'npy\' or \'txt\'
        type of the file storing the url list.'''))
parser.add_argument('--file_path', type = str, default = None,
        help = 'the path of the file storing the url list')
parser.add_argument('--save_dir', type = str, default = None,
        help = 'the directory to save the download images')
parser.add_argument('--splitter', type = str, default = '\t',
        help = textwrap.dedent('''\
        use when file_type=\'txt\'
        the splitter for each line in the txt file.'''))
parser.add_argument('--fields_num', type = int, default = 1,
        help = textwrap.dedent('''\
        use when file_type=\'txt\'
        the fields number each line.'''))
parser.add_argument('--url_field_idx', type = int, default = 0,
        help = textwrap.dedent('''\
        use when file_type=\'txt\'
        the idx of the url field in each line.'''))



def _get_img_by_url(url, save_dir):
    """
    download and save a image by url
    """
    if save_dir[-1] != '/': # make sure save_dir end with '/'
        save_dir += '/'

    # request url
    #response = requests.get(url,timeout=0.1,proxies=proxies)
    pic_name = url.split('/')[-1]
    response = requests.get(url,timeout=1)

    # save image
    with open(save_dir + pic_name, 'wb') as fd:
        for chunk in response.iter_content(128):  # iter_content saving while downloading
            fd.write(chunk)
    print(pic_name,'  OK')


def download_from_txt(url_file, save_dir, splitter = '\t', fields_num = 1, url_field_idx = 0):
    if os.path.exists(save_dir):
        print("warning! save_dir has exists, please check!")
    else:
        os.mkdir(save_dir)

    if splitter == '\\t':
        splitter = '\t'

    failure_count = 0
    failed_urls = []
    cnt = 0                 #spammer_urls
    with open(url_file, 'r') as reader:
        for line in reader.readlines():
            cnt += 1
            item = line.rstrip('\n').split(splitter)
            if len(item) == fields_num:
                try:
                    url = item[url_field_idx]
                    print(url)
                    _get_img_by_url(url, save_dir)
                except:
                    failure_count += 1
                    failed_urls.append(url)
    print("\nfailed record:")
    print(failure_count)
    print(failed_urls)

    # retry the download failed urls
    print("\nretry the download failed urls...")
    for url in failed_urls:
        try:
            print(url)
            _get_img_by_url(url, save_dir)
        except:
            failure_count += 1



def download_from_npy(npy_file, save_dir):
    if len(save_dir) == 0:
        raise("error! save_dir length is 0")
    if os.path.exists(save_dir):
        print("warning! save_dir has exists, please check!")
    else:
        os.mkdir(save_dir)

    url_list = np.load(npy_file)
    for url in url_list:
        #url = url.decode('utf8')
        _get_img_by_url(url, save_dir)



if __name__ == "__main__":
    # get options
    args = parser.parse_args()
    file_type = args.file_type
    file_path = args.file_path
    save_dir = args.save_dir

    if file_type == 'npy':
        download_from_npy(file_path, save_dir)
    elif file_type == 'txt':
        splitter = args.splitter
        fields_num = args.fields_num
        url_field_idx = args.url_field_idx
        download_from_txt(file_path, save_dir, splitter, fields_num, url_field_idx)
    else:
        print('Error use of --file_type! Please use -h to check')
        sys.exit()
