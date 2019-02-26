# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 17:20:22 2019

convert txt to excel

@author: zyb_as
"""
import os
import argparse, textwrap
import xlwt

# set options
parser = argparse.ArgumentParser(description = 'convert txt to excel', 
        usage = textwrap.dedent('''\
        command example:
        python %(prog)s --file_name='test.txt' --splitter='\\t' '''),
        formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument('--file_name', type = str, default = 'test.txt',
        help = 'the path of the txt file')
parser.add_argument('--splitter', type = str, default = '\t',
        help = 'the splitter for each line in the txt file.')
#parser.add_argument('--fields_num', type = int, default = 1,
#        help = 'the fields number each line.')
parser.add_argument('--max_lines', type = int, default = 50000,
        help = 'max lines number in one excel')


def download_from_txt():    
    # get options
    args = parser.parse_args()
    file_name = args.file_name
    splitter = args.splitter
    #fields_num = args.fields_num
    max_lines = args.max_lines
    
    if not os.path.exists(file_name):
        print("ERROR! the file need to be convert does't exists")

    excel_file = ''
    if file_name[-4:] == '.txt':
        excel_file = file_name[:-4] + '.xls'
    else:
        excel_file = file_name + '.xls'
    
    if splitter == '\\t':
        splitter = '\t'
    
    cnt = 0
    xls_index = 0
    cur_excel_file = excel_file[:-4] + '_' + str(xls_index) + '.xls'
    # 创建表
    workbook = xlwt.Workbook(encoding = 'utf-8')
    worksheet = workbook.add_sheet('temp', cell_overwrite_ok = True)
    worksheet.write(0, 0, label = 'Row 0, Column 0 Value')
    for line in open(file_name, 'r').readlines():
        if cnt == max_lines:
            workbook.save(cur_excel_file)
            xls_index += 1
            cur_excel_file = excel_file[:-4] + '_' + str(xls_index) + '.xls'
            workbook = xlwt.Workbook(encoding = 'utf-8')
            worksheet = workbook.add_sheet('temp')
            cnt = 0
        
        item = line.split(splitter)
        print(cnt)
        for idx, it in enumerate(item):
            worksheet.write(cnt, idx, it.decode('utf-8', 'ignore'))
        cnt += 1
    if cnt <= max_lines:
       workbook.save(cur_excel_file) 
       
if __name__ == "__main__":
    download_from_txt()