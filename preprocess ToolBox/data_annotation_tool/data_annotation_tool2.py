# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:24:48 2018

data annotation tool2

This is also a data annotation tool, some improvements have been made on the 
basis tool data_annotation_tool.py, and make it more efficient.

And data annotation tool2 is more suitable for such a scene:
We have a lot of images to do data annotation, but the distribution of data 
is extremely uneven, most of the samples are belong to one category, and a few
of samples are belong to another. We hope to separate the two categories more 
efficiently.

how to use?
first, the save path has shown in the text box(temporarily does not support modification)

then, we click 'start' button to select a directory

after that, all the images in the specified directory will be load one by one

press 'space' or click the 'Pass' button to mark it as category A

press 'd' or click the 'Remove' button to mark it as category B

click the 'QUIT' button to quit


python version: Python 2.7

@author: zyb_as
"""

import Tkinter  
from Tkinter import Button, Canvas, Entry, Frame
import tkFileDialog
import tkMessageBox
from Tix import Control 
from PIL import Image, ImageTk
import os







class DataAnnotationWindow(object):
    def __init__(self):
        self.img_Wid = 500
        self.img_Hei = 500
        self.win_wid = self.img_Wid + 200
        self.win_Hei = self.img_Hei
        
        # init state control variable
        self.has_select_path = False  # sign has select image path(to do data annotation) or not
        self.img_path = ''
        self.img_list = []
        
        # create main window
        self.mainWin = Tkinter.Tk()  
        self.mainWin.geometry(str(self.win_wid) + 'x' + str(self.win_Hei))  # init the size of window
        self.mainWin.title("data annotation tool")
        self.mainWin.tk.eval('package require Tix') 
        # bind the key press event(space to pass, d to delete, q to quit) 
        self.mainWin.bind('<KeyPress>', self._keyPressEvent)
        
        # create init image(a black background)
        self.img = Image.new("RGB", (self.img_Wid, self.img_Hei), (0,0,0))
        self.photo_img = ImageTk.PhotoImage(self.img)
            
        # create Canvas control
        self.cv = Canvas(self.mainWin, bg = 'white', width = self.img_Wid, height = self.img_Hei) 
        self.cv.create_image((0,0), anchor=Tkinter.NW, image=self.photo_img) 
        self.cv.pack(side=Tkinter.LEFT, expand=True)

        
        # create total Frame to lay out all components
        self.frame = Frame(self.mainWin)
        self.frame.pack(fill=Tkinter.X, expand=Tkinter.YES, side=Tkinter.LEFT)
        

        # create text control
        root_save_path = './data_annotation_result'
        self.entry= Entry(self.frame, state = 'normal')
        self.entry.pack(side = Tkinter.TOP, fill = Tkinter.X)
        self.entry.insert(0, root_save_path)
        # mkdir of annotation result
        self.categoryList = ['Pass', 'Remove']
        self.category_savepath_list = []
        for category in self.categoryList:
            cur_category_save_path = os.path.join(root_save_path, category)
            self.category_savepath_list.append(cur_category_save_path)
            if os.path.exists(cur_category_save_path) == False:
                os.mkdir(cur_category_save_path)
        
        # create 'START' button
        self.btn_start = Button(self.frame, text='START', command=self._selectPath, activeforeground='blue',
                        activebackground='white', bg='blue', fg='white')  
        self.btn_start.pack(side = Tkinter.TOP, pady=30)
        
        
        # create data annotation label button
        self.btn_dog = Button(self.frame, text='Pass',command=lambda:self._labelButtonClick('Pass'), activeforeground='black',
                        activebackground='blue', bg='white', fg='black')  
        self.btn_dog.pack(side = Tkinter.TOP, pady=10)
        
        # create data annotation label button
        self.btn_cat = Button(self.frame, text='Remove',command=lambda:self._labelButtonClick('Remove'), activeforeground='black',
                        activebackground='blue', bg='white', fg='black')  
        self.btn_cat.pack(side = Tkinter.TOP, pady=10)
        
        #NumericUpDown控件
        self.num_count = Control(self.frame,integer=True, max=-1, min=-1, value=-1, step=1,
                                 label='current Image:', command=self._showImage)
        self.num_count.label.config(font='Helvetica -14 bold')
        self.num_count.pack(side = Tkinter.TOP, pady=50)
        
        
        # create 'QUIT' button
        self.btn_quit = Button(self.frame, text='QUIT',command=self.mainWin.quit,activeforeground='blue',
                        activebackground='white', bg='red', fg='white')  
        self.btn_quit.pack(side = Tkinter.BOTTOM, pady=10)
    
    
    def _keyPressEvent(self, event):
        if event.keycode == 32:
            self._labelButtonClick('Pass')
        elif event.keycode == 68:
            self._labelButtonClick('Remove')
        elif event.keycode == 81:
            self.mainWin.destroy()


    def _showImage(self, ev=None):
        if self.has_select_path:
            if int(self.num_count['value']) == -1:
                # create init image(a black background)
                self.img = Image.new("RGB", (self.img_Wid, self.img_Hei), (0,0,0))
                self.photo_img = ImageTk.PhotoImage(self.img)
                self.cv.create_image((0,0), anchor=Tkinter.NW, image=self.photo_img) 
                return
            else:
                img_cur_path = self.img_list[int(self.num_count['value'])]
                self.img = Image.open(img_cur_path).resize((self.img_Wid, self.img_Hei))
                self.photo_img = ImageTk.PhotoImage(self.img)
                self.cv.create_image((0,0), anchor=Tkinter.NW, image=self.photo_img) 
    
    
    def _labelButtonClick(self, label):
        cur_idx = int(self.num_count['value'])
        if cur_idx != -1:
            # save cur image's annotation result
            catgory_idx = self.categoryList.index(label)
            save_path = self.category_savepath_list[catgory_idx]
            cur_img_name = self.img_list[cur_idx].split('\\')[-1]
            save_path = os.path.join(save_path, cur_img_name)
            self.img.save(save_path)
            
            # check has next image or not
            if cur_idx + 1 < len(self.img_list):
                self.num_count['value'] = str(cur_idx + 1)
            else:
                tkMessageBox.showinfo(title='thanks', message='all the data annotation mission has finished~')
    
    
    def _selectPath(self):
        self.img_path = tkFileDialog.askdirectory()
        self.img_list = self._getImagList()
        if len(self.img_list) > 0:
            self.has_select_path = True
            self.num_count.config(max = len(self.img_list))
            self.num_count['value'] = str(0)
        else:
            self.has_select_path = False
            self.num_count['value'] = str(-1)
            self.num_count.config(max = -1)
            tkMessageBox.showwarning('warning','No available images detected! please re-select image path and start')


    def _getImagList(self):
        img_list = []
        img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
        for root, dirs, files in os.walk(self.img_path):
            for file_name in files:
                suffix = file_name.split('.')[-1]
                if suffix in img_suffix:
                    cur_img_path = os.path.join(root, file_name)
                    img_list.append(cur_img_path)
        return img_list


if __name__ == '__main__':
    data_annotation_tool = DataAnnotationWindow()
    Tkinter.mainloop()  # run GUI