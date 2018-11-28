# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:05:50 2018

image select tool

this is a demo to demonstrate how to implement a image select tool

potential scene:
we have a lot of group images, we want to filter out one or several images for each set of images.
for this usage scene, you can use this tool to speed up your work.

The running logic of the program is:
first, the save path has shown in the text box(temporarily does not support modification)
then, we click 'start' button to select a root directory
after that, all the sub-directories in the specified root directory will be load one by one
all the images in one sub-directory will be shown in the canvas 
use the mouse to click and filter out the desired image, it will be auto saved into save path
use mouse wheel to scroll and quickly view all images
press 'space' to pass to next group of images
press 'Q' or click button 'QUIT' to quit

python version: Python 2.7

@author: zyb_as
"""

import Tkinter  
from Tkinter import Button, Canvas, Entry, Frame, Scrollbar
import tkFileDialog
import tkMessageBox
from Tix import Control 
from PIL import Image, ImageTk
import math
import os



class ImageSelectWindow(object):
    def __init__(self):
        self.image_Wid = 300
        self.image_Hei = 300
        self.image_interval = 10
        self.canvas_Wid = 930
        self.canvas_Hei = 600
        self.window_wid = self.canvas_Wid + 160
        self.window_Hei = self.canvas_Hei
        
        # init state control variable
        self.has_select_root_path = False  # sign has select root path(where the sub-directories save) or not
        self.root_path = ''
        self.group_list = []
        
        # create main window
        self.mainWin = Tkinter.Tk()  
        self.mainWin.geometry(str(self.window_wid) + 'x' + str(self.window_Hei))  # init the size of window
        self.mainWin.title("image select tool")
        self.mainWin.tk.eval('package require Tix') 
        # bind the key press event(space to pass, q to quit) 
        self.mainWin.bind('<KeyPress>', self._keyPressEvent)
        
        # create init image(a black background)
        self.img_path_list = []
        self.photo_img_list = []
            
        # create Canvas control
        self.max_shown_num = 90
        self.cv = Canvas(self.mainWin, bg = 'white', width = self.canvas_Wid, height = self.canvas_Hei
                         , scrollregion=(0, 0, self.canvas_Wid,(self.image_Hei + self.image_interval) * math.ceil(self.max_shown_num / 3)))
        self.vbar=Scrollbar(self.cv, orient=Tkinter.VERTICAL)
        self.vbar.pack(side=Tkinter.RIGHT,fill=Tkinter.Y)
        self.vbar.config(command = self.cv.yview)
        self.cv.config(yscrollcommand = self.vbar.set)
        self.cv.pack(side=Tkinter.LEFT, expand=True, fill=Tkinter.BOTH)
        # bind the mouse click event
        self.cv.bind('<Button-1>', self._mouseClickEvent)
        # bind the mouse wheel event
        self.cv.bind_all('<MouseWheel>', self._mouseWheelEvent)
        
        
        # create total Frame to lay out all components
        self.frame = Frame(self.mainWin, width = 250, height = self.window_Hei)
        self.frame.pack(side=Tkinter.LEFT)     

        # create text control
        self.save_path = './image_select_result'
        self.entry= Entry(self.frame, state = 'normal')
        self.entry.pack(side = Tkinter.TOP)
        self.entry.insert(0, self.save_path)
        
        # create 'START' button
        self.btn_start = Button(self.frame, text='START', command=self._selectPath, activeforeground='blue',
                        activebackground='white', bg='blue', fg='white')  
        self.btn_start.pack(side = Tkinter.TOP, pady=30)
                
        # create data annotation label button
        self.btn_pass = Button(self.frame, text='PASS', command=lambda:self._passCurrentGroup(), 
                               activeforeground='black', activebackground='blue', bg='white', fg='black')  
        self.btn_pass.pack(side = Tkinter.TOP, pady=30)
        
        #NumericUpDown控件
        self.num_count = Control(self.frame,integer=True, max=-1, min=-1, value=-1, step=1,
                                 label='current Image:', command=self._showImages)
        self.num_count.label.config(font='Helvetica -14 bold')
        self.num_count.pack(side = Tkinter.TOP, pady=30)
          
        # create 'QUIT' button
        self.btn_quit = Button(self.frame, text='QUIT',command=self.mainWin.quit,activeforeground='blue',
                        activebackground='white', bg='red', fg='white')  
        self.btn_quit.pack(side = Tkinter.BOTTOM, pady=100)



    def _keyPressEvent(self, event):
        if event.keycode == 32:
            self._passCurrentGroup()
        elif event.keycode == 81:
            self.mainWin.destroy()
    
    
    
    def _mouseClickEvent(self, event):
        # get coordinate of x
        x_coordinate = event.x
        
        # compute coordinate of y
        # get the location of the scrollBar in the scroll range 
        scale = self.vbar.get()[0]
        start_y = scale * ((self.image_Hei + self.image_interval) * math.ceil(self.max_shown_num / 3))
        y_coordinate = start_y + event.y
        self._selectOneImage(x_coordinate, y_coordinate)
    

    def _mouseWheelEvent(self, event):
        self.cv.yview_scroll(-1*(event.delta/20), "units")



    def _showImages(self, ev=None):
        if self.has_select_root_path:
            if int(self.num_count['value']) == -1:                
                # clear the images draw before first
                for idx in range(len(self.img_path_list)):
                    self.cv.delete(self.img_path_list[idx])
                
                self.img_path_list = []
                self.photo_img_list = []
                return
            else:           
                # clear the images draw before first
                for idx in range(len(self.img_path_list)):
                    self.cv.delete(self.img_path_list[idx])
                
                # get the current group path and images list
                img_group_path = self.group_list[int(self.num_count['value'])]
                self.img_path_list = self._getImagePathList(img_group_path)
                self.img_path_list = self.img_path_list[:min(len(self.img_path_list), self.max_shown_num)]
                self.photo_img_list = []
                for idx in range(len(self.img_path_list)):
                    # load the current image
                    cur_img_path = self.img_path_list[idx]
                    cur_img = Image.open(cur_img_path).resize((self.image_Wid, self.image_Hei))
                    self.photo_img_list.append(ImageTk.PhotoImage(cur_img))
                    # draw cur image to the appropriate location by index
                    x_coordinate = (idx % 3) * (self.image_Wid + self.image_interval)
                    y_coordinate = math.floor(idx/3) * (self.image_Hei + self.image_interval)
                    self.cv.create_image((x_coordinate, y_coordinate), anchor=Tkinter.NW, 
                            image = self.photo_img_list[idx], tags = self.img_path_list[idx])
                    # give a tag is convenient for delete later
                
    
    
    
    def _passCurrentGroup(self):
        cur_idx = int(self.num_count['value'])
        if cur_idx != -1:
            # check has next group or not
            if cur_idx + 1 < len(self.group_list):
                self.num_count['value'] = str(cur_idx + 1)
            else:
                tkMessageBox.showinfo(title='thanks', message='all the image select mission has finished~')
    
    
        
        
    def _selectOneImage(self, x_coordinate, y_coordinate):
        cur_idx = int(self.num_count['value'])
        if cur_idx == -1:
            return
        col = math.floor(x_coordinate / (self.image_Wid + self.image_interval))
        row = math.floor(y_coordinate / (self.image_Hei + self.image_interval))
        idx = int(row * 3 + col)
        if idx < len(self.img_path_list):
            img_click_path = self.img_path_list[idx]
            cur_img = Image.open(img_click_path)
            img_name = img_click_path.split('\\')[-1]
            cur_save_path = os.path.join(self.save_path, img_name)
            cur_img.save(cur_save_path)
    

    
    def _selectPath(self):
        self.root_path = tkFileDialog.askdirectory()
        self.group_list = self._getGroupPathList()
        if len(self.group_list) > 0:
            self.has_select_root_path = True
            self.num_count.config(max = len(self.group_list) - 1)
            self.num_count['value'] = str(0)
        else:
            self.has_select_root_path = False
            self.num_count['value'] = str(-1)
            self.num_count.config(max = -1)
            tkMessageBox.showwarning('warning','No available sub-directories detected! please re-select root path and start')



    def _getGroupPathList(self):
        group_list = []
        for root, dirs, files in os.walk(self.root_path):
            for dir_name in dirs:
                cur_group_path = os.path.join(root, dir_name)
                group_list.append(cur_group_path)
        return group_list



    def _getImagePathList(self, img_group_path):
        img_path_list = []
        for root, dirs, files in os.walk(img_group_path):
            for file_name in files:
                cur_img_path = os.path.join(img_group_path, file_name)
                img_path_list.append(cur_img_path)
        return img_path_list



if __name__ == '__main__':
    image_select_tool = ImageSelectWindow()
    Tkinter.mainloop()  # run GUI