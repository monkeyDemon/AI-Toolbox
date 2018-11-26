# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:53:10 2018

Tkinter exercise

a singal demo shows how to the python GUI package Tkinter

python version: Python 3.5

@author: zyb_as
"""

import Tkinter  
from Tkinter import Label, Button, Canvas, Scrollbar
from Tix import Control, ComboBox  #升级的组合控件包
from PIL import Image, ImageTk

#顶层窗口
top = Tkinter.Tk()  #创建顶层窗口
top.geometry('400x400')  #初始化窗口大小
top.title("Tkinter exercise")
top.tk.eval('package require Tix')  #引入升级包，这样才能使用升级的组合控件


# 创建一个Canvas
# bg = 'white' 设置其背景色为白色 
# scrollregion 设置滚动条控制范围（即使画布更大，你也无法移动到scrollregion设置区域以外的区域）
cv = Canvas(top, bg = 'white', scrollregion=(0,0,600,1000)) 

vbar=Scrollbar(cv,orient=Tkinter.VERTICAL)
vbar.pack(side=Tkinter.RIGHT,fill=Tkinter.Y)
vbar.config(command=cv.yview)
#cv.config(width=300,height=650)  # 由于布局的方式，这里设置的宽和高就不起作用了
cv.config(yscrollcommand=vbar.set)
cv.pack(side=Tkinter.LEFT, expand=True, fill=Tkinter.BOTH)

# 在画布中绘制指定的图片
img_path_list=['./food/[长眠] (462).jpg', './food/411.jpg', './food/[长眠] (461).jpg']
img_path_list = [path.decode('utf-8') for path in img_path_list]
imgs_list = [Image.open(path).resize((200,200)) for path in img_path_list]
imgs_list= [ImageTk.PhotoImage(img) for img in imgs_list]
for i in range(len(imgs_list)):
   cv.create_image((0,200*i), anchor=Tkinter.NW, image=imgs_list[i]) 
   # 这里要注意，不要使用临时变量对create_image函数的imgae参数赋值
   # 因为Canvas仅仅是维护一个引用，当创建的PhotoImage实例被销毁时，显示会出问题



#标签控件
label = Label(top, text='Hello World!',font='Helvetica -12 bold')   #创建标签
label.pack(fill=Tkinter.Y, expand=1)  #填充到界面

#按钮控件
button = Button(top, text='QUIT',command=top.quit,activeforeground='blue',
                activebackground='white', bg='red', fg='white')  #创建按钮，command为回调函数
button.pack()
#button.pack(fill=Tkinter.X, expand=1) #fill=tkinter.X表示横向拉伸完全


#自定义函数，控制控件的缩放
def resize(ev=None):
    label.config(font='Helvetica -%d bold' % scale.get())

#比例尺控件
scale = Tkinter.Scale(top, label='scale', from_=10, to=40,orient=Tkinter.HORIZONTAL, command=resize)  #缩放比例尺
scale.set(12)  #初始值
scale.pack(fill=Tkinter.X, expand=1)  #填充到界面

#NumericUpDown控件
ct = Control(top, label='NumberUpDown:',integer=True, max=12, min=2, value=2, step=2)
ct.label.config(font='Helvetica -14 bold')
ct.pack()

#ComboBox控件
cb = ComboBox(top, label='Type:', editable=True)
for animal in ('dog', 'cat', 'hamster', 'python'):
    cb.insert(Tkinter.END, animal)
cb.pack()


Tkinter.mainloop()  #运行这个GUI应用
