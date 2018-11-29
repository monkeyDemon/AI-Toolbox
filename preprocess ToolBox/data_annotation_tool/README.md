# data annotation demo

here has a few data annotation demos in this directory

`gui_tkinter_exercise.py`

`data_annotation_tool.py`

`image_select_tool.py`

## A singal GUI demo

`gui_tkinter_exercise.py` is a singal GUI demo, shows the most basic method to use the python GUI package Tkinter.

If you are new to python or Tkinter, reading the code in this demo is a good choice. In this demo, you will know how to create a GUI window and some necessary knowledge to build a data annotation tool.

![fft](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/readme_image/gui_exercise_1.png)


## A data annotation tool

`data_annotation_tool.py` is a demo to demonstrate how to implement a data annotation program

how to use?

first, the save path has shown in the text box(temporarily does not support modification)

then, we click 'start' button to select a directory

after that, all the images in the specified directory will be load one by one

click the category button(for example 'Dog' and 'Cat') to label image samples

click the 'NumericUpDown Control' to jump to specified image

click the 'QUIT' button to quit

![fft](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/readme_image/data_annotation_1.png)

![fft](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/readme_image/data_annotation_2.png)

## A image select tool

`image_select_tool.py` is a demo to demonstrate how to implement a image select tool

potential scene:

we have a lot of group images, we want to filter out one or several images for each set of images. for this usage scene, you can use this tool to speed up your work.

The running logic of the program is:

first, the save path has shown in the text box(temporarily does not support modification)

then, we click 'start' button to select a root directory

after that, all the sub-directories in the specified root directory will be load one by one

all the images in one sub-directory will be shown in the canvas

use the mouse to click and filter out the desired image, it will be auto saved into save path

use mouse wheel to scroll and quickly view all images

press 'space' to pass to next group of images

press 'Q' or click button 'QUIT' to quit

![fft](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/readme_image/image_select_tool_1.png)

![fft](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/readme_image/image_select_tool_2.png)
