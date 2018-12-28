# data annotation demo

here has a few data annotation demos in this directory

`gui_tkinter_exercise.py`

`data_annotation_tool.py`

`data_annotation_too2.py`

`data_annotation_too3.py`

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

## data annotation tool2

`data_annotation_tool2.py` is also a data annotation tool, some improvements have been made on the basis tool data_annotation_tool.py, and make it more efficient.

And data annotation tool2 is more suitable for such a scene:

We have a lot of images to do data annotation, but the distribution of data
is extremely uneven, most of the samples are belong to one category, and a few
of samples are belong to another. We hope to separate the two categories more efficiently.

how to use?

first, the save path has shown in the text box(temporarily does not support modification)

then, we click 'start' button to select a directory

after that, all the images in the specified directory will be load one by one

press 'space' or click the 'Pass' button to mark it as category A

press 'd' or click the 'Remove' button to mark it as category B

click the 'QUIT' button to quit

Compared with the tool `data_annotation_tool2.py`, you can now completely remove the mouse operation, and most of the time, you simply press the space bar repeatedly, the operation is much more efficient.

## data annotation tool3

`data_annotation_tool3.py` is also a data annotation tool, can handle situation for multiple categories.

Base on data_annotation_tool.py and data_annotation_tool2.py, we optimized the
visualization style so that images with large aspect ratios can also be
displayed properly.

how to use?

first, the save path has shown in the text box(temporarily does not support modification)

then, we click 'start' button to select a directory

after that, all the images in the specified directory will be load one by one

press 'q' or click the 'Category 1' button to mark it as category 1

press 'w' or click the 'Category 2' button to mark it as category 2

press 'e' or click the 'Category 3' button to mark it as category 3

click the 'QUIT' button to quit

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
