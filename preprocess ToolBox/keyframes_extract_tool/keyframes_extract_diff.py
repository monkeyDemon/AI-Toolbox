# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:48:57 2018

keyframes extract tool

this key frame extract algorithm is based on interframe difference.

The principle is very simple
First, we load the video and compute the interframe difference between each frames

Then, we can choose one of these three methods to extract keyframes, which are 
all based on the difference method:
    
1. use the difference order
    The first few frames with the largest average interframe difference 
    are considered to be key frames.
2. use the difference threshold
    The frames which the average interframe difference are large than the 
    threshold are considered to be key frames.
3. use local maximum
    The frames which the average interframe difference are local maximum are 
    considered to be key frames.
    It should be noted that smoothing the average difference value before 
    calculating the local maximum can effectively remove noise to avoid 
    repeated extraction of frames of similar scenes.

After a few experiment, the third method has a better key frame extraction effect.

The original code comes from the link below, I optimized the code to reduce 
unnecessary memory consumption.
https://blog.csdn.net/qq_21997625/article/details/81285096

@author: zyb_as
""" 
import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import gc
import psutil, os, sys
from time import time
import logging
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))

def smooth(x, window_len=13, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
        
    example:
    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """
    logger.info("length of frames: %d" % len(x))
    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    #
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    #
    # if window_len < 3:
    #     return x
    #
    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
 
    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    #print(len(s))
 
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]

class Frame():
    def __init__(self, id, diff):
        self.id, self.diff = id, diff

def rel_change(a, b):
   x = (b - a) / max(a, b)
   logger.debug(x)
   return x

class KeyFrames():
    USE_TOP_ORDER = 1  # Setting fixed threshold criteria
    USE_THRESH = 2
    USE_LOCAL_MAXIMA = 4
    def __init__(self, mode = 4, debug = False, **kwargs):
        #Constant

        self.set(mode, debug, **kwargs)

    def set(self, mode, debug = -1, **kwargs):
        if mode:
            self.use_top_order = 1 & mode
            self.use_thresh = 2 & mode
            self.use_local_maxima = 4 & mode
            if self.use_thresh:
                logger.info("Use Threshold")
                self.thresh = kwargs.get("thresh", 0.6)
            if self.use_top_order:
                logger.info("Use top order")
                self.num_top_frames = kwargs("num_top_frames", 50)
            if self.use_local_maxima:
                self.len_window = kwargs.get("len_window", 50)
                logger.info("Use Local Maxima")
        if debug == -1:
            pass
        elif debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    #@video_path : Video path of the source file
    #@dir : Directory to store the processed frames
    def extract(self, video_path, dir=None, mode=0, debug = -1, **kwargs):
        self.set(mode, debug, **kwargs)

        start_time = time()
        logger.debug(sys.executable)
        if not dir:
            dir = os.path.splitext(video_path)[0]
            if not os.path.exists(dir):
                os.mkdir(dir)

        logger.debug("initial allocated memory: %d MB" % (psutil.Process(os.getpid()).memory_info().rss >> 20))
        logger.info("target video: " + os.path.abspath(video_path))
        logger.info("frame save directory: " + os.path.abspath(dir))

        # load video and compute diff between frames
        cap = cv2.VideoCapture(str(video_path))
        curr_frame = None
        prev_frame = None
        frame_diffs = []
        frames = []
        success, frame = cap.read()
        i = 0
        while(success):
            luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
            curr_frame = luv
            if curr_frame is not None and prev_frame is not None:
                #logic here
                diff = cv2.absdiff(curr_frame, prev_frame)
                diff_sum = np.sum(diff)
                diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
                frame_diffs.append(diff_sum_mean)
                frame = Frame(i, diff_sum_mean)
                frames.append(frame)
            prev_frame = curr_frame
            i = i + 1
            success, frame = cap.read()
        cap.release()

        # compute keyframe
        keyframe_id_set = set()
        if self.use_top_order:
            # sort the list in descending order
            frames.sort(key=operator.attrgetter("diff"), reverse=True)
            for keyframe in frames[:self.num_top_frames]:
                keyframe_id_set.add(keyframe.id)
        if self.use_thresh:
            for i in range(1, len(frames)):
                if (rel_change(np.float(frames[i - 1].diff), np.float(frames[i].diff)) >= self.thresh):
                    keyframe_id_set.add(frames[i].id)
        if self.use_local_maxima:
            diff_array = np.array(frame_diffs)
            sm_diff_array = smooth(diff_array, self.len_window)
            frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
            for i in frame_indexes:
                keyframe_id_set.add(frames[i - 1].id)

            plt.figure(figsize=(40, 20))
            plt.locator_params(numticks=100)
            plt.stem(sm_diff_array)
            plt.savefig(os.path.join(dir, 'plot.png'))
            plt.close('all'); gc.collect(); # these 2 statements will release the memory of plt and save 60MB

        # save all keyframes as image
        cap = cv2.VideoCapture(str(video_path))
        curr_frame = None
        keyframes = []
        success, frame = cap.read()
        idx = 0

        while(success):
            if idx in keyframe_id_set:
                name = "keyframe_%04d.jpg" %idx
                cv2.imwrite(os.path.join(dir, name), frame)
                keyframe_id_set.remove(idx)
            idx = idx + 1
            success, frame = cap.read()

        cap.release()

        end_time = time()
        logger.info("elapsed time: %d" % (end_time - start_time))
        logger.debug("end allocated memory: %dMB" % (psutil.Process(os.getpid()).memory_info().rss >> 20))


if __name__ == "__main__":
    # Usage example:
    # from keyframes_extract_diff import KeyFrames
    keyframes = KeyFrames() # this is the easiest way to use
    # keyframes = KeyFrames(mode=6)  # this works
    # keyframes = KeyFrames(mode=KeyFrames.USE_LOCAL_MAXIMA | KeyFrames.USE_THRESH, debug = True, thresh=0.7, len_window=60 ) # this also works
    keyframes.extract("pikachu.mp4")