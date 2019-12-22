# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:27:37 2019

# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py

@author: zyb_as
"""
import os
import cv2 # TODO
import random
import numpy as np
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import pdb

# TODO: can be modify
random_mirror = True
for_autoaug = True


def polices2str(polices):
    polices_str = "" 
    for i in range(0, len(polices), 3):
        police_name = polices[i]
        pr = polices[i+1]
        magnitude = polices[i+2]
        polices_str += police_name + ',' + str(pr) + ',' + str(magnitude) + ',' 
    polices_str = polices_str[:-1]
    return polices_str



def str2polices(police_str):
    polices = [] 
    items = police_str.split(',')
    for i in range(0, len(items), 3):
        police_name = items[i].strip()
        pr = items[i+1].strip()
        magnitude = items[i+2].strip()
        polices.append(police_name)
        polices.append(float(pr))
        polices.append(float(magnitude))
    return polices





def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]

    return CutoutAbs(img, v)

    # x0 = np.random.uniform(w - v)
    # y0 = np.random.uniform(h - v)
    # xy = (x0, y0, x0 + v, y0 + v)
    # color = (127, 127, 127)
    # img = img.copy()
    # PIL.ImageDraw.Draw(img).rectangle(xy, color)
    # return img


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def augment_list():  # 16 operations and their ranges
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]
    if for_autoaug:
        l += [
            (CutoutAbs, 0, 20),  # compatible with auto-augment
            (Posterize2, 0, 4),  # 9
            (TranslateXAbs, 0, 10),  # 9
            (TranslateYAbs, 0, 10),  # 9
        ]
    return l


def get_augment_name_list():
    aug_list = augment_list()
    aug_name_list = [fn.__name__ for fn, _, _ in aug_list]
    return aug_name_list
    


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)




class data_generator:
    
    def __init__(self, img_root_dir, image_size, classes, batch_size, 
                 polices_str, max_sample_per_class):
        self.index=0
        self.img_root_dir = img_root_dir
        self.batch_size=batch_size
        self.image_size=image_size
        self.classes=classes
        self.polices = str2polices(polices_str)
        self.max_sample_per_class = max_sample_per_class
        self.load_images_labels(self.img_root_dir)

        # show the basic infomation of dataset
        idx = 0
        for label_name in os.listdir(self.img_root_dir):
            print("label index: {}, label name: {}".format(idx, label_name)) 
            idx += 1

    def load_images_labels(self, img_root_dir):
        # set random seed
        seed = int(random.uniform(1,1000))

        labels_dir_list = []
        labels_name_list = []
        for label_name in os.listdir(img_root_dir):
            labels_name_list.append(label_name)
            labels_dir_list.append(os.path.join(img_root_dir, label_name))

        self.imgs_path_list = []
        self.labels = []
        for idx, label_dir in enumerate(labels_dir_list):
            cur_path_list = []
            cur_labels_list = []
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                label = np.zeros((self.classes))
                label[idx] = 1
                cur_path_list.append(img_path)
                cur_labels_list.append(label)

            # random sample $max_sample_per_class samples
            if len(cur_path_list) <= self.max_sample_per_class or self.max_sample_per_class == -1:
                rand_index_list = np.arange(len(cur_path_list))
            else:
                rand_index_list = np.random.choice(len(cur_path_list), self.max_sample_per_class, replace=True)
            for rand_idx in rand_index_list:
                img_path = cur_path_list[rand_idx]
                label = cur_labels_list[rand_idx]
                self.imgs_path_list.append(img_path)
                self.labels.append(label)
        self.num_of_examples = len(self.labels)

        # shuffle
        np.random.seed(seed)
        np.random.shuffle(self.imgs_path_list)
        np.random.seed(seed)
        np.random.shuffle(self.labels)


    def do_augment(self, img):
        img = img.resize((self.image_size[0], self.image_size[1]),PIL.Image.ANTIALIAS)

        if len(self.polices) % 6 != 0:
            raise RuntimeError("polices format error")

        # random the police to use
        polices_num = int(len(self.polices) / 6)
        police_idx = int(random.random() * polices_num)

        idx = police_idx * 6

        # sub police 1
        police1_name = self.polices[idx]
        pr1 = self.polices[idx+1]
        level1 = self.polices[idx+2] 
        if random.random() <= pr1:
            img = apply_augment(img, police1_name, level1)

        # sub police 2
        police2_name = self.polices[idx]
        pr2 = self.polices[idx+1]
        level2 = self.polices[idx+2] 
        if random.random() <= pr2:
            img = apply_augment(img, police2_name, level2)

        img=np.array(img, dtype=np.int32)
        
        # normalize
        img = img / 255.0
        return img
    #def do_augment(self, img):
    #    img = img.resize((self.image_size[0], self.image_size[1]),PIL.Image.ANTIALIAS)

    #    for idx in range(0, len(self.polices), 3):
    #        police_name = self.polices[idx]
    #        pr = self.polices[idx+1]
    #        level = self.polices[idx+2] 

    #        if random.random() > pr:
    #            continue
    #        img = apply_augment(img, police_name, level)

    #    img=np.array(img, dtype=np.int32)
    #    
    #    # normalize
    #    img = img / 255.0
    #    return img

    
    def get_mini_batch(self, use_aug=False):
        while True:
            batch_images=[]
            batch_labels=[]
            i = 0
            while i < self.batch_size:
            #for i in range(self.batch_size):
                if(self.index==len(self.labels)): 
                    self.index=0
                    # load data and shuffle again
                    self.load_images_labels(self.img_root_dir)
                img_path = self.imgs_path_list[self.index]
                label = self.labels[self.index]

                try:
                    img=PIL.Image.open(img_path)

                    # check channel
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    if use_aug == True:
                        img = self.do_augment(img)
                    else:
                        img = img.resize((self.image_size[0], self.image_size[1]),PIL.Image.ANTIALIAS)
                        img = np.array(img)
                        img = img / 255.0
                except:
                    print("warning: error occur in get_mini_batch")
                    self.index += 1
                    continue

                batch_images.append(img)
                batch_labels.append(self.labels[self.index])
                self.index+=1
                i += 1
            batch_images=np.array(batch_images)
            batch_labels=np.array(batch_labels)
            yield batch_images, batch_labels

