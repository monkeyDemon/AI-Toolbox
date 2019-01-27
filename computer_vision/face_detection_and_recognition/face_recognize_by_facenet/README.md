# Face Recognition Demo by FaceNet

Here is a face recognition demo base on Google's FaceNet.

You need to specify the target persons, then our demo will help you find these people among a large number of images.

In our work, we use this program to detect the user's head sculptur which infringes the right of celebrity portrait.

Here are part of the illegal images we detected.

![chenduxiu](https://raw.githubusercontent.com/wiki/monkeyDemon/AI-Toolbox/readme_image/face_recognize/chenduxiu.png)

# Usage

## 1. download the code of FaceNet

Our demo is base on FaceNet, so download FaceNet's source code from [github](https://github.com/davidsandberg/facenet/tree/master).


## 2. set environmental variable

Add these two line at the end of `~/.bashrc` to add FaceNet and our demo's root path into the environmental variable(you should use your own path).

```
export PYTHONPATH=$PATH:/.../.../facenet-master/src
export PYTHONPATH=$PYTHONPATH:/.../.../face_recognize_by_facenet/
```

## 3. download the pretrained FaceNet model

Download the pretrained facenet model from [here](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-).

If you are Chinese, you can download [here](https://download.csdn.net/download/rookie_wei/10609076).

After download finish, put them in `./facenet_pretrained_model`, and this directory should contains four file:

20180402-114759.pb

model-20180402-114759.ckpt-275.data-00000-of-00001

model-20180402-114759.ckpt-275.index

model-20180402-114759.meta

## 4. set basic parameters

Set the basic parameters for our model in `sys_config.py`

Remember to modify the following two path parameters according to your actual environment.

```
facenet_weight_dir = '/.../.../face_recognize_by_facenet/facenet_pretrained_model'
template_face_embeddings_file = '/.../.../face_recognize_by_facenet/template/template_face_embeddings.npy'
```

You can use default values for other parameters.

## 5. prepare template image

Put the target person's face image in `./template/src_template`

Run `template_feature_extract.py`

Then the programme will generate two file in `./template`

`target_people_names.npy` and `template_face_embeddings.npy`

`target_people_names.npy` saves each target person's name and `template_face_embeddings.npy` saves these people's corresponding face feature.

## 6. run test demo

Put the test images in `./test/test_img`

Run `test_face_detect.py` to test the effect of face detection

or

Run `test_face_recognition.py` to test the effect of face recognition

The test result will be save in `./test/temp`
