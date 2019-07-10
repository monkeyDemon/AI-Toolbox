#!/bin/bash


_CUDA_VISIBLE_DEVICES='0'
_ACTION='convert'  # print of convert
_CHECKPOINT_PATH='path to trained model/resnet_v1_50.ckpt'
_PB_MODEL_SAVE_PATH='path to save the frozen pb model/frozen.pb'
python src/convert_ckpt2pb.py \
    --action $_ACTION \
    --checkpoint_path $_CHECKPOINT_PATH \
    --pb_model_save_path $_PB_MODEL_SAVE_PATH \
    --cuda_visible_devices $_CUDA_VISIBLE_DEVICES 
