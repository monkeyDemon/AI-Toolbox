#!/bin/bash

# convert ckpt to savemodel format pb
# savemodel format pb file is used for TF-Serving

_CUDA_VISIBLE_DEVICES='6'
_ACTION='convert'  # print of convert
_CHECKPOINT_PATH='./log/padding_best/resnet50-zyb_v1-epoch21.ckpt-161784'
_PB_MODEL_SAVE_PATH='./log/padding_best/frozen/savemodel_base64'
python src/convert_ckpt2pb_savemodel.py \
    --action $_ACTION \
    --checkpoint_path $_CHECKPOINT_PATH \
    --pb_model_save_path $_PB_MODEL_SAVE_PATH \
    --cuda_visible_devices $_CUDA_VISIBLE_DEVICES 
