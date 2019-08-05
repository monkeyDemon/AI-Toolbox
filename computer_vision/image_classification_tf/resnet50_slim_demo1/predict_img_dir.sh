#!/bin/bash

# predict the specified image directory
echo "start predicting by src/predict.py ..."

# model path can be both .pb or .ckpt or directory of savemodel
model_path='log/train_tf_log/trained model ckpt path'
#model_path='log/train_tf_log/frozen/frozen inference graph pb file' 
#model_path='log/train_tf_log/frozen/directory of savemodel pb' 
weight_format='ckpt'  # ckpt, pb or savemodel
test_img_dir='dataset/testset'
result_dir='test_predict'
devices='0'

#nohup python src/predict_img_dir.py \
#    --weight_path $model_path \
#    --weight_format $weight_format \
#    --images_dir $test_img_dir \
#    --output_dir $result_dir \
#    --gpu_device $devices \
#    > log/test_record.out 2>&1 &
python src/predict_img_dir.py \
    --weight_path $model_path \
    --weight_format $weight_format \
    --images_dir $test_img_dir \
    --output_dir $result_dir \
    --gpu_device $devices 
