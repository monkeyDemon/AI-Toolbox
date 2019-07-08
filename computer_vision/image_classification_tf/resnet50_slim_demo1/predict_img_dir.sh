#!/bin/bash

# predict the specified image directory
echo "start predicting by src/predict.py ..."

# model path can be both .pb or .ckpt
#model_path='log/train_tf_log/frozen/frozen inference graph pb file' 
model_path='log/train_tf_log/trained model ckpt path'

test_img_dir='dataset/testset'
result_dir='test_predict'
devices='0'

#nohup python src/predict_img_dir.py \
#    --weight_path $model_path \
#    --images_dir $test_img_dir \
#    --output_dir $result_dir \
#    --gpu_device $devices \
#    > log/test_record.out 2>&1 &
python src/predict_img_dir.py \
    --weight_path $model_path \
    --images_dir $test_img_dir \
    --output_dir $result_dir \
    --gpu_device $devices 
