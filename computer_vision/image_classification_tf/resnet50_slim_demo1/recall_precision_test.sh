#!/bin/bash

# test recall and precision
echo "start predicting by src/recall_precision_test.py ..."

# model path can be both .pb or .ckpt
#model_path='log/train_tf_log/frozen/frozen inference graph pb file' 
model_path='log/train_tf_log/trained model ckpt path'

positive_img_dir='dataset/testset/positive_img_dir'
negative_img_dir='dataset/testset/negative_img_dir'

result_dir='test_recall_precision'
devices='0'
#nohup python src/recall_precision_test.py \
#    --weight_path $model_path \
#    --positive_img_dir $positive_img_dir \
#    --negative_img_dir $negative_img_dir \
#    --output_dir $result_dir \
#    --gpu_device $devices \
#    > log/test_record.out 2>&1 &
python src/recall_precision_test.py \
    --weight_path $model_path \
    --positive_img_dir $positive_img_dir \
    --negative_img_dir $negative_img_dir \
    --output_dir $result_dir \
    --gpu_device $devices 
