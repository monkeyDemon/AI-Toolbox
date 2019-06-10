#!/bin/bash


# test recall and precision
echo "start predicting by src/recall_precision_test.py ..."
# model path can be both .pb or .ckpt
#model_path='log/train_tf_log/frozen/frozen_inference_graph.pb' # TODO: now use pb file has wrong pridict bug
model_path='log/head_porn_v2/resnet50-epoch16.ckpt-130464'
positive_img_dir='positice img dir'
negative_img_dir='negative img dir'
result_dir='test'
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






# predict the specified image directory
#echo "start predicting by src/predict.py ..."
## model path can be both .pb or .ckpt
##model_path='log/train_tf_log/frozen/frozen_inference_graph.pb' # TODO: now use pb file has wrong pridict bug
#model_path='log/train_tf_log/resnet50-epoch17.ckpt-93534'
#test_img_dir='test_img_dir'
#result_dir='test'
#devices='0'
##nohup python src/predict.py \
##    --weight_path $model_path \
##    --images_dir $test_img_dir \
##    --output_dir $result_dir \
##    --gpu_device $devices \
##    > log/test_record.out 2>&1 &
#python src/predict.py \
#    --weight_path $model_path \
#    --images_dir $test_img_dir \
#    --output_dir $result_dir \
#    --gpu_device $devices 
