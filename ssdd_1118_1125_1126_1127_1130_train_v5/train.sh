#!/bin/bash

docker run \
--name ssdd_1118_1125_1126_1127_1130_train_v5_gpu5 \
--runtime=nvidia \
-d \
-it \
-e NVIDIA_VISIBLE_DEVICES=5 \
-v /home/samjin/working/deep_learning/repos/yodj_models:/tf/model \
-v /home/storage_disk2/saved_models/ssdd_1118_1125_1126_1127_1130_train_v5:/tf/saved_model \
-v /home/storage_disk2/datasets/private/lens_tube_1118_1125_1126_1127_1130_v5:/tf/dataset \
yodj/dlt \
python ./model/tf_2/segmentation/ssdd/train/train.py --config-path /tf/saved_model/train.yml

