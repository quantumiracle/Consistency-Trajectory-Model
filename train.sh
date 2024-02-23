#!/bin/bash

export CUDA_VISIBLE_DEVICES=1


DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
# mkdir -p log/$DATE


nohup python -W ignore  ctm_train_image.py >> log/$DATE.log &        
