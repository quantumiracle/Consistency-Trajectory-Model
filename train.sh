#!/bin/bash
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
# mkdir -p log/$DATE


nohup python -W ignore  ctm_train_cifar.py >> log/$DATE.log &        
