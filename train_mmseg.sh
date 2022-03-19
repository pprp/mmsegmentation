#!/bin/bash 

module load cuda/10.1
module load anaconda
source activate pool 

CONFIG_FILE=./configs/pspnet/pspnet_r50-d8_512x512_20k_voc12aug.py

./tools/dist_train.sh ${CONFIG_FILE} 1
