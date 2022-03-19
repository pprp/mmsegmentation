#!/bin/bash 

module load anaconda/2021.05
module load  cuda/11.1
module load cudnn/8.2.1_cuda11.x
module load gcc/9.3 
source activate hb

export PYTHONUNBUFFERED=1

# bash ./tools/dist_train.sh ./configs/deeplabv3/deeplabv3_r50-d8_769x769_40k_cityscapes.py 
bash ./tools/dist_train.sh configs/deeplabv3/rf_deeplabv3_r18b-d8_512x1024_80k_cityscapes.py 1

# bash ./tools/dist_train.sh configs/deeplabv3/se_deeplabv3_r18b-d8_512x1024_80k_cityscapes.py 1

# bash ./tools/dist_train.sh configs/deeplabv3/cbam_deeplabv3_r18b-d8_512x1024_80k_cityscapes.py 1