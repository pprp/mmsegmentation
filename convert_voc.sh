#!/bin/bash 

module load cuda/10.1
module load anaconda
source activate pool 

python tools/convert_datasets/voc_aug.py /data/public/PascalVOC/2012/VOC2012 /data/public/PascalVOC/2012/VOC2012aug --nproc 8 
