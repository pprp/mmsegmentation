#!/bin/bash 

module load anaconda/2021.05
module load  cuda/11.1
module load cudnn/8.2.1_cuda11.x
module load gcc/9.3 
source activate hb

export PYTHONUNBUFFERED=1

# bash ./tools/dist_train.sh ./configs/deeplabv3/deeplabv3_r50-d8_769x769_40k_cityscapes.py 
# bash ./tools/dist_train.sh configs/deeplabv3/rf_deeplabv3_r18b-d8_512x1024_80k_cityscapes.py 1
# bash ./tools/dist_train.sh configs/deeplabv3/se_deeplabv3_r18b-d8_512x1024_80k_cityscapes.py 1
# bash ./tools/dist_train.sh configs/deeplabv3/cbam_deeplabv3_r18b-d8_512x1024_80k_cityscapes.py 1


# baseline r50
# bash ./tools/dist_train.sh configs/deeplabv3/pprp_deeplabv3_r50-d8_512x1024_40k_cityscapes.py 4 --work-dir ./work_dirs/pprp_deeplabv3_r50-d8_512x1024_40k_cityscapes_baseline_rerun

# r50 + RF
# bash ./tools/dist_train.sh configs/deeplabv3/pprp_deeplabv3_r50-d8_512x1024_40k_cityscapes.py 4 --work-dir ./work_dirs/pprp_deeplabv3_r50-d8_512x1024_40k_cityscapes_RF_rerun_fix_noise

# r50 + SE 
# bash ./tools/dist_train.sh configs/deeplabv3/pprp_deeplabv3_r50-d8_512x1024_40k_cityscapes.py 4 --work-dir ./work_dirs/pprp_deeplabv3_r50-d8_512x1024_40k_cityscapes_SE_rerun

# r50 + CBAM 
# bash ./tools/dist_train.sh configs/deeplabv3/pprp_deeplabv3_r50-d8_512x1024_40k_cityscapes.py 4 --work-dir ./work_dirs/pprp_deeplabv3_r50-d8_512x1024_40k_cityscapes_CBAM_rerun 

# 重新复现原始结果 
# bash ./tools/dist_train.sh configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py 4 --work-dir ./work_dirs/deeplabv3_r50-d8_512x1024_40k_cityscapes_baseline

# r50 + RF 继续跑 --auto-resume
bash ./tools/dist_train.sh configs/deeplabv3/pprp_deeplabv3_r50-d8_512x1024_40k_cityscapes.py 4 --work-dir ./work_dirs/pprp_deeplabv3_r50-d8_512x1024_40k_cityscapes_RF_rerun_fix_noise_continue_50k
