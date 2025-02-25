_base_ = './deeplabv3_r50-d8_512x1024_80k_cityscapes.py'
model = dict(
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet_Att', 
                 depth=18, 
                 att="SE"),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
