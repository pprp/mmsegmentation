_base_ = [
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    # pretrained='open-mmlab://resnet50_v1c',
    # pretrained='/HOME/scz0088/run/datasets/weights/model_r50_rf.pth.tar',
    # pretrained='/HOME/scz0088/run/project/mmsegmentation/work_dirs/resnet50_rf/model_r50_rf.pth.tar',
    # pretrained='/HOME/scz0088/run/project/mmsegmentation/work_dirs/seresnet50/model_seresnet5.pth.tar',
    pretrained='/HOME/scz0088/run/project/mmsegmentation/work_dirs/resnet50_cbam/model_r50_cbam.pth',
    backbone=dict(
        type='ResNet_Att',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        att='CBAM'),
    # decode_head=dict(
    #     type='ASPPHead',
    #     in_channels=2048,
    #     in_index=3,
    #     channels=512,
    #     dilations=(1, 12, 24, 36),
    #     dropout_ratio=0.1,
    #     num_classes=19,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    decode_head=dict(
        type='FCNHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
