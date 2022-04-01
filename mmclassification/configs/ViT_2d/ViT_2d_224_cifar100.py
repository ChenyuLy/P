_base_ = [
    '../_base_/models/ViT_2db.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py',
    '../_base_/default_runtime.py'
]
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='vision_2dtransformer',
        img_size=32, ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        topk=(1, 5),

        init_cfg=dict(type='TruncNormal', layer='Linear', std=.02)))

train_cfg = dict(
    augments=dict(type='BatchMixup', alpha=0.2, num_classes=10,
                  prob=1.))
# img_norm_cfg = dict(
#     mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='RandomResizedCrop', size=224, backend='pillow'),
#     dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'])
# ]
#
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', size=(224, -1), backend='pillow'),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img'])
# ]
#
# data = dict(
#     train=dict(pipeline=train_pipeline),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline),
# )
