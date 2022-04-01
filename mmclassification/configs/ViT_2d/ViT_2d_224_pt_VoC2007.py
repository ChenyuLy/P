_base_ = [
    '../_base_/models/ViT_2db.py',
    '../_base_/datasets/voc_bs16.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]
# train_cfg = dict(
#     augments=dict(type='BatchMixup', alpha=0.2, num_classes=20,
#                   prob=1.),
#     )
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='vision_2dtransformer'),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=20,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        topk=(1, 5),
        init_cfg=dict(type='TruncNormal', layer='Linear', std=.02)))