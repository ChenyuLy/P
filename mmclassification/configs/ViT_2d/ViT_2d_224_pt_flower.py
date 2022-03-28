_base_ = [
    '../_base_/models/ViT_2db.py',
    '../_base_/datasets/flower_photos_bs8.py',
    '../_base_/schedules/flower.py',
    '../_base_/default_runtime.py'
]
# train_cfg = dict(
#     augments=dict(type='BatchMixup', alpha=0.2, num_classes=20,
#                   prob=1.),
#     )
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Vison_transformer'),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        topk=(1, 5),
        init_cfg=dict(type='TruncNormal', layer='Linear', std=.02)))
runner = dict(type='EpochBasedRunner', max_epochs=30)