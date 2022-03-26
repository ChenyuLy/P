# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Vison_transformer'),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        topk=(1, 5),
        init_cfg=dict(type='TruncNormal', layer='Linear', std=.02)))
