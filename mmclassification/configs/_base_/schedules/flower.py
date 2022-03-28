# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.00001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr_ratio=1e-5)
runner = dict(type='EpochBasedRunner', max_epochs=200)
