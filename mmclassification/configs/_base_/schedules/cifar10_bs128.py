# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[85, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)
