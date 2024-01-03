# optimizer
optimizer = dict(
    classifier=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005),
    metanet=dict(type='Adam', lr=3e-4, weight_decay=0))
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
# optimizer_config = dict(grad_clip=None)
optimizer_config = dict(type='MetaOptimizerHook')
# learning policy
lr_config = dict(
    classifier=dict(policy='CosineAnnealing',
                    min_lr=0,
                    warmup='linear',
                    warmup_iters=5,
                    warmup_ratio=0.1,
                    warmup_by_epoch=True),
    metanet=dict(policy='fixed'),
)
runner = dict(type='DynamicEpochBasedRunner',
                max_epochs=325,
                warm_epoch=5,
                second_random_select_ratio=0.5,
                first_random_select_ratio=0.5)
