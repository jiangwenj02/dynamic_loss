_base_ = [
    '../_base_/models/resnet32.py', '../_base_/datasets/cor_cifar10.py',
    '../_base_/schedules/meta_cifar10_adam.py', '../_base_/default_runtime.py'
]

model = dict(
        head=dict(
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0, reduction='none'),
        ))

warm_epoch=5
rank_num=100

meta = dict(type='DyanamicMeta',
            metanet=dict(type='DynamicLoss', 
                    use_LC=True,
                    use_MG=True,
                    LC_classwise=True,
                    h_dim=64,
                    rank_num=rank_num))

# checkpoint saving
optimizer = dict(
    _delete_=True,
    classifier=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4),
    metanet=dict(type='Adam', lr=3e-3, weight_decay=0))
lr_config = dict(
    _delete_=True,
    classifier=dict(
            policy='CosineAnnealing',
            min_lr=0,
            warmup='linear',
            warmup_iters=5,
            warmup_ratio=0.1,
            warmup_by_epoch=True),
    metanet=dict(policy='fixed'))
# dataset settings
noise_rate = 0.4
imb_ratio = 0.1
data = dict(train=dict(noise_rate=noise_rate, imb_ratio=imb_ratio))

# checkpoint saving
checkpoint_config = dict(interval=5)
log_config = dict(interval=20)
runner = dict(max_epochs=325, warm_epoch=warm_epoch, rank_num=rank_num)