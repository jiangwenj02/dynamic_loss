_base_ = [
    'pipelines/auto_aug_cifar.py',
]
# dataset settings
dataset_type = 'CIFAR100Cor'
img_norm_cfg = dict(
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='AutoAugment', policies={{_base_.policy_cifar10}}),
    dict(type='Cutout', shape=16, prob=1.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='ToTensor', keys=['true_label']),
    dict(type='ToTensor', keys=['idx']),
    dict(type='ToTensor', keys=['rank']),
    dict(type='Collect', keys=['img', 'gt_label', 'true_label', 'idx', 'rank'])
]
weak_train_pipeline = [
    # dict(type='RandomCrop', size=32, padding=4),
    # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='ToTensor', keys=['true_label']),
    dict(type='ToTensor', keys=['idx']),
    dict(type='ToTensor', keys=['rank']),
    dict(type='Collect', keys=['img', 'gt_label', 'true_label', 'idx', 'rank'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

noise_rate = 0
imb_ratio = 1
data = dict(
    samples_per_gpu=512,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix='data/cifar100',
        noise_rate=noise_rate, imb_ratio=imb_ratio,
        pipeline=train_pipeline,
        weak_pipeline=weak_train_pipeline),
    val=dict(
        type=dataset_type, data_prefix='data/cifar100', pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type, data_prefix='data/cifar100', pipeline=test_pipeline,
        test_mode=True))
