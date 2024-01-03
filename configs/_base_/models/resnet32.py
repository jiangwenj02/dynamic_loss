# model settings
num_classes=10
model = dict(
        type='ImageClassifier',
        backbone=dict(
            type='resnet32',
            ),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=num_classes,
            in_channels=64,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        ))
