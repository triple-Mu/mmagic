_base_ = '../_base_/default_runtime.py'

experiment_name = 'nafnet_plus_n'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'
max_iter = 200_000
val_iter = 200
checkpoint_iter = 200
log_iter = 100

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='NAFNet',
        img_channels=3,
        mid_channels=16,
        enc_blk_nums=[2, 2, 1, 1],
        middle_blk_num=1,
        dec_blk_nums=[2, 2, 2, 2],
    ),
    pixel_loss=dict(
        type='PSNRLoss'),  # L1Loss, CharbonnierLoss, PSNRLoss, MSELoss
    train_cfg=dict(),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
    ))

train_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=1)),
    dict(type='NAFNetTransform', keys=['img', 'gt']),
    dict(type='PackInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]
# dataset settings
dataset_type = 'BasicImageDataset'

train_dataloader = dict(
    num_workers=8,
    batch_size=8,  # gpus 4
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='sidd', task_name='denoising'),
        data_root='./data/SIDD/train',
        data_prefix=dict(gt='target', img='input'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=4,
    batch_size=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='sidd', task_name='denoising'),
        data_root='./data/SIDD/val/',
        data_prefix=dict(gt='target', img='input'),
        pipeline=val_pipeline))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='PSNR'),
    dict(type='SSIM'),
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iter, val_interval=val_iter)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=1e-3, betas=(0.9, 0.9)))

# learning policy
param_scheduler = dict(
    type='CosineAnnealingLR', by_epoch=False, T_max=max_iter, eta_min=1e-8)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=checkpoint_iter,
        save_best='PSNR',
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
        max_keep_ckpts=30),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=log_iter),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

visualizer = dict(bgr2rgb=False)
