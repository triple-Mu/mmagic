default_scope = 'mmagic'
save_dir = './work_dirs/'

experiment_name = 'unet'
work_dir = f'./work_dirs/{experiment_name}'

load_from = None
resume = False

max_iter = 200000
val_iter = 1000
checkpoint_iter = 1000
log_iter = 500

log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=False)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='UNetBaidu',
        in_channels=3,
        out_channels=3,
        num_down=8,
        base_channels=16,
        norm_cfg=dict(type='BN'),
        use_dropout=True,
    ),
    # L1Loss, CharbonnierLoss, PSNRLoss, MSELoss
    pixel_loss=dict(type='CharbonnierLoss', reduction='sum'),
    train_cfg=dict(),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
    ))

train_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb', use_cache=True),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb', use_cache=True),
    # dict(type='Resize', keys=['img', 'gt'], scale=(256, 256)),
    dict(type='SetValues', dictionary=dict(scale=1)),
    dict(type='NAFNetTransform', keys=['img', 'gt']),
    dict(type='PackInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb', use_cache=True),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb', use_cache=True),
    # dict(type='Resize', keys=['img', 'gt'], scale=(256, 256)),
    dict(type='PackInputs')
]
# dataset settings
dataset_type = 'BasicImageDataset'

train_dataloader = dict(
    num_workers=8,
    batch_size=64,  # gpus 4
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='sidd', task_name='denoising'),
        data_root='./data/SIDD/train',
        data_prefix=dict(gt='target', img='input'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=8,
    batch_size=1,
    persistent_workers=False,
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
    dict(type='PSNR', crop_border=4, convert_to='Y'),
    dict(type='SSIM', crop_border=4, convert_to='Y'),
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
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.0, betas=(0.9, 0.9)))

# learning policy
param_scheduler = dict(
    type='CosineAnnealingLR', by_epoch=False, T_max=max_iter, eta_min=8e-7)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=checkpoint_iter,
        save_best='PSNR',
        save_optimizer=True,
        rule='greater',
        by_epoch=False,
        out_dir=save_dir,
        max_keep_ckpts=30),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=log_iter),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=False)

custom_hooks = [dict(type='BasicVisualizationHook', interval=100000)]
