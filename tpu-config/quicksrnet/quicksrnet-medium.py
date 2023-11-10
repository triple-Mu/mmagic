default_scope = 'mmagic'
log_level = 'INFO'

load_from = 'weights/checkpoint_float32.pth.tar'
resume = False

scale = 4
num_workers = 8
batch_size = 32
log_iter = 100
checkpoint_iter = 200
total_iter = 100_000
base_lr = 1e-3

# dataset settings
dataset_type = 'BasicImageDataset'
data_root = 'data/tpu_data'

experiment_name = 'quicksrnet-medium'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='QuickSRNetMedium',
        scaling_factor=scale,
        in_channels=3,
        out_channels=3),
    # L1Loss, CharbonnierLoss, PSNRLoss, MSELoss
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_pipeline = [
    # dict(
    #     type='LoadImageFromFile',
    #     key='img',
    #     color_type='color',
    #     channel_order='rgb'),
    # dict(
    #     type='LoadImageFromFile',
    #     key='gt',
    #     color_type='color',
    #     channel_order='rgb'),
    dict(type='LoadNpyFromFile', key='img', swap_channel=False),
    dict(type='LoadNpyFromFile', key='gt', swap_channel=False),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='PackInputs')
]

val_pipeline = [
    # dict(
    #     type='LoadImageFromFile',
    #     key='img',
    #     color_type='color',
    #     channel_order='rgb',
    #     imdecode_backend='cv2'),
    # dict(
    #     type='LoadImageFromFile',
    #     key='gt',
    #     color_type='color',
    #     channel_order='rgb',
    #     imdecode_backend='cv2'),
    dict(type='LoadNpyFromFile', key='img', swap_channel=False),
    dict(type='LoadNpyFromFile', key='gt', swap_channel=False),
    dict(type='PackInputs')
]

train_dataloader = dict(
    num_workers=num_workers * 2,
    batch_size=batch_size,
    persistent_workers=num_workers > 0,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='',
        metainfo=dict(dataset_type='tpu', task_name='sr'),
        data_root=data_root + '/train',
        data_prefix=dict(img='LR', gt='HR'),
        filename_tmpl=dict(img='{}', gt='{}'),
        img_suffix='.npy',
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=num_workers,
    batch_size=1,
    persistent_workers=num_workers > 0,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='',
        metainfo=dict(dataset_type='tpu', task_name='sr'),
        data_root=data_root + '/val',
        data_prefix=dict(img='LR', gt='HR'),
        img_suffix='.npy',
        pipeline=val_pipeline))

val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='MAE'),
        dict(type='PSNR', crop_border=scale),
        dict(type='SSIM', crop_border=scale),
    ])

test_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='MAE'),
        dict(type='PSNR', crop_border=scale),
        dict(type='SSIM', crop_border=scale),
    ])

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=total_iter,
    val_interval=checkpoint_iter)

val_cfg = dict(type='MultiValLoop')

test_cfg = dict(type='MultiTestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=base_lr, betas=(0.9, 0.999)))

# learning policy
param_scheduler = dict(
    type='CosineRestartLR',
    by_epoch=False,
    periods=[
        total_iter // 4, total_iter // 4, total_iter // 4, total_iter // 4
    ],
    restart_weights=[1, 1, 1, 1],
    eta_min=1e-7)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=log_iter),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=checkpoint_iter,
        save_best='PSNR',
        save_optimizer=True,
        rule='greater',
        by_epoch=False,
        out_dir=save_dir,
        max_keep_ckpts=30,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=100, by_epoch=False)

# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='ConcatImageVisualizer',
#     vis_backends=vis_backends,
#     fn_key='gt_path',
#     img_keys=['gt_img', 'input', 'pred_img'],
#     bgr2rgb=True)
# custom_hooks = [dict(type='BasicVisualizationHook', interval=1)]
