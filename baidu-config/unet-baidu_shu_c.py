_base_ = './unet-baidu_shu.py'


experiment_name = 'unet_shu_c'
work_dir = f'./work_dirs/{experiment_name}'

# model settings
model = dict(
    # type='BaseEditModel',
    generator=dict(
        # type='UNetBaidu',
        # in_channels=3,
        # out_channels=3,
        # num_down=8,
        base_channels=26,
        num_down=5,
        # norm_cfg=dict(type='BN'),
        # use_dropout=True,
        # use_shu=True,
    ),
    # # L1Loss, CharbonnierLoss, PSNRLoss, MSELoss
    # pixel_loss=dict(type='L1Loss', reduction='mean'),
    # train_cfg=dict(),
    # test_cfg=dict(),
    # data_preprocessor=dict(
    #     type='DataPreprocessor',
    #     mean=[0.0, 0.0, 0.0],
    #     std=[255.0, 255.0, 255.0],
    # )
)


