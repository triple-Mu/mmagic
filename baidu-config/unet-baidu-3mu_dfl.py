_base_ = './unet-baidu-3mu.py'

experiment_name = 'unet-baidu-3mu_dfl'
work_dir = f'./work_dirs/{experiment_name}'

# model settings
model = dict(
    generator=dict(
        type='ReconstructiveSubNetworkDFL',
    ),
    # L1Loss, CharbonnierLoss, PSNRLoss, MSELoss
    # pixel_loss=dict(type='CharbonnierLoss', reduction='sum'),
)
