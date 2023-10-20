_base_ = './unet-baidu_shu.py'
save_dir = './work_dirs/'

experiment_name = 'unet_shu_dfl'
work_dir = f'./work_dirs/{experiment_name}'

# model settings
model = dict(generator=dict(type='UNetBaiduDFL'))

train_dataloader = dict(
    batch_size=16,  # gpus 4
)
