# 百度网盘 AI 大赛-祛法令纹方向识别赛第 9 名方案


## 代码结构介绍
* 我们队伍采用 mmagic 作为训练框架。
* 在 mmagic 训练完毕后，将模型转为 onnx ，再将 onnx 转为 paddle model ，以进行线上提交。


## 代码clone、环境搭建
参照 mmagic 官方提供的命令，进行环境搭建：

代码 clone ：注意需要 clone baidu-disk/repvgg-unet 这个分支的代码。
```
git clone https://github.com/triple-Mu/mmagic.git -b baidu-disk/repvgg-unet
```

安装命令：
```
pip3 install openmim
mim install 'mmcv>=2.0.0'
mim install 'mmengine'
cd mmagic
pip3 install -e .
```


## 推理精度复现
模型权重下载：
链接: https://pan.baidu.com/s/1MbDQRreO40eo5YQxzEnFog?pwd=jq71 提取码: jq71 

文件介绍：
* submit_1114_53_6655.zip: B榜线上提交压缩包。里面包含:pdmodel、模型推理代码。
* best_PSNR_iter_169600_53.6655.pth: 使用 mmagaic 框架训练得到的 Pytorch 模型权重。
* repvgg-unetv6-step1.py: 最优配置 step1。
* repvgg-unetv6-step2.py: 最优配置 step2。


下载该文件后，解压 `submit_1114_53_6655.zip` ，使用以下命令，即可复现 B 榜表现最优结果：
```
python predict.py src_image_dir save_dir 0
```

## 训练精度复现

模型配置保存在 unet_config 文件夹内。

复现B榜最优成绩，需要使用以下命令：
```shell
# 一阶段训练
./tools/dist_train.sh unet_config/repvgg-unetv6-step1.py 4

# 在一阶段训练完成后，需要在 unet_config/repvgg-unetv6-ldmark-step2.py 中，将 load_from 变量的值填写为一阶段训练的最后保存的模型路径。
# 二阶段训练
./tools/dist_train.sh unet_config/repvgg-unetv6-step2.py 4
```


将训练好的模型转换为pdmodel:
```shell
# mmagic -> onnx  需手动将脚本中的模型路径填写为二阶段具有最高 PSNR 精度的模型路径
python export_onnx.py

# onnx -> pdmodel
pip install x2paddle
x2paddle --framework=onnx --model=model.onnx --save_dir=onnx2paddle
mv onnx2paddle/inference_model model
```


使用转换好的 pdmodel ，进行图像推理：
```shell
python predict.py src_image_dir save_dir 0
```

