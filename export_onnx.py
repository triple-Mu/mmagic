import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.runner import Runner
from io import BytesIO
import onnx
import onnxsim


class MyNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x / 255.
        x = self.model(x)
        x = torch.clip(x, 0., 1.)
        x = x * 255.
        return x


cfg = Config.fromfile('unet_config/repvgg-unetv6-step2.py')
runner = Runner.from_cfg(cfg)
generator = runner.model.generator
device = next(generator.parameters()).device
runner.load_checkpoint('work_dirs/repvgg-unetv6-ldmark-step2/best_PSNR_iter_82000.pth', device, strict=True)

generator = MyNet(generator)
generator.eval()

input = torch.randn([1, 3, 1024, 1024], device=device, dtype=torch.float32)
output = generator(input)
with BytesIO() as f:
    torch.onnx.export(generator, input, f, opset_version=13, input_names=['image'],
                      output_names=['new_image'])
    f.seek(0)
    onnx_model = onnx.load(f)
    onnx_model, _ = onnxsim.simplify(onnx_model)
    onnx.save(onnx_model, 'model.onnx')

'''
pip install x2paddle
x2paddle --framework=onnx --model=model.onnx --save_dir=onnx2paddle
mv onnx2paddle/inference_model model
'''
