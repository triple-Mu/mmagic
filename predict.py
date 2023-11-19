# 代码示例
# python predict.py [src_image_dir] [results] [infer_time]

import os
import sys
import cv2
import time
import glob

import numpy as np

# import subprocess
# import importlib
# try:
#     subprocess.check_output(f'pip install --no-cache ./paddlepaddle_gpu-2.5.1.post112-cp37-cp37m-linux_x86_64.whl',
#                             shell=True)
#     paddle = importlib.import_module('paddle')
# except Exception:
#     import paddle

from paddle.inference import Config
from paddle.inference import create_predictor
from tqdm import tqdm

np.random.seed(0)


def init_model():
    config = Config('inference_model/model.pdmodel', 'inference_model/model.pdiparams')
    config.enable_memory_optim()
    config.enable_use_gpu(10000, 0)
    config.switch_ir_optim()
    config.glog_info_disabled()
    predictor = create_predictor(config)
    return predictor


def process(model, input_tensor, output_tensor, src_image_dir, save_dir, infer_time):
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    infer_time = float(infer_time)

    for image_path in tqdm(image_paths):
        filename = os.path.basename(image_path)
        img = cv2.imread(image_path)
        before_time = time.time()

        img = np.ascontiguousarray(img.transpose([2, 0, 1])[np.newaxis, ::-1, :, :], dtype=np.float32)
        input_tensor.copy_from_cpu(img)
        model.run()
        output_data = output_tensor.copy_to_cpu()
        output = output_data[0, ::-1, :, :].transpose([1, 2, 0]).round()
        new_img = np.ascontiguousarray(output, dtype=np.uint8)
        infer_time += (time.time() - before_time)
        cv2.imwrite(os.path.join(save_dir, filename.replace('.jpg', '.png')), new_img)
    return infer_time


if __name__ == "__main__":
    assert len(sys.argv) == 4

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]
    infer_time = sys.argv[3]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = init_model()

    input_name = model.get_input_names()[0]
    output_name = model.get_output_names()[0]
    input_tensor = model.get_input_handle(input_name)
    input_tensor.reshape([1, 3, 1024, 1024])
    output_tensor = model.get_output_handle(output_name)

    for i in range(30):
        tmp_input = np.random.randint(0, 255, (1, 3, 1024, 1024)).astype(np.float32)
        input_tensor.copy_from_cpu(tmp_input)
        model.run()
        output_data = output_tensor.copy_to_cpu()

    all_infer_time = process(model, input_tensor, output_tensor, src_image_dir, save_dir, infer_time)
    print('all_infer_time:', all_infer_time)
    # zip submit_1010_v1.zip model predict.py -r -y
