# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmrazor.models import ConvModuleConnector, L2Loss

from mmagic.registry import MODELS
from mmagic.structures import DataSample


@MODELS.register_module()
class BaseEditModel(BaseModel):
    """Base model for image and video editing.

    It must contain a generator that takes frames as inputs and outputs an
    interpolated frame. It also has a pixel-wise loss for training.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.

    Attributes:
        init_cfg (dict, optional): Initialization config dict.
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`. Default: None.
    """

    def __init__(self,
                 generator: dict,
                 pixel_loss: dict,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None):
        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generator
        self.generator = MODELS.build(generator)

        # loss
        self.pixel_loss = MODELS.build(pixel_loss)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor',
                **kwargs) -> Union[torch.Tensor, List[DataSample], dict]:
        """Returns losses or predictions of training, validation, testing, and
        simple inference process.

        ``forward`` method of BaseModel is an abstract method, its subclasses
        must implement this method.

        Accepts ``inputs`` and ``data_samples`` processed by
        :attr:`data_preprocessor`, and returns results according to mode
        arguments.

        During non-distributed training, validation, and testing process,
        ``forward`` will be called by ``BaseModel.train_step``,
        ``BaseModel.val_step`` and ``BaseModel.val_step`` directly.

        During distributed data parallel training process,
        ``MMSeparateDistributedDataParallel.train_step`` will first call
        ``DistributedDataParallel.forward`` to enable automatic
        gradient synchronization, and then call ``forward`` to get training
        loss.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``. Default: 'tensor'.

                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of ``BaseDataElement`` results used for
                  computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.

        Returns:
            ForwardResults:

                - If ``mode == loss``, return a ``dict`` of loss tensor used
                  for backward and logging.
                - If ``mode == predict``, return a ``list`` of
                  :obj:`BaseDataElement` for computing metric
                  and getting inference result.
                - If ``mode == tensor``, return a tensor or ``tuple`` of tensor
                  or ``dict`` or tensor for custom use.
        """
        if isinstance(inputs, dict):
            inputs = inputs['img']
        if mode == 'tensor':
            return self.forward_tensor(inputs, data_samples, **kwargs)

        elif mode == 'predict':
            predictions = self.forward_inference(inputs, data_samples,
                                                 **kwargs)
            predictions = self.convert_to_datasample(predictions, data_samples,
                                                     inputs)
            return predictions

        elif mode == 'loss':
            return self.forward_train(inputs, data_samples, **kwargs)

    def convert_to_datasample(self, predictions: DataSample,
                              data_samples: DataSample,
                              inputs: Optional[torch.Tensor]
                              ) -> List[DataSample]:
        """Add predictions and destructed inputs (if passed) to data samples.

        Args:
            predictions (DataSample): The predictions of the model.
            data_samples (DataSample): The data samples loaded from
                dataloader.
            inputs (Optional[torch.Tensor]): The input of model. Defaults to
                None.

        Returns:
            List[DataSample]: Modified data samples.
        """

        if inputs is not None:
            destructed_input = self.data_preprocessor.destruct(
                inputs, data_samples, 'img')
            data_samples.set_tensor_data({'input': destructed_input})
        # split to list of data samples
        data_samples = data_samples.split()
        predictions = predictions.split()

        for data_sample, pred in zip(data_samples, predictions):
            data_sample.output = pred

        return data_samples

    def forward_tensor(self,
                       inputs: torch.Tensor,
                       data_samples: Optional[List[DataSample]] = None,
                       **kwargs) -> torch.Tensor:
        """Forward tensor. Returns result of simple forward.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Tensor: result of simple forward.
        """

        feats = self.generator(inputs, **kwargs)

        return feats

    def forward_inference(self,
                          inputs: torch.Tensor,
                          data_samples: Optional[List[DataSample]] = None,
                          **kwargs) -> DataSample:
        """Forward inference. Returns predictions of validation, testing, and
        simple inference.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            DataSample: predictions.
        """

        feats = self.forward_tensor(inputs, data_samples, **kwargs)
        feats = self.data_preprocessor.destruct(feats, data_samples)

        # create a stacked data sample here
        predictions = DataSample(pred_img=feats.cpu())

        return predictions

    def forward_train(self,
                      inputs: torch.Tensor,
                      data_samples: Optional[List[DataSample]] = None,
                      **kwargs) -> Dict[str, torch.Tensor]:
        """Forward training. Returns dict of losses of training.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            dict: Dict of losses.
        """

        feats = self.forward_tensor(inputs, data_samples, **kwargs)
        batch_gt_data = data_samples.gt_img

        loss = self.pixel_loss(feats, batch_gt_data)

        return dict(loss=loss)



@MODELS.register_module()
class BaseTeacher(BaseEditModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = MODELS.build(
            dict(type='NAFNet', img_channels=3, mid_channels=64,
                   middle_blk_num=6, enc_blk_nums=[2, 2, 4, 6],
                   dec_blk_nums=[2, 2, 2, 2])
        ).eval()


        # TODO: 加载训练好的模型参数
        state_dict = torch.load('naf_paddle.pth', map_location=torch.device('cpu'))
        self.teacher_model.load_state_dict(state_dict['state_dict'])
        print('loading statedict')

        # 将teacher_model所有参数设置为不可训练
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # 加载connector
        self.con1 = ConvModuleConnector(16, 128, norm_cfg=None, act_cfg=None)
        self.con2 = ConvModuleConnector(64, 512, norm_cfg=None, act_cfg=None)
        self.dist_loss = L2Loss(loss_weight=10.)


    def forward_train(self,
                      inputs: torch.Tensor,
                      data_samples: Optional[List[DataSample]] = None,
                      **kwargs) -> Dict[str, torch.Tensor]:
        # print('xxx', inputs.shape)
        # 先推理得到teacher的特征
        with torch.no_grad():
            res, tea_encs = self.teacher_model(inputs)
            # !!!!!!!
            # [torch.Size([8, 64, 1024, 1024]), torch.Size([8, 128, 512, 512]), torch.Size([8, 256, 256, 256]), torch.Size([8, 512, 128, 128]), torch.Size([8, 512, 128, 128]), torch.Size([8, 256, 256, 256]), torch.Size([8, 128, 512, 512]), torch.Size([8, 64, 1024, 1024])]
            # [torch.Size([8, 32, 256, 256]), torch.Size([8, 64, 128, 128]), torch.Size([8, 128, 64, 64]), torch.Size([8, 64, 128, 128]), torch.Size([8, 32, 256, 256]), torch.Size([8, 16, 512, 512])]

            # print('!!!!!!!')
            # print([i.shape for i in tea_encs])

        # torch.Size([1, 32, 256, 256]), torch.Size([1, 64, 128, 128])
        feats, stu_encs = self.forward_tensor(inputs, data_samples, **kwargs)
        batch_gt_data = data_samples.gt_img
        # print([i.shape for i in stu_encs])

        # 蒸馏loss
        tea_feat1 = tea_encs[6]    # (1, 128, 256, 256)
        stu_feat1 = stu_encs[5]    # (1, 16, 256, 256)
        tea_feat2 = tea_encs[3]    # (1, 512, 128, 128)
        stu_feat2 = stu_encs[1]    # (1, 64, 128, 128)
        # print(tea_feat1.shape, stu_feat1.shape, tea_feat2.shape, stu_feat2.shape)
        con_feat1 = self.con1(stu_feat1)
        con_feat2 = self.con2(stu_feat2)
        dist_loss = self.dist_loss(con_feat1, tea_feat1) + self.dist_loss(con_feat2, tea_feat2)

        pixel_loss = self.pixel_loss(feats, batch_gt_data)

        return dict(pixel_loss=pixel_loss, dist_loss=dist_loss)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_model.eval()

        return self

    def forward_inference(self,
                          inputs: torch.Tensor,
                          data_samples: Optional[List[DataSample]] = None,
                          **kwargs) -> DataSample:

        feats, _ = self.forward_tensor(inputs, data_samples, **kwargs)
        feats = self.data_preprocessor.destruct(feats, data_samples)

        # create a stacked data sample here
        predictions = DataSample(pred_img=feats.cpu())

        return predictions

