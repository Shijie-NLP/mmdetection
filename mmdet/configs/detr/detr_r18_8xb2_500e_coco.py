# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base


with read_base():
    from .detr_r50_8xb2_500e_coco import *  # noqa

from mmengine.model.weight_init import PretrainedInit


model = model  # noqa: F405

model.update(
    dict(
        backbone=dict(
            depth=18,
            init_cfg=dict(type=PretrainedInit, checkpoint="torchvision://resnet18"),
        ),
        neck=dict(in_channels=[512]),
    )
)
