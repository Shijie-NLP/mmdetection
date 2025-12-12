# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base


with read_base():
    from .detr_r50_8xb2_150e_coco import *  # noqa


from mmengine.optim.scheduler.lr_scheduler import MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop


train_cfg = train_cfg  # noqa: F405
default_hooks = default_hooks  # noqa: F405


# learning policy
max_epochs = 500
train_cfg.update(type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=10)

param_scheduler = [dict(type=MultiStepLR, begin=0, end=max_epochs, by_epoch=True, milestones=[334], gamma=0.1)]

# only keep latest 2 checkpoints
default_hooks.update(checkpoint=dict(max_keep_ckpts=2))
