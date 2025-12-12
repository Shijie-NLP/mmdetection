from mmengine.config import read_base


with read_base():
    from ..faster_rcnn.faster_rcnn_r50_fpn_1x_coco import *  # noqa

from mmdet.models.necks import PAFPN


model = model  # noqa: F405


model.update(
    dict(
        neck=dict(
            type=PAFPN,
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
        )
    )
)
