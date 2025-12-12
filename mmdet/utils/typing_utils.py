# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in mmdetection."""

from collections.abc import Sequence
from typing import Optional, Union

from mmengine.config import ConfigDict
from mmengine.structures import InstanceData, PixelData


# TODO: Need to avoid circular import with assigner and sampler
# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]

# Type hint of one or more config data
MultiConfig = Union[ConfigType, list[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

InstanceList = list[InstanceData]
OptInstanceList = Optional[InstanceList]

PixelList = list[PixelData]
OptPixelList = Optional[PixelList]

RangeType = Sequence[tuple[int, int]]
