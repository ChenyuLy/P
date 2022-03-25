import torch
import torch.nn as nn

from ..builder import HEADS
from .decode_head import BaseDecodeHead

#
# @HEADS.register_module()
# class  CUSTOMHead(BaseDecodeHead):