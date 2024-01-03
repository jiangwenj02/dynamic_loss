from abc import abstractmethod

import torch.nn as nn
from ..builder import METANETS, build_metanet

class BaseMetaNet(nn.Module):
    """Base head.

    """

    def __init__(self):
        super(BaseMetaNet, self).__init__()

    def init_weights(self):
        pass

    @abstractmethod
    def forward_train(self, x, gt_label, **kwargss):
        pass
