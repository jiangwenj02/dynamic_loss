import mmcv
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule,constant_init, normal_init
import torch.nn.functional as F

class LocalAtten(nn.Module):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
    """

    def __init__(self,
                 channels,
                 padding_size=1,
                 atten_size=3,
                 ratio=16,
                 conv_cfg=None):
        super(LocalAtten, self).__init__()
        
        self.atten_size = atten_size
        self.inter_channels = int(channels / ratio)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.padding_size = padding_size
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg)
        self.conv2 = ConvModule(
            in_channels=channels,
            out_channels=self.atten_size * self.atten_size,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg)
        self.conv_out = ConvModule(
            in_channels=self.inter_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg)

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.conv1, self.conv2, self.conv_out]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std)

    def forward(self, x):
        N, _, H, W = x.shape
        
        x_ref = self.conv1(x)
        x_ref = F.pad(x_ref, (self.padding_size,self.padding_size,self.padding_size,self.padding_size), "constant", 0)
        x_ref = x_ref.unfold(2, self.atten_size, 1)
        x_ref = x_ref.unfold(3, self.atten_size, 1)
        x_ref = x_ref.permute(0, 2, 3, 1, 4, 5).reshape(N, H * W, self.inter_channels, self.atten_size * self.atten_size) # N, HW, C, SIZE^2

        x_attn = self.conv2(x).permute(0, 2, 3, 1).reshape(N, H*W, -1, 1) #N, HW, SIZE^2, 1
        x_attn = x_attn.softmax(-2)

        x_ref = torch.matmul(x_ref, x_attn) #N, HW, C, 1
        x_ref = x_ref.reshape(N, H, W, -1).permute(0, 3, 1, 2) #N, C, H, W
        x_ref = self.conv_out(x_ref)
        x = x + x_ref
        return x
