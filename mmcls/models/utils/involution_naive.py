import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer, build_conv_layer
from mmcv.runner import BaseModule

class involution(BaseModule):

    def __init__(self,
                 channels,
                 kernel_size,
                 stride):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels // reduction_ratio,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size**2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out

def _sample_logistic(shape, out=None):

    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    #U2 = out.resize_(shape).uniform_() if out is not None else th.rand(shape)

    return torch.log(U) - torch.log(1-U)


def _sigmoid_sample(logits, tau=1):
    """
    Implementation of Bernouilli reparametrization based on Maddison et al. 2017
    """
    dims = logits.dim()
    logistic_noise = _sample_logistic(logits.size(), out=logits.data.new())
    y = logits + logistic_noise
    return torch.sigmoid(y / tau)


def gumbel_sigmoid(logits, tau=1, hard=False):

    shape = logits.size()
    y_soft = _sigmoid_sample(logits, tau=tau)
    if hard:
        zeros_tensor = logits.data.zero_()
        ones_tensor = logits.data.zero_() + 1
        y_hard = torch.where(y_soft > 0.5, ones_tensor, zeros_tensor)
        y = y_hard.data - y_soft.data + y_soft
    else:
        y = y_soft
    return y

class aug_layer(BaseModule):
    """Bottleneck block for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): ``"pytorch"`` or ``"caffe"``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 reduction_ratio=2,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 downsample_ratio=1,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(aug_layer, self).__init__()
        assert style in ['pytorch', 'caffe']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = in_channels // reduction_ratio
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, self.out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = involution(self.mid_channels, 7, self.conv2_stride)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            self.out_channels,
            kernel_size=1,
            bias=True)
        # self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.downsample_ratio = downsample_ratio

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            # out = self.norm3(out)

            return out
        B, _, H, W = x.shape
        if self.downsample_ratio is not None:
            x = F.interpolate(x, size=(H // self.downsample_ratio, W // self.downsample_ratio), mode='bilinear', align_corners=True)
        x = _inner_forward(x)
        x = gumbel_sigmoid(x, hard=True)
        # import pdb
        # pdb.set_trace()
        # if self.downsample_ratio is not None:
        return x
