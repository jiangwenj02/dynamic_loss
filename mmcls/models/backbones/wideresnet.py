import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from .base_backbone import BaseBackbone
from ..builder import BACKBONES

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.9)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.9)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

@BACKBONES.register_module()
class WideResNet(BaseBackbone):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        # self.linear = nn.Linear(nStages[3], num_classes)

        # self.apply(conv_init)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        outs = []
        out = self.conv1(x)
        out = self.layer1(out)
        outs.append(out)
        out = self.layer2(out)
        outs.append(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        outs.append(out)
        # out = F.avg_pool2d(out, 8)
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        # outs.append(out)
        return tuple(outs)

    def meta_forward(self, x, mask_dict):
        outs = []
        out = self.conv1(x)
        out = self.layer1(out)
        if 'auglayer1' in mask_dict:
            _, _, H, W = out.shape
            mask_dict['auglayer1']  = F.interpolate(mask_dict['auglayer1'] , size=(H, W), mode='nearest')
            out = out * mask_dict['auglayer1']            
        outs.append(out)
        
        out = self.layer2(out)
        if 'auglayer2' in mask_dict:
            _, _, H, W = out.shape
            mask_dict['auglayer2']  = F.interpolate(mask_dict['auglayer2'] , size=(H, W), mode='nearest')
            out = out * mask_dict['auglayer2']
        outs.append(out)

        out = self.layer3(out)
        if 'auglayer3' in mask_dict:
            _, _, H, W = out.shape
            mask_dict['auglayer3']  = F.interpolate(mask_dict['auglayer3'] , size=(H, W), mode='nearest')
            out = out * mask_dict['auglayer3']
        outs.append(out)

        out = F.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        # outs.append(out)
        return tuple(outs)
