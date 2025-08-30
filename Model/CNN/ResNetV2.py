import torch
from torch import nn

class ResV2Block(nn.Module):
    def __init__(self, in_planes, planes, mid_conv_stride, downsample=None, residual=True):
        super().__init__()

        self.residual = residual
        self.downsample = downsample

        self.bn0 = nn.BatchNorm2d(in_planes)
        self.relu0 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=mid_conv_stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        identity = x

        x = self.bn0(x)
        x = self.relu0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)

        if self.residual:
            if self.downsample is not None:
                identity = self.downsample
            x += identity
        return x
