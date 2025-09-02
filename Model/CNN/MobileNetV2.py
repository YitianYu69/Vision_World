import torch
from torch import nn

def make_divisor(value, divisor, min_value=None):
    new_value = max(min_vale, int(value + divisor / 2)  // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value

def Conv3x3(in_planes, planes, stride):
    return nn.Sequential(
        nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(planes),
        nn.ReLU6()
    )


def Conv1x1(in_planes, planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(planes),
        nn.ReLU6()
    )

class InvertedResidual(nn.Module):
    def __init__(self, in_planes, planes, stride, expand_ratio):
        super().__init__()

        hidden_dim = int(in_planes * expand_ratio)
        self.residual = stride == 1

        if in_planes == hidden_dim:
            self.conv = nn.Sequential(
                # dw conv
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU6(),
                # pw conv
                nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_planes, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),
                nn.Conv2d(hidden_dim, planes, kernel_size=1, stride=1, bias=False),
                nn.BatcjNorm2d(planes)
            )
    
    def forward(self, x):
        if self.residual:
            return x + self.conv(x)
        return x