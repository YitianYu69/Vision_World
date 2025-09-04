import torch
from torch import nn

# -----------------------------------------------------------------
# This MobileNetV3 model implemented according to the Paper:
# arXiv:1905.02244 - Searching for MobileNetV3
# -----------------------------------------------------------------

def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor

    new_value = max(min_value, int(v +  (divisor / 2)) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class h_sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self):
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        return x * self.sigmoid

class SE(nn.Module):
    def __init__(self in_planes, reducion=6):
        super().__init__()
        reduced_planes = make_divisible(in_planes // reducion, 8)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_planes, reduced_planes, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(reduced_planes, in_planes, kernel_size=1, stride=1, bias=False)
        self.h_sigmoid = h_sigmoid()

    def forward(self, x):
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x * self.h_sigmoid(x)


class Conv_3x3(nn.Module):
    def __init__(self, in_planes, planes, stride):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, bias=False),
            nn.BatchNorm2d(planes)
            h_swish()
        )

    def forward(self, x):
        return self.block(x)


class Conv_1x1(nn.Module):
    def __init__(self, in_planes, planes):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes),
            h_swish()
        )

    def forward(self, x):
        return self.block(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_planes, hidden_planes, planes, kernel_size, stride, use_se, use_hs):
        super().__init__()

        self.residual = (stride == 1 and in_planes == hidden_planes)

        if in_planes != hidden_planes:
            self.block = nn.Sequential(
                # dw
                nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size / 2), groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                h_swish() if use_hs else nn.ReLU(),
                SE(in_planes) if use_se else nn.Identity(),
                # pw
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.block = nn.Sequential(
                # pw
                nn.Conv2d(in_planes, hidden_planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_planes)
                h_swish() if use_hs else nn.ReLU(),
                # dw
                nn.Conv2d(hidden_planes, hidden_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size / 2), groups=hidden_planes, bias=False),
                nn.BatchNorm2d(hidden_planes),
                h_swish() if use_hs else nn.ReLU(),
                SE(hidden_planes) if use_se else nn.Identity(),
                # pw
                nn.Conv2d(hidden_planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        if self.residual:
            return x + self.block(x)
        return self.block(x)


class _MobileNetV3(nn.Module):
    def __init__(self, cfgs, num_classes, width_mul, mode):
        super().__init__()

        layers = []
        first_channels = make_divisible(32 * width_mul)
        layers += [Conv_3x3(3, first_channels, 2)]

        in_planes = 16
        for k, t, c, use_se, use_hs, s in cfgs:
            out_channels = make_divisible(c * width_mul, 8)
            hidden_channels = make_divisible(in_planes * t, 8)
            layers += [InvertedResidual(in_planes, hidden_channels, out_planes, kernel_size=k, stride=s, use_se=use_se, use_hs=use_hs)]
            in_planes = out_planes
        self.features = nn.Sequential(*layers)
        
        self.final_conv = Conv_1x1(in_planes, hidden_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        output_channels = {'large' : 1280, 'small' : 1024}
        output_channeles = make_divisible(output_channels[mode] * width_mul, 8) if width_mul > 1.0 else output_channels[mode]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, output_channels),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channels, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.final_conv(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return self.classifier(x)


def MobileNetV3_Large(num_classes, mode):

    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return _MobileNetV3(cfgs, num_classes, mode)

def MobileNetV3_Small(num_classes, mode):

    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]
    return _MobileNetV3(cfgs, num_classes, mode)