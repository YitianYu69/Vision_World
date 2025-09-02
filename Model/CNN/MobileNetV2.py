import torch
from torch import nn

def make_divisible(value, divisor, min_value=None):
    if min_value is None:
        min_value = divisor 

    new_value = max(min_value, int(value + divisor / 2)  // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value

def Conv_3x3(in_planes, planes, stride):
    return nn.Sequential(
        nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(planes),
        nn.ReLU6()
    )


def Conv_1x1(in_planes, planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(planes),
        nn.ReLU6()
    )

class InvertedResidual(nn.Module):
    def __init__(self, in_planes, planes, stride, expand_ratio):
        super().__init__()

        hidden_dim = int(in_planes * expand_ratio)
        self.residual = (stride == 1 and in_planes == planes)

        if in_planes == hidden_dim:
            self.conv = nn.Sequential(
                # dw conv
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.ReLU6(),
                # pw conv
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False),
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
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        if self.residual:
            return x + self.conv(x)
        return self.conv(x)

    
class MobileNetV2(nn.Module):
    def __init__(self, num_classes, width_mul=1.0):
        super().__init__()

        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        layers = []
        in_planes = make_divisible(32 * width_mul, 4 if width_mul == 0.1 else 8)
        layers += [Conv_3x3(3, in_planes, 2)]

        for t, c, n, s in self.cfgs:
            out_planes = make_divisible(c * width_mul, 4 if width_mul == 0.1 else 8)
            for i in range(n):
                stride = s if i == 0 else 1
                layers += [InvertedResidual(in_planes, out_planes, stride=stride, expand_ratio=t)]
                in_planes = out_planes
        
        final_out_planes = make_divisible(1280 * width_mul, 8) if width_mul > 1.0 else 1280
        layers += [Conv_1x1(in_planes, final_out_planes)]
        self.feature_forward = nn.Sequential(*layers)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(final_out_planes, num_classes)

    def forward(self, x):
        x = self.feature_forward(x)
        x = self.avg(x)
        x =  x.view(x.size(0), -1)
        return self.classifier(x)