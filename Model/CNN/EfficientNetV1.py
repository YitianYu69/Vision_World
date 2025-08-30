import torch
from torch import nn

import math

# --------------------------------------------------------------------------------------------
# This EfficientNetV1 model implemented according to the Paper:
# arXiv:1905.11946 - EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
# --------------------------------------------------------------------------------------------


params = {
    "efficientnet_b0": (1.0, 1.0, 224, 0.2),
    "efficientnet_b1": (1.0, 1.1, 240, 0.2),
    "efficientnet_b2": (1.1, 1.2, 260, 0.3),
    "efficientnet_b3": (1.2, 1.4, 300, 0.3),
    "efficientnet_b4": (1.4, 1.8, 380, 0.4),
    "efficientnet_b5": (1.6, 2.2, 456, 0.4),
    "efficientnet_b6": (1.8, 2.6, 528, 0.5),
    "efficientnet_b7": (2.0, 3.1, 600, 0.5),
}


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = self._same_padding(kernel_size, stride)
        super().__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_planes),
            Swish()
        )

    def _same_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]

class SE(nn.Module):
    def __init__(self, in_planes, reduce_ratio):
        super().__init__()
        mid = max(1, in_planes // reduce_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, mid, kernel_size=1),
            Swish(),
            nn.Conv2d(mid, in_planes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class DropPath(nn.Module):
    def __init__(self, drop_p):
        super().__init__()
        self.drop_p = float(drop_p)

    def forward(self, x):
        if self.drop_p == 0.0 or not self.training:
            return x
        keep_p = 1.0 - self.drop_p
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) + keep_p
        mask_tensor = torch.floor(random_tensor)
        return x.div(keep_p) * mask_tensor

class MBConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, expand_ratio, reduce_ratio=4, drop_path_p=0.2):
        super().__init__()
        self.residual = in_planes == out_planes and stride == 1
        hidden_dim = int(in_planes * expand_ratio)
        layers = []

        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, kernel_size=1)]
        
        layers += [
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            SE(hidden_dim, reduce_ratio),
            nn.Conv2d(hidden_dim, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes)
        ]

        self.drop_path = DropPath(drop_path_p)
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual:
            return x + self.drop_path(self.conv(x))
        else:
            return self.conv(x)


def _make_divisible(value, divisor=8):
    new_value = max(8, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    else:
        return int(_make_divisible(filters * width_mult))

def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    else:
        return int(math.ceil(repeats * depth_mult))
        
        
class _EfficientNetV1(nn.Module):
    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_p=0.2, num_classes=1000):
        super().__init__()

        # yapf: disable
        settings = [
            # t,  c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
        ]
        # yapf: enable

        first_channels = _round_filters(32, width_mult)
        layers = [ConvBNReLU(3, first_channels, kernel_size=3, stride=2)]
        in_planes = first_channels

        for t, c, n, s, k in settings:
            repeats = _round_repeats(n, depth_mult)
            out_planes = _round_filters(c, width_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                layers += [MBConvBlock(in_planes, out_planes, kernel_size=k, stride=stride, expand_ratio=t)]
                in_planes = out_planes
            
        final_channels = _round_filters(1280, width_mult)
        layers += [ConvBNReLU(in_planes, final_channels, kernel_size=1)]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(final_channels, num_classes)
        )

                # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.layers(x)
        x = x.mean([2, 3])
        return self.classifier(x)


def EfficientNetV1(arch, **kwargs):
    width_mult, depth_mult, _, dropout_p = params[arch]
    model = EfficientNetV1(width_mult, depth_mult, dropout_p, **kwargs)
    return model