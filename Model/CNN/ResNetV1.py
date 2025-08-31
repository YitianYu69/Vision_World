import torch
from torch import nn


# -----------------------------------------------------------------
# This ResNetV1 model implemented according to the Paper:
# arXiv:1512.03385 - Deep Residual Learning for Image Recognition
# -----------------------------------------------------------------



class ResV1Block(nn.Module):
    def __init__(self, in_planes, planes, mid_conv_stride=1, downsample=None, residual=True):
        super().__init__()

        self.downsample = downsample
        self.residual = residual

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=mid_conv_stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.residual:
            if self.downsample is not None:
                identity = self.downsample(identity)
            x += identity
        return self.relu3(x)



class ResNetV1(nn.Module):
    def __init__(self, num_classes, layer_counts):
        super().__init__()

        self.in_planes = 64
        

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(layer_counts[0], planes=64, stride=1)
        self.layer2 = self._make_layer(layer_counts[1], planes=128, stride=2)
        self.layer3 = self._make_layer(layer_counts[2], planes=256, stride=2)
        self.layer4 = self._make_layer(layer_counts[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512*4, num_classes)

    def _make_layer(self, num_blocks, planes, stride):
        layers = []
        downsample = None

        if stride != 1 or self.in_planes != planes*4:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes*4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*4)
            )

        layers.append(
            ResV1Block(self.in_planes,
                    planes,
                    mid_conv_stride=stride,
                    downsample=downsample)
        )
        self.in_planes = planes * 4
        for _ in range(num_blocks - 1):
            layers.append(ResV1Block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
