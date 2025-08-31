import torch
from torch import nn


# -----------------------------------------------------------------
# This ResNetV1 model implemented according to the Paper:
# arXiv:1610.02915 - Deep Pyramidal Residual Networks
# -----------------------------------------------------------------


class PyramidNetBlock(nn.Module):
    def __init__(self, in_planes, planes, mid_conv_stride=1, downsample=None, residual=True):
        super().__init__()
        self.downsample = downsample
        self.residual = residual

        self.bn0 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=mid_conv_stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)

    def forward(self, x):
        identity = x

        x = self.bn0(x)

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
        feature_dim = identity.size()[2 : 4]
        
        batch_size = x.size(0)
        residual_channels = x.size(1)
        identity_channels = identity.size(1)

        if residual_channels != identity_channels:
            zero_pad = torch.zeros(batch_size, residual_channels - identity_channels, feature_dim[0], feature_dim[1], dtype=x.dtype, device=x.device)
            identity = torch.cat([identity, zero_pad], dim=1)
        
        if self.residual:
            x += identity
        return x



class PyramidNet(nn.Module):
    def __init__(self, num_classes, layer_counts, alpha):
        super().__init__()
        self.in_planes = 64
        self.add_rate = alpha / sum(layer_counts)


        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.feature_dim = self.in_planes
        self.layer1 = self._make_layer(layer_counts[0], stride=1)
        self.layer2 = self._make_layer(layer_counts[1], stride=2)
        self.layer3 = self._make_layer(layer_counts[2], stride=2)
        self.layer4 = self._make_layer(layer_counts[3], stride=2)
        self.final_dims = self.in_planes

        self.final_bn = nn.BatchNorm2d(self.final_dims)
        self.final_relu = nn.ReLU()
        self.final_avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.final_dims, num_classes)


    def _make_layer(self, num_blocks, stride):
        downsample = None
        layers = []

        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
            )

        self.feature_dim = self.feature_dim + self.add_rate
        layers.append(PyramidNetBlock(self.in_planes, int(round(self.feature_dim)),
                                        mid_conv_stride=stride, downsample=downsample))

        for _ in range(num_blocks - 1):
            temp_dim = self.feature_dim + self.add_rate
            layers.append(PyramidNetBlock(int(round(self.feature_dim)) * 4, int(round(temp_dim))))
            self.feature_dim = temp_dim
        self.in_planes = int(round(self.feature_dim)) * 4
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

        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.final_avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)