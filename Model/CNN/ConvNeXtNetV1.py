import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


# -----------------------------------------------------------------
# This ConvNeXtV1 model implemented according to the Paper:
# arXiv:2201.03545 - A ConvNet for the 2020s
# -----------------------------------------------------------------


class DropPath(nn.Module):
    def __init__(self, drop_p):
        super().__init__()
        self.drop_p = drop_p

    def forward(self, x):
        if self.drop_p == 0.0 or not self.training:
            return x

        keep_p = 1 - self.drop_p
        shape = (x.size(0),) + (1,) * (x.dim() - 1)
        random_tensor = keep_p + torch.rand(shape, device=x.device, dtype=x.dtype)
        mask_tensor = torch.floor(random_tensor)
        return x.div(keep_p) * mask_tensor

    
class LayerNorm(nn.Module):
    def __init__(self, normalize_shape, data_format='channel_last', eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalize_shape))
        self.bias = nn.Parameter(torch.zeros(normalize_shape))

        if data_format not in ['channel_first', 'channel_last']:
            raise ValueError("data_format has to be either channel_first or channel_last")
        self.data_format = data_format
        self.normalize_shape = normalize_shape
        self.eps = eps

    def forward(self, x):
        if self.data_format == 'channel_last':
            return F.layer_norm(x, self.normalize_shape, self.weight, self.bias, self.eps)
        
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_p, layer_scale_value):
        super().__init__()

        self.dw = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.ln = LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim*4)
        self.gelu = nn.GELU()
        self.pw2 = nn.Linear(dim*4, dim)

        self.gamma = nn.Parameter(layer_scale_value * torch.ones(dim)) if layer_scale_value > 0.0 else None
        self.drop_path = DropPath(drop_p) if drop_p > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x

        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = self.pw1(x)
        x = self.gelu(x)
        x = self.pw2(x)
        x = x.permute(0, 3, 1, 2)

        if self.gamma is not None:
            x = self.gamma * x
        return identity + self.drop_path(x)


class ConvNeXt(nn.Module):
    def __init__(self, num_classes, in_channels=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0.0, layer_scale_init_value=1e-6, head_init_scale=1):
        super().__init__()

        self.downsample = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4, bias=False),
            LayerNorm(dims[0], data_format='channel_first')
        )
        self.downsample.append(stem)

        for i in range(3):
            self.downsample.append(nn.Sequential(
                LayerNorm(dims[i], data_format='channel_first'),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2, bias=False)
            ))
        
        self.stages = nn.ModuleList()
        drop_p = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[ConvNeXtBlock(dims[i], drop_p=drop_p[cur + j], layer_scale_value=layer_scale_init_value) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]
        
        self.final_norm = LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._initialize_weight)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    
    def _initialize_weight(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample[i](x)
            x = self.stages[i](x)
        return self.final_norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)



def convnext_tiny(num_classes=1000, **kwargs):
    return ConvNeXt(num_classes, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

def convnext_small(num_classes=1000, **kwargs):
    return ConvNeXt(num_classes, depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)

def convnext_base(num_classes=1000, **kwargs):
    return ConvNeXt(num_classes, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)

def convnext_large(num_classes=1000, **kwargs):
    return ConvNeXt(num_classes, depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
