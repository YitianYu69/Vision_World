import torch
from torch import nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, 
                 in_channels: int = 3,
                 patch_size: int = 14,
                 image_size: int = 224,
                 embed_dim: int = 768):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        x = self.conv(x) # batch_size, embed_dim, sqrt(num_patches), sqrt(num_patches)
        x = x.flatten(2) # batch_size, embed_dim, num_patches
        return x.transpose(1, 2).contiguous()


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 num_head: int,
                 embed_dim: int,
                 attn_p: float = 0.2):
        super().__init__()

        assert embed_dim % num_head == 0, "The embed dim must be divisible by the num of head"
        self.num_head = num_head
        self.head_dim = embed_dim // num_head

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_p = attn_p

    def forward(self, x):
        batch_size, num_patch, embed_dim = x.size()

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, num_patch, 3, self.num_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = F.scaled_dot_product_attention(q, k, v,
                                                  dropout_p=self.attn_p if self.training else 0.0)
        out = attn_out.transpose(1, 2).contiguous().view(batch_size, num_patch, -1)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 mlp_p):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(mlp_p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, 
                 embed_dim,
                 num_head,
                 mlp_ratio,
                 attn_p,
                 mlp_p):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadAttention(num_head, embed_dim, attn_p)

        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_dim, mlp_p)

    def forward(self, x):
        x = x + self.attn(self.layernorm1(x))
        return x + self.mlp(self.layernorm2(x))


class VisionTransformer(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 image_size,
                 patch_size,
                 embed_dim,
                 depth,
                 num_head,
                 mlp_ratio,
                 attn_p,
                 mlp_p,
                 pos_p):
        super().__init__()

        self.patch_embed = PatchEmbed(in_channels, patch_size, image_size, embed_dim)

        self.cls_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_tokens = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(pos_p)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_head, mlp_ratio, attn_p, mlp_p)
                for _ in range(depth)
            ]
        )

        self.final_layernorm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)

        cls_tokens = self.cls_tokens.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_tokens
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)
        x = self.final_layernorm(x)
        cls_tokens_final = x[:, 0, :]
        return self.head(cls_tokens_final)


