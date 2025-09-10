import torch
from torch import nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, 
                 in_channels,
                 patch_size,
                 image_size,
                 embed_dim):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        return x.transpose(1, 2).contiguous()


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_head,
                 attn_p):
        super().__init__()

        assert embed_dim % num_head == 0, "The embed_dim must be divisible by the number of head"
        self.num_head = num_head
        self.head_dim = embed_dim // num_head

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Lienar(embed_dim, embed_dim)
        self.attn_p = attn_p

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.size()

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, num_patches, 3, self.num_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, v, k = qkv[0], qkv[1], qkv[2]

        attn_out = F.scaled_dot_product_attention(q, k, v,
                                                  dropout_p=self.attn_p)
        out = attn_out.transpose(1, 2).contiguous().view(batch_size, num_patches, -1)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self,
                 embed_dim,
                 hidden_dim,
                 mlp_p):
        super().__init__()

        self.fc1 = nn.Lienar(embed_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(mlp_p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class DeiTBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_head,
                 mlp_ratio,
                 attn_p,
                 mlp_p):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mha = MultiHeadAttention(embed_dim, num_head, attn_p)

        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_dim, mlp_p)

    def forward(self, x):
        x = x + self.mha(self.layernorm1(x))
        return x + self.mlp(self.layernorm2(x))

class DeiT(nn.Module):
    def __init__(self,
                 in_channels,
                 patch_size,
                 image_size,
                 num_classes,
                 depth,
                 embed_dim,
                 num_head,
                 mlp_ratio,
                 attn_p,
                 mlp_p,
                 pos_p):
        super().__init__()

        self.patch_embed = PatchEmbed(in_channels, patch_size, image_size, embed_dim)

        self.cls_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_tokesns = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 2, embed_dim))
        self.pos_dropout = nn.Dropout(pos_p)

        self.blocks = nn.Module(
            [
              DeiTBlock(embed_dim, num_head, mlp_ratio, attn_p, mlp_p)
              for _ in range(depth)  
            ]
        )

        self.final_layernorm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        x = self.patch_embed(x)

        cls_tokens = self.cls_tokens.expand(x.size(0), -1, -1)
        dist_tokens = self.dist_tokens.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)

        x += self.pos_tokens
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_layernorm(x)

        cls_final, dist_final = x[:, 0, :], x[:, 1, :]
        cls_out = self.head(cls_final)
        dist_out = self.head_dist(dist_final)
        if self.training:
            return cls_out, dist_out
        else:
            return (cls_out + dist_out) / 2