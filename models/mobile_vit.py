import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """
    轻量 MobileViT：局部 CNN + 小型 Transformer（token 化和 fold back）
    """

    def __init__(self, dim, heads=4, depth=2, patch=(2, 2), dropout=0.0):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )
        self.patch = patch
        self.transformer = nn.ModuleList(
            [AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(depth)]
        )
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x: torch.Tensor):
        local_feat = self.local(x)
        B, C, H, W = local_feat.shape
        ph, pw = self.patch
        new_h = math.ceil(H / ph) * ph
        new_w = math.ceil(W / pw) * pw
        if new_h != H or new_w != W:
            local_feat = F.interpolate(local_feat, size=(new_h, new_w), mode="bilinear", align_corners=False)
            H, W = new_h, new_w

        tokens = local_feat.unfold(2, ph, ph).unfold(3, pw, pw)  # B,C,nh,nw,ph,pw
        tokens = tokens.contiguous().view(B, C, -1, ph, pw)
        tokens = tokens.permute(0, 2, 3, 4, 1).reshape(B, -1, C)

        for blk in self.transformer:
            tokens = blk(tokens)

        feat = tokens.view(B, -1, ph * pw, C).permute(0, 3, 1, 2)
        nh = H // ph
        nw = W // pw
        feat = feat.view(B, C, nh, nw, ph, pw).permute(0, 1, 2, 4, 3, 5)
        feat = feat.reshape(B, C, H, W)

        if feat.shape[-2:] != x.shape[-2:]:
            feat = F.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out = self.fuse(torch.cat([x, feat], dim=1))
        return out