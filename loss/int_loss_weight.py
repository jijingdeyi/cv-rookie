import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 灵感来源于GIFNet论文及其代码
# 利用预训练DenseNet121提取多层特征，计算梯度能量
# 以此作为红外和可见光图像的权重，计算加权MSE损失

class _GradKernel(nn.Module):
    """gradient(x): 使用与示例一致的 3x3 核，分组卷积逐通道计算"""
    def __init__(self):
        super().__init__()
        k = torch.tensor([[1/8, 1/8, 1/8],
                          [1/8, -1. , 1/8],
                          [1/8, 1/8, 1/8]], dtype=torch.float32)[None, None]  # [1,1,3,3]
        self.register_buffer("k", k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        k = self.k.repeat(c, 1, 1, 1)            # [C,1,3,3]
        x = F.pad(x, (1,1,1,1), mode="reflect")   # ReflectionPad2d(1)
        return F.conv2d(x, k, stride=1, padding=0, groups=c)


class GradGuidedIntLoss(nn.Module):
    """
    item1_IM_loss_cnn = w_ir * MSE(out_f, ir) + w_vi * MSE(out_f, vi)
    其中 w_* 由 DenseNet121 特征的梯度能量经 softmax 得到
    """
    def __init__(self):
        super().__init__()
        densenet = models.densenet121(pretrained=True)
        feats = list(densenet.features.children())
        # 与示例完全一致的切片
        self.f1 = nn.Sequential(*feats[:4])
        self.f2 = nn.Sequential(*feats[4:6])
        self.f3 = nn.Sequential(*feats[6:8])
        self.f4 = nn.Sequential(*feats[8:10])
        self.f5 = nn.Sequential(*feats[10:11])

        # 冻结并设为 eval
        for m in [self.f1, self.f2, self.f3, self.f4, self.f5]:
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()

        self.grad = _GradKernel()
        self.mse = nn.MSELoss(reduction="mean")

    @torch.no_grad()
    def _grad_energy_from_backbone(self, x3: torch.Tensor) -> torch.Tensor:
        """
        x3: [N,3,H,W]，返回每个 batch 的标量梯度能量平均（与示例一致为各层 mean 的平均）
        """
        l1 = self.f1(x3)
        l2 = self.f2(l1)
        l3 = self.f3(l2)
        l4 = self.f4(l3)
        l5 = self.f5(l4)

        def layer_energy(t):
            g = self.grad(t) ** 2     # 与示例：gradient(layer)**2
            return g.mean()           # 对全张量取 mean（H,W,C,N 一起）

        e = (layer_energy(l1) + layer_energy(l2) +
             layer_energy(l3) + layer_energy(l4) +
             layer_energy(l5)) / 5.0
        return e  # 标量（标量 tensor），可广播到后续 softmax

    def forward(self, out_f: torch.Tensor,
                batch_ir: torch.Tensor,
                batch_vi: torch.Tensor):
        """
        out_f, batch_ir, batch_vi: [N,1,H,W]（单通道）
        返回：loss_item1, w_ir, w_vi
        """
        assert out_f.dim() == 4 and out_f.size(1) == 1, "expect [N,1,H,W]"
        assert batch_ir.shape == out_f.shape and batch_vi.shape == out_f.shape, "shapes must match"

        device = out_f.device
        # 复制到 3 通道以喂 DenseNet
        dup_ir = batch_ir.repeat(1, 3, 1, 1).to(device)
        dup_vi = batch_vi.repeat(1, 3, 1, 1).to(device)

        # 计算两路的 CNN 梯度能量（不参与反传）
        grad_ir_cnn = self._grad_energy_from_backbone(dup_ir)
        grad_vi_cnn = self._grad_energy_from_backbone(dup_vi)

        # 稳定的 2 类 softmax 权重
        w = torch.softmax(torch.stack([grad_ir_cnn, grad_vi_cnn], dim=0), dim=0)  # [2]
        w_ir, w_vi = w[0], w[1]  # 标量 tensor

        # item1: 加权 MSE
        loss_item1 = w_ir * self.mse(out_f, batch_ir) + w_vi * self.mse(out_f, batch_vi)
        return loss_item1, w_ir.detach(), w_vi.detach()


class PixelWiseIntLoss(nn.Module):
    """
    我的改进版：
    逐像素权重版本：
      w_map = softmax([E_ir(x,y), E_vi(x,y)], dim=1)  ⟹  w_ir(x,y), w_vi(x,y)
      loss = mean_{N,H,W} [ w_ir * (out - ir)^2 + w_vi * (out - vi)^2 ]
    其中 E_* 为 DenseNet 各层特征经 gradient^2 后在通道维求均值、上采样到输入尺寸再平均。
    """
    def __init__(self):
        super().__init__()
        backbone = models.densenet121(pretrained=True)
        feats = list(backbone.features.children())
        # 与你片段保持一致的分段
        self.f1 = nn.Sequential(*feats[:4])
        self.f2 = nn.Sequential(*feats[4:6])
        self.f3 = nn.Sequential(*feats[6:8])
        self.f4 = nn.Sequential(*feats[8:10])
        self.f5 = nn.Sequential(*feats[10:11])

        for m in [self.f1, self.f2, self.f3, self.f4, self.f5]:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

        self.grad = _GradKernel()

    @torch.no_grad()
    def _energy_map(self, x3: torch.Tensor, out_size) -> torch.Tensor:
        """
        x3: [N,3,H,W]
        返回: E_map [N,1,H,W] —— 各层 grad^2 的通道均值图，上采样到 out_size 并平均
        """
        l1 = self.f1(x3)
        l2 = self.f2(l1)
        l3 = self.f3(l2)
        l4 = self.f4(l3)
        l5 = self.f5(l4)

        def e_map(layer):
            g2 = self.grad(layer)**2                 # [N,C,h,w]
            e  = g2.mean(dim=1, keepdim=True)        # 通道均值 -> [N,1,h,w]
            e  = F.interpolate(e, size=out_size, mode="bilinear", align_corners=False)
            return e

        E = (e_map(l1) + e_map(l2) + e_map(l3) + e_map(l4) + e_map(l5)) / 5.0  # [N,1,H,W]
        return E

    def forward(self, out_f: torch.Tensor, ir: torch.Tensor, vi: torch.Tensor):
        """
        out_f, ir, vi: [N,1,H,W] 单通道张量
        返回: loss, w_ir_map, w_vi_map  （权重图已 detach 便于观测）
        """
        assert out_f.shape == ir.shape == vi.shape and out_f.size(1) == 1, "expect [N,1,H,W]"

        N, _, H, W = out_f.shape
        device = out_f.device

        # 复制到 3 通道以喂 DenseNet
        ir3 = ir.repeat(1, 3, 1, 1).to(device)
        vi3 = vi.repeat(1, 3, 1, 1).to(device)

        # 逐像素能量图（不进主干计算图）
        E_ir = self._energy_map(ir3, out_size=(H, W))   # [N,1,H,W]
        E_vi = self._energy_map(vi3, out_size=(H, W))   # [N,1,H,W]

        # 在 dim=1（2 类）上做 softmax，得到逐像素权重
        stacked = torch.cat([E_ir, E_vi], dim=1)        # [N,2,H,W]
        weights = torch.softmax(stacked, dim=1)         # [N,2,H,W]
        w_ir = weights[:, :1]                           # [N,1,H,W]
        w_vi = weights[:, 1:]                           # [N,1,H,W]

        # 逐像素加权 MSE（不用 F.mse_loss，自己做 elementwise 再平均）
        se_ir = (out_f - ir)**2                         # [N,1,H,W]
        se_vi = (out_f - vi)**2
        loss_map = w_ir * se_ir + w_vi * se_vi          # [N,1,H,W]
        loss = loss_map.mean()                          # mean over N, C, H, W

        return loss, w_ir.detach(), w_vi.detach()