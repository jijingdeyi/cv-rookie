import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_grad import Sobelxy
from pytorch_msssim import ssim, ms_ssim


class SSIM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, Y):
        return ssim(X, Y)


class MS_SSIM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, Y):
        return ms_ssim(X, Y)


class WeightedSSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM


if __name__ == "__main__":
    image_A = torch.randn(1, 1, 256, 256)
    image_B = image_A.clone()
    image_fused = image_A.clone()
    loss = WeightedSSIM()
    print(loss(image_A, image_B, image_fused))
