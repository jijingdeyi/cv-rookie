import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusionloss(nn.Module):
    """
    Fusion loss = int_loss + 10*grad_loss
    """
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        # loss_grad=0.
        loss_total = loss_in + 10 * loss_grad
        return loss_total, loss_in, loss_grad


class Sobelxy(nn.Module):
    # 标量梯度强度图
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)
    

class Sobel2D(nn.Module):
    """返回 Gx, Gy；对单/多通道都兼容"""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32)[None, None]  # [1,1,3,3]
        ky = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32)[None, None]  # [1,1,3,3]
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x: torch.Tensor):
        C = x.shape[1]
        # 逐通道分组卷积，保持每个通道独立
        kx = self.kx.repeat(C, 1, 1, 1)  # [C,1,3,3]
        ky = self.ky.repeat(C, 1, 1, 1)  # [C,1,3,3]
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        return gx, gy

class FusionlossAligned(nn.Module):
    """
    我的改进版：
    Fusion loss = int_loss + grad_weight * grad_loss
    新的梯度项：分别在 x/y 方向按 |G| 选择更强的源梯度，并保留符号
    """
    def __init__(self, grad_weight: float = 10.0):
        super().__init__()
        self.grad_weight = grad_weight
        self.sobel2d = Sobel2D()

    def forward(self, image_vis: torch.Tensor, image_ir: torch.Tensor, generate_img: torch.Tensor):
        # === 强度项（保留你的写法） ===
        # 注：若 image_vis 是 RGB，这里只取 Y/亮度通道；若已是单通道则等价
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)

        # === 新的梯度项（方向+符号保持） ===
        gx_f, gy_f   = self.sobel2d(generate_img)
        gx_y, gy_y   = self.sobel2d(image_y)
        gx_ir, gy_ir = self.sobel2d(image_ir)

        # 按方向逐分量挑选：谁的 |G| 大，就跟谁，并保留该源的符号
        sel_gx = torch.where(torch.abs(gx_y) >= torch.abs(gx_ir), gx_y, gx_ir)
        sel_gy = torch.where(torch.abs(gy_y) >= torch.abs(gy_ir), gy_y, gy_ir)

        loss_grad = F.l1_loss(gx_f, sel_gx) + F.l1_loss(gy_f, sel_gy)

        loss_total = loss_in + self.grad_weight * loss_grad
        return loss_total, loss_in, loss_grad



def cc(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Correlation coefficient for images

    Shape: (B, C, H, W) 

    Type: torch.float32 
    
    Range: [0.,1.]
    """
    eps = torch.finfo(torch.float32).eps
    B, C, _, _ = img1.shape
    img1 = img1.reshape(B, C, -1)
    img2 = img2.reshape(B, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (
        eps
        + torch.sqrt(torch.sum(img1**2, dim=-1))
        * torch.sqrt(torch.sum(img2**2, dim=-1))
    )  # (B, C)
    cc = torch.clamp(cc, -1.0, 1.0)
    return cc.mean()
