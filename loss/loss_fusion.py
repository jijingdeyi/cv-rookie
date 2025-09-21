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
    """返回 Gx, Gy；对单/多通道都兼容；支持 reflect padding 以减小边界伪影"""
    def __init__(self, reflect_pad: bool = True):
        super().__init__()
        self.reflect_pad = reflect_pad
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
        kx = self.kx.repeat(C, 1, 1, 1)  # [C,1,3,3]
        ky = self.ky.repeat(C, 1, 1, 1)  # [C,1,3,3]

        if self.reflect_pad:
            x = F.pad(x, (1, 1, 1, 1), mode="reflect")
            padding = 0
        else:
            padding = 1

        gx = F.conv2d(x, kx, padding=padding, groups=C)
        gy = F.conv2d(x, ky, padding=padding, groups=C)
        return gx, gy


class FusionlossAligned(nn.Module):
    """
    多尺度梯度项（第6点）+ Sobel 反射填充（第4点）
    其他保持你的写法：强度项用 max(Y, IR)，梯度项逐方向硬选择且保留符号
    """
    def __init__(self, grad_weight: float = 10.0,
                 scales=(1.0, 0.5, 0.25), scale_weights=None,
                 reflect_pad: bool = True):
        super().__init__()
        self.grad_weight = grad_weight
        self.scales = scales
        # 默认高分辨率更重要：1.0:1, 0.5:0.5, 0.25:0.25
        if scale_weights is None:
            self.scale_weights = [s for s in scales]
        else:
            self.scale_weights = scale_weights
        self.sobel2d = Sobel2D(reflect_pad=reflect_pad)

    def _resize(self, x, s: float):
        if s == 1.0:
            return x
        H, W = x.shape[-2:]
        Hs = max(1, int(round(H * s)))
        Ws = max(1, int(round(W * s)))
        return F.interpolate(x, size=(Hs, Ws), mode='bilinear', align_corners=False)

    def forward(self, image_vis: torch.Tensor, image_ir: torch.Tensor, generate_img: torch.Tensor):
        # === 强度项：保持你的写法 ===
        image_y = image_vis[:, :1, :, :]  # 若 image_vis 为 RGB，这里仍然取第一个通道（与你原版一致）
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)

        # === 多尺度梯度项 ===
        loss_grad_ms = 0.0
        for s, w in zip(self.scales, self.scale_weights):
            gen_s = self._resize(generate_img, s)
            y_s   = self._resize(image_y, s)
            ir_s  = self._resize(image_ir, s)

            gx_f, gy_f   = self.sobel2d(gen_s)
            gx_y, gy_y   = self.sobel2d(y_s)
            gx_ir, gy_ir = self.sobel2d(ir_s)

            # 逐方向硬选择（与你原版一致）
            sel_gx = torch.where(torch.abs(gx_y) >= torch.abs(gx_ir), gx_y, gx_ir)
            sel_gy = torch.where(torch.abs(gy_y) >= torch.abs(gy_ir), gy_y, gy_ir)

            loss_grad_s = F.l1_loss(gx_f, sel_gx) + F.l1_loss(gy_f, sel_gy)
            loss_grad_ms = loss_grad_ms + w * loss_grad_s

        loss_total = loss_in + self.grad_weight * loss_grad_ms
        return loss_total, loss_in, loss_grad_ms


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
