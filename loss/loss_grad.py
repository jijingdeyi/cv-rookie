import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------ conventional gradient loss -----------


class Fusionloss(nn.Module):

    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        with torch.no_grad():
            x_grad_joint = torch.max(y_grad, ir_grad)
        generate_img_grad = self.sobelconv(generate_img)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        return loss_grad


class Sobelxy(nn.Module):
    # Sobelxy gradient loss with just norm
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


# ----------- grad loss from TC-MoA -----------


class SobelxyRGB(nn.Module):
    def __init__(self, isSignGrad=True):
        super(SobelxyRGB, self).__init__()
        self.isSignGrad = isSignGrad
        kernelx = [[-0.2, 0, 0.2], [-1, 0, 1], [-0.2, 0, 0.2]]
        kernely = [[0.2, 1, 0.2], [0, 0, 0], [-0.2, -1, -0.2]]
        self.kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        self.kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        C = x.shape[1]
        weightx = self.kernelx.to(x.device).repeat(C, 1, 1, 1)  # [C,1,3,3]
        weighty = self.kernely.to(x.device).repeat(C, 1, 1, 1)
        sobelx = F.conv2d(x, weightx, padding=1, groups=C)
        sobely = F.conv2d(x, weighty, padding=1, groups=C)
        if self.isSignGrad:
            return sobelx + sobely
        else:
            return torch.abs(sobelx) + torch.abs(sobely)


class MaxGradLoss(nn.Module):
    """Loss function for the grad loss.

    Args:
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, loss_weight=1.0, isSignGrad=True):
        super(MaxGradLoss, self).__init__()
        self.loss_weight = loss_weight
        self.sobelconv = SobelxyRGB(isSignGrad)
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir, *args, **kwargs):
        """Forward function.

        Args:
            im_fusion (Tensor): Fusion image with shape (N, C, H, W).
            im_rgb (Tensor): TIR image with shape (N, C, H, W).
        """
        if im_tir != None:
            rgb_grad = self.sobelconv(im_rgb)
            tir_grad = self.sobelconv(im_tir)

            mask = torch.ge(torch.abs(rgb_grad), torch.abs(tir_grad))
            max_grad_joint = tir_grad.masked_fill_(mask, 0) + rgb_grad.masked_fill_(
                ~mask, 0
            )

            generate_img_grad = self.sobelconv(im_fusion)

            sobel_loss = self.L1_loss(generate_img_grad, max_grad_joint)
            loss_grad = self.loss_weight * sobel_loss
        else:
            rgb_grad = self.sobelconv(im_rgb)
            generate_img_grad = self.sobelconv(im_fusion)
            sobel_loss = self.L1_loss(generate_img_grad, rgb_grad)
            loss_grad = self.loss_weight * sobel_loss

        return loss_grad


# ----------- Multi-scale gradient loss with direction alignment -----------


class Sobel2D(nn.Module):
    """Return Gx, Gy; using normal zero padding (padding=1)"""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)[
            None, None
        ]  # [1,1,3,3]
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)[
            None, None
        ]  # [1,1,3,3]
        self.kx = nn.Parameter(kx, requires_grad=False)
        self.ky = nn.Parameter(ky, requires_grad=False)

    def forward(self, x: torch.Tensor):
        C = x.shape[1]
        kernel_x = self.kx.repeat(C, 1, 1, 1)  # [C,1,3,3]
        kernel_y = self.ky.repeat(C, 1, 1, 1)  # [C,1,3,3]
        gx = F.conv2d(x, kernel_x, padding=1, groups=C)
        gy = F.conv2d(x, kernel_y, padding=1, groups=C)
        return gx, gy


class GradlossAligned(nn.Module):
    """
    Multi-scale direction-aligned gradient loss.
    - Scales: spatial downsampling ratios (e.g., 1.0, 0.5, 0.25)
    - Same weights for each scale
    - Sobel with normal zero padding
    """

    def __init__(self, scales=(1.0, 0.5, 0.25)):
        super().__init__()
        self.scales = tuple(scales)
        n = len(self.scales)
        assert n >= 1
        self.scale_weights = [1.0 / n] * n  # Same weights for each scale
        # self.scale_weights = [0.5, 0.309, 0.191]
        self.sobel2d = Sobel2D()

    def _resize(self, x, ratio: float):
        if ratio == 1.0:
            return x
        H, W = x.shape[-2:]
        Hs = max(1, int(round(H * ratio)))
        Ws = max(1, int(round(W * ratio)))
        return F.interpolate(x, size=(Hs, Ws), mode="bilinear", align_corners=False)

    def forward(
        self,
        image_vis: torch.Tensor,
        image_ir: torch.Tensor,
        generate_img: torch.Tensor,
    ):
        # luminance proxy: first channel of VIS
        image_y = image_vis[:, :1, :, :]

        loss = 0.0
        for ratio, w in zip(self.scales, self.scale_weights):
            gen_s = self._resize(generate_img, ratio)
            y_s = self._resize(image_y, ratio)
            ir_s = self._resize(image_ir, ratio)

            gx_f, gy_f = self.sobel2d(gen_s)
            gx_y, gy_y = self.sobel2d(y_s)
            gx_ir, gy_ir = self.sobel2d(ir_s)

            # axis-wise, sign-preserving hard selection
            sel_gx = torch.where(torch.abs(gx_y) >= torch.abs(gx_ir), gx_y, gx_ir)
            sel_gy = torch.where(torch.abs(gy_y) >= torch.abs(gy_ir), gy_y, gy_ir)

            loss_s = F.l1_loss(gx_f, sel_gx) + F.l1_loss(gy_f, sel_gy)
            loss = loss + w * loss_s

        return loss
