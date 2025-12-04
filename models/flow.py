import torch
import torch.nn.functional as F

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2) or (N, 2, H, W), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'
    Returns:
        Tensor: warped image or feature map
    """
    # Handle different flow formats: (N, 2, H, W) or (N, H, W, 2)
    if flow.dim() == 4 and flow.size(1) == 2:  # flow is (N, 2, H, W)
        # Check spatial dimensions before permute
        assert x.size()[-2:] == flow.size()[2:4], f"Spatial dimensions mismatch: x {x.size()[-2:]} vs flow {flow.size()[2:4]}"
        flow = flow.permute(0, 2, 3, 1)  # (N, 2, H, W) -> (N, H, W, 2)
    else:  # flow is (N, H, W, 2)
        # Check spatial dimensions
        assert x.size()[-2:] == flow.size()[1:3], f"Spatial dimensions mismatch: x {x.size()[-2:]} vs flow {flow.size()[1:3]}"
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output