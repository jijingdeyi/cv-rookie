import torch.nn as nn

class PointConv(nn.Module):
    """
    Point convolution block: input: x with size(B C H W); output size (B C1 H W)
    """
    def __init__(self, in_dim=64, out_dim=64,dilation=1, norm_layer=nn.BatchNorm2d):
        super(PointConv, self).__init__()
        self.kernel_size = 1
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dilation = dilation       
        conv_padding = (self.kernel_size // 2) * self.dilation        
                
        self.pconv = nn.Sequential(
                nn.Conv2d(self.in_dim, self.out_dim, self.kernel_size, padding=conv_padding, dilation = self.dilation),
                norm_layer(self.out_dim),
                nn.ReLU(inplace=True)
                )        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.pconv(x) 
        return x