import torch
import torch.nn as nn

class ChannelAttentionBlock(torch.nn.Module):
    def __init__(self):
        super(ChannelAttentionBlock, self).__init__()
        self.ca1 = nn.AdaptiveAvgPool2d(1)
        self.ca2 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x1, x2):
        EPSILON = 1e-10
        ca1 = self.ca1(x1)
        ca2 = self.ca2(x2)
        mask1 = torch.exp(ca1) / (torch.exp(ca2) + torch.exp(ca1) + EPSILON)
        mask2 = torch.exp(ca2) / (torch.exp(ca1) + torch.exp(ca2) + EPSILON)
        x1_a = mask1 * x1
        x2_a = mask2 * x2
        return x1_a, x2_a