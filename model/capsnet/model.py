import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# 1) CAPSNET FEATURE EXTRACTOR
# -----------------------------------------------------------------------------

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)

class PrimaryCaps(nn.Module):
    def __init__(self, in_channels=256, num_capsules=32, cap_dim=8):
        super().__init__()
        self.num_capsules, self.cap_dim = num_capsules, cap_dim
        self.capsules = nn.Conv2d(in_channels,
                                  num_capsules * cap_dim,
                                  kernel_size=9,
                                  stride=2)

    def squash(self, u):
        # u: [B, num_capsules, H, W, cap_dim]
        mag_sq = (u**2).sum(dim=-1, keepdim=True)
        mag    = torch.sqrt(mag_sq + 1e-8)
        return (mag_sq/(1+mag_sq)) * (u/(mag+1e-8))

    def forward(self, x):
        B = x.size(0)
        u = self.capsules(x)                           # [B, 32*8, 6,6]
        u = u.view(B, self.num_capsules, self.cap_dim, 6, 6)
        u = u.permute(0,1,3,4,2).contiguous()          # [B,32,6,6,8]
        u = u.view(B, -1, self.cap_dim)                # [B,1152,8]
        return self.squash(u)

class DigitCaps(nn.Module):
    def __init__(self, in_dim=8, out_dim=16, num_capsules=10, num_routes=32*6*6):
        super().__init__()
        self.num_routes  = num_routes
        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.num_caps    = num_capsules
        self.W = nn.Parameter(torch.randn(1,
                                          num_routes,
                                          num_capsules,
                                          out_dim,
                                          in_dim))

    def squash(self, s):
        # s: [B, num_capsules, out_dim]
        mag_sq = (s**2).sum(dim=-1, keepdim=True)
        mag    = torch.sqrt(mag_sq + 1e-8)
        return (mag_sq/(1+mag_sq))*(s/(mag+1e-8))

    def forward(self, x):
        # x: [B, num_routes, in_dim]
        B = x.size(0)
        x = x.unsqueeze(2).unsqueeze(-1)               # [B, R, 1, D_in, 1]
        W = self.W.repeat(B,1,1,1,1)                   # [B, R, C, D_out, D_in]
        u_hat = torch.matmul(W, x).squeeze(-1)         # [B, R, C, D_out]

        b_ij = torch.zeros(B, self.num_routes, self.num_caps, device=x.device)
        for _ in range(3):  # routing iterations
            c_ij = F.softmax(b_ij, dim=2)              # [B, R, C]
            s_j  = (c_ij.unsqueeze(-1)*u_hat).sum(dim=1)  # [B, C, D_out]
            v_j  = self.squash(s_j)                    # [B, C, D_out]
            if _ < 2:
                # agreement
                b_ij = b_ij + (u_hat * v_j.unsqueeze(1)).sum(dim=-1)
        return v_j  # [B, C, D_out]

class CapsNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # matches Sabour et al. “Conv1”
        self.conv1 = ConvLayer(1, 256, kernel_size=9, stride=1, padding=0)
        self.primary_caps = PrimaryCaps(in_channels=256,
                                        num_capsules=32,
                                        cap_dim=8)
        self.digit_caps = DigitCaps(in_dim=8,
                                    out_dim=16,
                                    num_capsules=10,
                                    num_routes=32*6*6)

    def forward(self, x):
        # x: [B,1,28,28]
        x = self.conv1(x)              # [B,256,20,20]
        x = self.primary_caps(x)       # [B,1152,8]
        x = self.digit_caps(x)         # [B,10,16]
        # flatten all 10 capsules × 16 dims = 160
        return x.view(x.size(0), -1)   # [B,160]

