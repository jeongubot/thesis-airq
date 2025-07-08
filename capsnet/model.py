import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=2)

    def forward(self, x):
        return F.relu(self.conv(x))

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=64, out_channels=8, kernel_size=9):
        super().__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2)
            for _ in range(num_capsules)
        ])

    def squash(self, x):
        mag_sq = torch.sum(x ** 2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq + 1e-9)
        return (mag_sq / (1 + mag_sq)) * (x / (mag + 1e-9))

    def forward(self, x):
        u = [caps(x) for caps in self.capsules]
        u = torch.stack(u, dim=1)
        B, C, _, H, W = u.shape
        u = u.permute(0, 1, 3, 4, 2).contiguous().view(B, -1, 8)
        return self.squash(u)

class DigitCaps(nn.Module):
    def __init__(self, in_dim, out_dim, num_capsules, num_routes):
        super().__init__()
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_dim, in_dim))

    def squash(self, x):
        mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        mag = torch.sqrt(mag_sq + 1e-9)
        return (mag_sq / (1. + mag_sq)) * (x / (mag + 1e-9))

    def forward(self, x):
        B = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)
        W = self.W.repeat(B, 1, 1, 1, 1)
        u_hat = torch.matmul(W, x).squeeze(-1)
        b_ij = torch.zeros(B, u_hat.size(1), u_hat.size(2)).to(x.device)

        for _ in range(3):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)
            v_j = self.squash(s_j)
            if _ < 2:
                b_ij = b_ij + (u_hat * v_j.unsqueeze(1)).sum(dim=-1)

        return v_j

class CapsNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(1, 64, kernel_size=5)
        self.primary_caps = PrimaryCaps(num_capsules=8, in_channels=64, out_channels=8)
        self.digit_caps = None  # initialized in forward()

    def forward(self, x):
        print("Input:", x.shape)
        x = self.conv1(x)
        print("After conv1:", x.shape)
        x = self.primary_caps(x)
        print("After primary_caps:", x.shape)  # [B, num_routes, in_dim]

        num_routes = x.size(1)
        in_dim = x.size(2)

        if self.digit_caps is None or self.digit_caps.W.size(1) != num_routes:
            self.digit_caps = DigitCaps(
                in_dim=in_dim,
                out_dim=16,
                num_capsules=10,
                num_routes=num_routes
            ).to(x.device)

        x = self.digit_caps(x)
        print("After digit_caps:", x.shape)
        return x.view(x.size(0), -1)  # [B, 10*16 = 160]

