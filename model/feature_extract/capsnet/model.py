# model/capsnet/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    def __init__(self,
                 num_capsules,
                 in_channels,
                 out_channels,
                 kernel_size=None,
                 stride=None,
                 num_routes=0,
                 routing_iters=3):
        super().__init__()
        self.num_capsules = num_capsules
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_routes = num_routes
        self.routing_iters = routing_iters

        if num_routes == 0:
            # PrimaryCaps layer
            self.capsules = nn.Conv2d(
                in_channels,
                num_capsules * out_channels,
                kernel_size=kernel_size,
                stride=stride
            )
        else:
            # DigitCaps routing layer
            self.route_weights = nn.Parameter(
                torch.randn(num_capsules, num_routes, out_channels, in_channels)
            )

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1.0 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)

    def forward(self, x):
        if self.num_routes == 0:
            # PrimaryCaps forward
            out = self.capsules(x)
            b, c, h, w = out.size()
            out = out.view(b, self.num_capsules, self.out_channels, h, w)
            out = out.permute(0, 1, 3, 4, 2).contiguous()
            out = out.view(b, -1, self.out_channels)
            return self.squash(out)
        else:
            # DigitCaps forward with routing-by-agreement
            batch_size = x.size(0)
            x = x.unsqueeze(1).unsqueeze(4)  # [B,1,num_routes,in_channels,1]
            u_hat = torch.matmul(self.route_weights, x)  # [B,num_caps,num_routes,out_channels,1]

            b_ij = torch.zeros(batch_size, self.num_capsules, self.num_routes, 1, 1, device=x.device)

            for i in range(self.routing_iters):
                c_ij = F.softmax(b_ij, dim=1)  # Softmax across capsules
                s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)  # weighted sum
                v_j = self.squash(s_j, dim=-1)  # squash activation

                if i < self.routing_iters - 1:
                    delta = (u_hat * v_j).sum(dim=-1, keepdim=True)
                    b_ij = b_ij + delta

            return v_j.squeeze(2).squeeze(-1)  # [B, num_capsules, out_channels]

class CapsNet(nn.Module):
    """
    Optimized 2-layer Capsule Network:
    Conv1 → PrimaryCaps → DigitCaps
    """
    def __init__(self, num_classes=10, img_size=128, primary_channels=16, routing_iters=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)

        self.primary_caps = CapsuleLayer(
            num_capsules=primary_channels,     # reduced from 32 to 16
            in_channels=256,
            out_channels=8,
            kernel_size=9,
            stride=2,
            num_routes=0  # indicates PrimaryCaps
        )

        # Dynamically compute num_routes (number of primary capsules)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            x = F.relu(self.conv1(dummy))
            x = self.primary_caps(x)  # shape: [1, num_routes, 8]
            n_primary = x.size(1)     # total primary capsules

        self.digit_caps = CapsuleLayer(
            num_capsules=num_classes,
            in_channels=8,
            out_channels=16,
            num_routes=n_primary,
            routing_iters=routing_iters
        )

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        return x  # [B, num_classes, 16]
