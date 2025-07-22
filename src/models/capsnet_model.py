"""
CapsNet Model Definitions
Combined model with hyperparameter support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PrimaryCapsules(nn.Module):
    """Primary Capsules Layer with configurable parameters"""
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2, capsule_dim=8):
        super(PrimaryCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        
        # Create separate conv layer for each capsule
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0)
            for _ in range(num_capsules)
        ])
    
    def forward(self, x):
        # Apply each capsule convolution
        outputs = [capsule(x) for capsule in self.capsules]
        
        # Stack and reshape
        outputs = torch.stack(outputs, dim=1)  # [batch, num_capsules, out_channels, H, W]
        batch_size = outputs.size(0)
        
        # Reshape to [batch, num_capsules * H * W, capsule_dim]
        outputs = outputs.view(batch_size, self.num_capsules, -1)
        outputs = outputs.transpose(1, 2)  # [batch, H*W, num_capsules]
        
        # Apply squashing
        return self.squash(outputs)
    
    def squash(self, tensor):
        """Squashing function for capsules"""
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)


class DigitCapsules(nn.Module):
    """Digit Capsules Layer with dynamic routing"""
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16, num_iterations=3):
        super(DigitCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.num_iterations = num_iterations
        
        # Weight matrix for transformation
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Transform input capsules
        x = x.unsqueeze(2).unsqueeze(4)  # [batch, num_routes, 1, in_channels, 1]
        W = self.W.repeat(batch_size, 1, 1, 1, 1)  # [batch, num_routes, num_capsules, out_channels, in_channels]
        
        # Compute predictions
        u_hat = torch.matmul(W, x).squeeze(4)  # [batch, num_routes, num_capsules, out_channels]
        
        # Dynamic routing
        b_ij = torch.zeros(batch_size, self.num_routes, self.num_capsules, 1).to(x.device)
        
        for iteration in range(self.num_iterations):
            # Softmax over capsules
            c_ij = F.softmax(b_ij, dim=2)
            
            # Weighted sum
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)  # [batch, 1, num_capsules, out_channels]
            
            # Squashing
            v_j = self.squash(s_j)
            
            if iteration < self.num_iterations - 1:
                # Update routing coefficients
                a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)  # [batch, num_routes, num_capsules, 1]
                b_ij = b_ij + a_ij
        
        return v_j.squeeze(1)  # [batch, num_capsules, out_channels]
    
    def squash(self, tensor):
        """Squashing function"""
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)


class CapsNetFeatureExtractor(nn.Module):
    """
    CapsNet Feature Extractor with configurable hyperparameters
    """
    def __init__(self, 
                 input_channels=3, 
                 input_size=256,
                 feature_dim=128,
                 num_primary_capsules=8,
                 primary_capsule_dim=8,
                 num_digit_capsules=10,
                 digit_capsule_dim=16,
                 dropout_rate=0.2):
        super(CapsNetFeatureExtractor, self).__init__()
        
        self.input_size = input_size
        self.feature_dim = feature_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=9, stride=1, padding=0)
        self.conv1_bn = nn.BatchNorm2d(256)
        
        # Primary Capsules
        self.primary_capsules = PrimaryCapsules(
            num_capsules=num_primary_capsules,
            in_channels=256,
            out_channels=32,
            kernel_size=9,
            stride=2,
            capsule_dim=primary_capsule_dim
        )
        
        # Calculate the number of primary capsule outputs
        primary_output_size = ((input_size - 9 + 1) - 9) // 2 + 1
        num_primary_outputs = primary_output_size * primary_output_size * num_primary_capsules
        
        # Digit Capsules
        self.digit_capsules = DigitCapsules(
            num_capsules=num_digit_capsules,
            num_routes=num_primary_outputs,
            in_channels=primary_capsule_dim,
            out_channels=digit_capsule_dim,
            num_iterations=3
        )
        
        # Feature extraction head
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_digit_capsules * digit_capsule_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, feature_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Convolutional layer
        x = F.relu(self.conv1_bn(self.conv1(x)))
        
        # Primary capsules
        x = self.primary_capsules(x)
        
        # Digit capsules
        x = self.digit_capsules(x)
        
        # Flatten capsule outputs
        x = x.view(x.size(0), -1)
        
        # Extract features
        features = self.feature_extractor(x)
        
        return features


def create_capsnet_feature_extractor(input_channels=3, input_size=256, feature_dim=128, **kwargs):
    """
    Create CapsNet feature extractor with optional hyperparameters
    
    Args:
        input_channels: Number of input channels (default: 3)
        input_size: Input image size (default: 256)
        feature_dim: Output feature dimension (default: 128)
        **kwargs: Additional hyperparameters for advanced configuration
    
    Returns:
        CapsNetFeatureExtractor model
    """
    return CapsNetFeatureExtractor(
        input_channels=input_channels,
        input_size=input_size,
        feature_dim=feature_dim,
        **kwargs
    )
