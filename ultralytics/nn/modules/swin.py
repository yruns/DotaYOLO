# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Swin Transformer modules for YOLO models."""

import torch
import torch.nn as nn


class SwinMultiScale(nn.Module):
    """Swin Transformer backbone with multi-scale feature outputs.
    
    This module wraps torchvision's Swin Transformer and extracts features from
    multiple stages (P3, P4, P5) for use in FPN/PAN style detection heads.
    
    Attributes:
        model: The underlying Swin Transformer model.
        out_channels: List of output channel numbers for each scale.
    
    Args:
        variant (str): Swin variant to use. Options: 'swin_t', 'swin_s', 'swin_b'. Default: 'swin_t'.
        pretrained (bool): Whether to load pretrained weights. Default: True.
    
    Returns:
        list[torch.Tensor]: List of feature tensors [P3, P4, P5] with shapes:
            - P3: (B, 192, H/8, W/8) for swin_t
            - P4: (B, 384, H/16, W/16) for swin_t
            - P5: (B, 768, H/32, W/32) for swin_t
    """
    
    def __init__(self, variant: str = 'swin_t', pretrained: bool = True):
        """Initialize SwinMultiScale module.
        
        Args:
            variant: Swin variant ('swin_t', 'swin_s', 'swin_b').
            pretrained: Whether to load pretrained weights.
        """
        super().__init__()
        import torchvision
        
        # Get the appropriate Swin model
        weights = 'DEFAULT' if pretrained else None
        
        if variant == 'swin_t':
            if hasattr(torchvision.models, 'get_model'):
                self.model = torchvision.models.get_model('swin_t', weights=weights)
            else:
                from torchvision.models import swin_t, Swin_T_Weights
                self.model = swin_t(weights=Swin_T_Weights.DEFAULT if pretrained else None)
            self.out_channels = [192, 384, 768]  # C3, C4, C5 channels
        elif variant == 'swin_s':
            if hasattr(torchvision.models, 'get_model'):
                self.model = torchvision.models.get_model('swin_s', weights=weights)
            else:
                from torchvision.models import swin_s, Swin_S_Weights
                self.model = swin_s(weights=Swin_S_Weights.DEFAULT if pretrained else None)
            self.out_channels = [192, 384, 768]
        elif variant == 'swin_b':
            if hasattr(torchvision.models, 'get_model'):
                self.model = torchvision.models.get_model('swin_b', weights=weights)
            else:
                from torchvision.models import swin_b, Swin_B_Weights
                self.model = swin_b(weights=Swin_B_Weights.DEFAULT if pretrained else None)
            self.out_channels = [256, 512, 1024]
        else:
            raise ValueError(f"Unknown Swin variant: {variant}")
        
        # Remove the classification head
        self.model.head = nn.Identity()
        
        # Store references to feature extraction layers
        self.features = self.model.features
        self.norm = self.model.norm
        self.permute = self.model.permute
        
    def forward(self, x: torch.Tensor) -> list:
        """Forward pass extracting multi-scale features.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
        
        Returns:
            list: Feature tensors [P3, P4, P5] from stages 2, 3, 4.
        """
        outputs = []
        
        # Swin-T features structure:
        # features[0]: patch embedding (stride 4)
        # features[1]: stage 1 (keeps stride 4, but outputs at stride 4)
        # features[2]: patch merging (stride 8)
        # features[3]: stage 2 (stride 8) -> P3
        # features[4]: patch merging (stride 16)
        # features[5]: stage 3 (stride 16) -> P4
        # features[6]: patch merging (stride 32)
        # features[7]: stage 4 (stride 32) -> P5
        
        # Process through all stages and collect multi-scale outputs
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Collect outputs after each stage (after patch merging + transformer blocks)
            # Stage 2 output (P3): after features[3]
            # Stage 3 output (P4): after features[5]
            # Stage 4 output (P5): after features[7]
            if i in [3, 5, 7]:
                # x is (B, H, W, C), convert to (B, C, H, W)
                out = x.permute(0, 3, 1, 2).contiguous()
                outputs.append(out)
        
        return outputs


class SwinIndex(nn.Module):
    """Extract a specific index from SwinMultiScale output list.
    
    This module selects a specific feature map from the multi-scale output
    and optionally applies layer normalization.
    
    Attributes:
        index (int): Index of feature to extract.
        out_channels (int): Number of output channels (for info only).
    
    Args:
        out_channels (int): Expected number of output channels.
        index (int): Index to select from input list. Default: 0.
    """
    
    def __init__(self, out_channels: int, index: int = 0):
        """Initialize SwinIndex module.
        
        Args:
            out_channels: Expected number of output channels.
            index: Index to select from input list.
        """
        super().__init__()
        self.index = index
        self.out_channels = out_channels
        
    def forward(self, x: list) -> torch.Tensor:
        """Select feature at specified index.
        
        Args:
            x: List of feature tensors from SwinMultiScale.
        
        Returns:
            torch.Tensor: Feature tensor at the specified index.
        """
        return x[self.index]


__all__ = ['SwinMultiScale', 'SwinIndex']

