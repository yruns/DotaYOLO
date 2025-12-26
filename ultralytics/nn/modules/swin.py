# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Swin Transformer modules for YOLO models.

This module implements Swin Transformer components for integration with YOLO architectures,
following the approach from https://github.com/YJHCUI/SwinTransformer-YOLOv5/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Tuple
import numpy as np


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows.
    
    Args:
        x: Input tensor of shape (B, H, W, C).
        window_size: Window size.
    
    Returns:
        windows: Tensor of shape (num_windows*B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition.
    
    Args:
        windows: Tensor of shape (num_windows*B, window_size, window_size, C).
        window_size: Window size.
        H: Height of the feature map.
        W: Width of the feature map.
    
    Returns:
        x: Tensor of shape (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    
    Args:
        dim: Number of input channels.
        window_size: The height and width of the window.
        num_heads: Number of attention heads.
        qkv_bias: If True, add a learnable bias to query, key, value.
        attn_drop: Dropout ratio of attention weight.
        proj_drop: Dropout ratio of output.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward pass.
        
        Args:
            x: Input features of shape (num_windows*B, N, C).
            mask: Attention mask of shape (num_windows, Wh*Ww, Wh*Ww) or None.
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # 确保relative_position_bias的类型与attn一致（解决AMP混合精度训练问题）
        attn = attn + relative_position_bias.unsqueeze(0).to(dtype=attn.dtype)

        if mask is not None:
            nW = mask.shape[0]
            # 确保mask的类型与attn一致（解决AMP混合精度训练问题）
            mask = mask.to(dtype=attn.dtype)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.
    
    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size.
        shift_size: Shift size for SW-MSA.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: If True, add a learnable bias to query, key, value.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate.
        norm_layer: Normalization layer.
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.
    
    Splits the input image into non-overlapping patches and projects them to embedding dimension.
    
    Args:
        c1: Number of input channels.
        c2: Embedding dimension (output channels).
        patch_size: Patch size. Default: 4.
    
    Example YAML:
        [-1, 1, PatchEmbed, [96, 4]]  # 0 [b, 96, 160, 160] for 640x640 input
    """

    def __init__(self, c1, c2, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.c1 = c1
        self.c2 = c2
        
        self.proj = nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
        
        Returns:
            Tensor of shape (B, C, H/patch_size, W/patch_size) in BCHW format.
        """
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.proj(x)  # (B, c2, H/patch_size, W/patch_size)
        
        # Flatten and normalize, then reshape back
        _, _, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, Hp*Wp, c2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, self.c2, Hp, Wp)  # (B, c2, Hp, Wp)
        
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer.
    
    Merges adjacent patches to reduce spatial resolution by 2x and increase channels.
    
    Args:
        c1: Number of input channels.
        c2: Number of output channels.
        norm_layer: Normalization layer.
    
    Example YAML:
        [-1, 1, PatchMerging, [192]]  # doubles spatial downsample, increases channels
    """

    def __init__(self, c1, c2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.reduction = nn.Linear(4 * c1, c2, bias=False)
        self.norm = norm_layer(4 * c1)

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W) in BCHW format.
        
        Returns:
            Tensor of shape (B, c2, H/2, W/2) in BCHW format.
        """
        B, C, H, W = x.shape
        
        # Convert to (B, H, W, C) for processing
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # Pad if necessary
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        _, H_pad, W_pad, _ = x.shape
        
        # Merge patches
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        
        x = self.norm(x)
        x = self.reduction(x)  # B H/2 W/2 c2
        
        # Convert back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        return x


class SwinStage(nn.Module):
    """A Swin Transformer Stage consisting of multiple Swin Transformer Blocks.
    
    Implements one stage of the Swin Transformer with configurable depth and attention heads.
    
    Args:
        c1: Number of input channels.
        c2: Number of output channels (same as c1 for stage, channels don't change within stage).
        depth: Number of Swin Transformer Blocks.
        num_heads: Number of attention heads.
        window_size: Local window size.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: If True, add a learnable bias to query, key, value.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate.
        norm_layer: Normalization layer.
        use_checkpoint: Whether to use checkpointing to save memory.
    
    Example YAML:
        [-1, 1, SwinStage, [96, 2, 3, 7]]  # c2=96, depth=2, num_heads=3, window_size=7
    """

    def __init__(self, c1, c2, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        self.shift_size = window_size // 2

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=c1,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

    def create_mask(self, x, H, W):
        """Create attention mask for SW-MSA."""
        # Calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W) in BCHW format.
        
        Returns:
            Tensor of shape (B, c2, H, W) in BCHW format.
        """
        B, C, H, W = x.shape
        
        # Convert to (B, H*W, C) for transformer blocks
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Create attention mask for shifted window
        attn_mask = self.create_mask(x, H, W)
        
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        
        # Convert back to (B, C, H, W)
        x = x.transpose(1, 2).view(B, self.c2, H, W)
        
        return x


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


__all__ = ['SwinMultiScale', 'SwinIndex', 'PatchEmbed', 'PatchMerging', 'SwinStage']

