"""Slot attention module and 3D positional embeddings for video spatiotemporal representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_3d_grid(resolution):
    """Construct a normalized 3D coordinate grid with complement coordinates.

    Args:
        resolution (tuple[int, int, int]): (T, H, W) grid resolution.

    Returns:
        torch.Tensor: Normalized grid coordinates of shape (1, T, H, W, 6).
    """
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing='ij' if hasattr(torch, 'meshgrid') else None)
    # Shape after stack: (T, H, W, 3)
    grid = torch.stack(grid, dim=-1)
    # Shape: (1, T, H, W, 3)
    grid = torch.reshape(grid, [resolution[0], resolution[1], resolution[2], -1]).unsqueeze(0)
    # Concatenate coordinates with (1 - coordinates): shape (1, T, H, W, 6)
    return torch.cat([grid, 1.0 - grid], dim=-1)


class SoftPositionEmbed3D(nn.Module):
    """Learnable soft position embedding for 3D spatiotemporal feature volumes."""

    def __init__(self, hidden_size, resolution):
        """
        Args:
            hidden_size (int): Dimension of feature vectors (dim).
            resolution (tuple[int, int, int]): Spatiotemporal dimensions (T, H, W).
        """
        super().__init__()
        self.embedding = nn.Linear(6, hidden_size, bias=True)
        # Register fixed grid buffer of shape (1, T, H, W, 6)
        self.register_buffer("grid", build_3d_grid(resolution))

    def forward(self, inputs):
        """Add 3D positional embeddings to feature volume.

        Args:
            inputs (torch.Tensor): Feature volume of shape (B, T, H, W, dim).

        Returns:
            torch.Tensor: Positionally-augmented features of shape (B, T, H, W, dim).
        """
        B, T, H, W, dim = inputs.shape
        # Dynamically build grid if input spatial/temporal resolution differs from pre-registered buffer
        if (
            self.grid is None
            or self.grid.shape[1] != T
            or self.grid.shape[2] != H
            or self.grid.shape[3] != W
        ):
            grid = build_3d_grid((T, H, W)).to(device=inputs.device, dtype=inputs.dtype)
        else:
            grid = self.grid.to(device=inputs.device, dtype=inputs.dtype)

        grid = self.embedding(grid)
        return inputs + grid


class SlotAttention(nn.Module):
    """Spatiotemporal 3D Slot Attention mechanism.

    Iteratively extracts object-centric or action-centric slot representations
    from 3D spatiotemporal video feature volumes.
    """

    def __init__(
        self,
        num_slots,
        dim,
        num_actor_class=64,
        eps=1e-8,
        input_dim=64,
        resolution=[16, 8, 24],
        allocated_slot=True,
    ):
        """
        Args:
            num_slots (int): Total number of slots.
            dim (int): Slot and feature dimension.
            num_actor_class (int): Number of target actor action classes.
            eps (float): Epsilon for numerical stability in attention normalization.
            input_dim (int): Dimension of input features before linear projection.
            resolution (list[int]): (T, H, W) resolution for 3D positional encoding.
            allocated_slot (bool): If True, slots are directly assigned to action classes.
        """
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_actor_class = num_actor_class
        self.allocated_slot = allocated_slot
        self.eps = eps
        self.scale = dim**-0.5
        self.resolution = resolution

        # Learnable Gaussian parameters for slot initialization: shape (1, num_slots, dim)
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, num_slots, dim).abs())

        self.FC1 = nn.Linear(dim, dim)
        self.FC2 = nn.Linear(dim, dim)
        self.LN = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        # Initialize slots buffer: shape (1, num_slots, dim)
        with torch.no_grad():
            slots = torch.normal(self.slots_mu, self.slots_sigma)
        self.register_buffer("slots", slots.contiguous())

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pe = SoftPositionEmbed3D(
            dim, [resolution[0], resolution[1], resolution[2]]
        )

    def get_3d_slot(self, slots, inputs):
        """Perform slot attention aggregation over spatiotemporal input features.

        Args:
            slots (torch.Tensor): Initial slot embeddings of shape (1, num_slots, dim).
            inputs (torch.Tensor): Feature volume of shape (B, T, H, W, dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - slots: Updated slot features of shape (B, num_slots, dim).
                - attn_ori: Unnormalized attention map of shape (B, num_slots, T*H*W).
        """
        b, l, h, w, d = inputs.shape  # B, T, H, W, dim

        # Add 3D positional embeddings: (B, T, H, W, dim)
        inputs = self.pe(inputs)
        # Flatten spatial-temporal dimensions: (B, T*H*W, dim)
        inputs = torch.reshape(inputs, (b, -1, d))

        # MLP pre-processing
        inputs = self.LN(inputs)
        inputs = self.FC1(inputs)
        inputs = F.relu(inputs)
        inputs = self.FC2(inputs)

        # Expand slots across batch: (B, num_slots, dim)
        if slots.shape[0] != b:
            slots = slots.expand(b, -1, -1)

        inputs = self.norm_input(inputs)   # (B, T*H*W, dim)
        k = self.to_k(inputs)              # (B, T*H*W, dim)
        v = self.to_v(inputs)              # (B, T*H*W, dim)
        slots = self.norm_slots(slots)     # (B, num_slots, dim)
        q = self.to_q(slots)               # (B, num_slots, dim)

        # Cache latest activation stats for model diagnostics/tracking
        self.last_q = q.detach()
        self.last_k = k.detach()
        self.last_v = v.detach()

        # Scaled dot-product attention over slots: dots shape (B, num_slots, T*H*W)
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn_ori = dots.softmax(dim=1) + self.eps  # Softmax over slot dimension
        attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)  # Normalize across keys
        self.last_attn = attn_ori.detach()

        # Slot feature update: (B, num_slots, dim)
        slots = torch.einsum('bjd,bij->bid', v, attn)
        slots = slots.reshape(b, -1, d)

        # Slice relevant slot indices
        if self.allocated_slot:
            slots = slots[:, :self.num_actor_class, :]  # (B, num_actor_class, dim)
        else:
            slots = slots[:, :self.num_slots, :]        # (B, num_slots, dim)

        # Feed-forward residual connection: (B, num_slots, dim)
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots, attn_ori

    def forward(self, inputs, num_slots=None):
        """Forward pass for SlotAttention.

        Args:
            inputs (torch.Tensor): Input video feature volume of shape (B, T, H, W, dim).
            num_slots (int, optional): Ignored placeholder for compatibility.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - slots_out: Refined slot vectors of shape (B, S, dim).
                - attns: Raw attention weights of shape (B, S, T*H*W).
        """
        slots_out, attns = self.get_3d_slot(self.slots, inputs)
        return slots_out, attns
