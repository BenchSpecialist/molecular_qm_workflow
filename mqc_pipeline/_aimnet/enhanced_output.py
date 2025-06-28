from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .core import MLP


class EnhancedElectronicLevelsOutput(nn.Module):

    def __init__(self,
                 mlp: Union[Dict, nn.Module],
                 n_in: int,
                 key_in: str = 'aim_enhanced_electronic'):
        super().__init__()
        self.key_in = key_in
        #self.joint_key = joint_key

        # Add normalization buffers
        #Aimnet2
        self.register_buffer(
            'orbital_mean',
            torch.tensor([-6.08881950378418,
                          -2.019904375076294]))  # [homo_mean, lumo_mean]
        self.register_buffer('orbital_std',
                             torch.tensor(
                                 [1.0473166704177856,
                                  1.2768841981887817]))  # [homo_std, lumo_std]
        # Mixed 29M
        # self.register_buffer('orbital_mean', torch.tensor([-6.067822456359863, -0.9488564729690552]))  # [homo_mean, lumo_mean]
        # self.register_buffer('orbital_std', torch.tensor([0.7097107172012329, 1.1951872110366821]))    # [homo_std, lumo_std]

        ### only internal data statistics
        # self.register_buffer('orbital_mean', torch.tensor([-6.108067512512207, -0.5065680146217346]))  # [homo_mean, lumo_mean]
        # self.register_buffer('orbital_std', torch.tensor([0.5163620710372925, 0.8657941222190857]))    # [homo_std, lumo_std]

        # # Enhanced feature transformation pipeline
        self.feature_transformer = EnhancedFeatureTransformationPipeline(
            aim_dim=n_in, num_heads=8, dropout=0.1)

        # Main MLP
        if not isinstance(mlp, nn.Module):
            mlp = MLP(
                n_in=n_in,
                n_out=2,
                **
                mlp  # This will unpack hidden, activation_fn, dropout, etc. from yaml
            )
        self.mlp = mlp

        # Orbital-specific branches
        self.homo_branch = OrbitralBranch(n_in)
        self.lumo_branch = OrbitralBranch(n_in)

        # Atomic importance weighting
        self.atom_weights = nn.Sequential(nn.Linear(n_in, n_in // 2),
                                          nn.SiLU(), nn.Linear(n_in // 2, 1))

        # fusion layer to combine the two inputs
        # self.fusion = nn.Sequential(
        #     nn.Linear(n_in * 2, n_in),
        #     nn.SiLU(),
        #     nn.LayerNorm(n_in)
        # )

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        features = data[self.key_in].contiguous()
        # Check if joint representation is available
        # if self.joint_key in data:
        #     joint_vector = data[self.joint_key].contiguous()
        #     # Concatenate and fuse the representations
        #     combined_vector = torch.cat([features, joint_vector], dim=-1)
        #     features = self.fusion(combined_vector)
        # Create mask for padding if needed
        mask = None
        if data['_input_padded'].item():
            mask = data['mask_i']

        # Apply enhanced feature transformation
        enhanced_features = self.feature_transformer(features, mask)

        # Base predictions using the enhanced features
        base_orbitals = self.mlp(enhanced_features)
        homo_base, lumo_base = base_orbitals.chunk(2, dim=-1)

        # Specialized predictions
        homo_specific = self.homo_branch(enhanced_features)
        lumo_specific = self.lumo_branch(enhanced_features)

        homo_local = homo_base + homo_specific
        lumo_local = lumo_base + lumo_specific

        # Calculate weights without inplace operations
        weights = self.atom_weights(enhanced_features)
        weights = F.softmax(weights,
                            dim=1)  # Use F.softmax instead of torch.softmax_

        # Save metrics data
        data['_attention_weights'] = weights.clone().detach()
        data['_homo_atomic_weights'] = (homo_local * weights).clone().detach()
        data['_lumo_atomic_weights'] = (lumo_local * weights).clone().detach()

        # Handle masking without inplace operations
        if data['_input_padded'].item():
            mask = data['mask_i'].unsqueeze(-1)
            weights = weights.masked_fill(mask, 0.0)

        # Calculate molecular orbitals
        homo_norm = (homo_local * weights).sum(dim=1)
        lumo_norm = (lumo_local * weights).sum(dim=1)

        # Add normalization parameters
        data['_orbital_mean'] = self.orbital_mean
        data['_orbital_std'] = self.orbital_std

        # Denormalize
        homo = homo_norm * self.orbital_std[0] + self.orbital_mean[0]
        lumo = lumo_norm * self.orbital_std[1] + self.orbital_mean[1]

        # Ensure HOMO < LUMO
        gap = F.softplus(lumo - homo)
        lumo = homo + gap

        data['homo'] = homo
        data['lumo'] = lumo
        data['homo_lumo_gap'] = gap

        return data


class EnhancedESPOutput(nn.Module):

    def __init__(self,
                 mlp: Union[Dict, nn.Module],
                 n_in: int,
                 key_in: str = 'aim_enhancedesp'):
        super().__init__()
        self.key_in = key_in
        self.n_in = n_in  # For compatibility with extra passes MLP generation

        # Add normalization buffers
        #Aimnet2 data
        self.register_buffer('esp_mean',
                             torch.tensor(
                                 [-1.6275646686553955,
                                  1.551095962524414]))  # [min_mean, max_mean]
        self.register_buffer('esp_std',
                             torch.tensor(
                                 [0.5648373961448669,
                                  0.6180419325828552]))  # [min_std, max_std]
        # Mixed 29M
        # self.register_buffer('esp_mean', torch.tensor([-1.7353110313415527, 1.296004295349121]))      # [min_mean, max_mean]
        # self.register_buffer('esp_std', torch.tensor([0.39517319202423096, 0.4828762710094452]))      # [min_std, max_std]

        # only internal data statistics
        # self.register_buffer('esp_mean', torch.tensor([-1.7616937160491943, 1.179327130317688]))      # [min_mean, max_mean]
        # self.register_buffer('esp_std', torch.tensor([0.2889586389064789, 0.33620786666870117]))       # [min_std, max_std]

        # ESP-specific branches
        self.min_branch = ESPBranch(n_in)
        self.max_branch = ESPBranch(n_in)

        # Main MLP
        if not isinstance(mlp, nn.Module):
            mlp = MLP(
                n_in=n_in,
                n_out=2,
                **mlp  # This will unpack hidden, activation_fn, dropout, etc.
            )
        self.mlp = mlp

        # Atomic importance weighting
        self.atom_weights = nn.Sequential(nn.Linear(n_in, n_in // 2),
                                          nn.SiLU(), nn.Linear(n_in // 2, 1))

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Get the module-specific AIM vector
        feature_vector = data[self.key_in].contiguous()

        # Create mask if needed
        mask = None
        if data['_input_padded'].item():
            mask = data['mask_i']

        # Base predictions
        base_esp = self.mlp(feature_vector)
        min_base, max_base = base_esp.chunk(2, dim=-1)

        # Specialized predictions
        min_specific = self.min_branch(feature_vector)
        max_specific = self.max_branch(feature_vector)

        min_local = min_base + min_specific
        max_local = max_base + max_specific

        # Calculate weights
        weights = self.atom_weights(feature_vector)
        weights = F.softmax(weights, dim=1)

        # Save metrics data
        #data['_esp_attention_weights'] = weights.clone().detach()
        data['_esp_min_atomic_weights'] = (min_local *
                                           weights).clone().detach()
        data['_esp_max_atomic_weights'] = (max_local *
                                           weights).clone().detach()

        # Handle masking
        if data['_input_padded'].item():
            mask = data['mask_i'].unsqueeze(-1)
            weights = weights.masked_fill(mask, 0.0)

        # Calculate molecular ESP values
        min_norm = (min_local * weights).sum(dim=1)
        max_norm = (max_local * weights).sum(dim=1)

        # Add normalization parameters
        data['_esp_mean'] = self.esp_mean
        data['_esp_std'] = self.esp_std

        # Denormalize
        esp_min = min_norm * self.esp_std[0] + self.esp_mean[0]
        esp_max = max_norm * self.esp_std[1] + self.esp_mean[1]

        # Ensure ESP_max > ESP_min
        esp_gap = F.softplus(esp_max - esp_min)
        esp_max = esp_min + esp_gap

        data['esp_min'] = esp_min
        data['esp_max'] = esp_max
        data['esp_gap'] = esp_gap

        return data


# Utility branches and helper modules
class OrbitralBranch(nn.Module):
    """Specialized branch for orbital-specific features"""

    def __init__(self, n_in: int):
        super().__init__()

        self.refine = MLP(n_in=n_in,
                          n_out=1,
                          hidden=[n_in // 2, n_in // 4],
                          activation_fn='torch.nn.GELU',
                          dropout=0.1)

        self.norm = nn.LayerNorm(n_in)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        orbital = self.refine(x)
        return orbital


class ESPBranch(nn.Module):
    """Specialized branch for ESP predictions"""

    def __init__(self, n_in: int):
        super().__init__()

        self.refine = MLP(n_in=n_in,
                          n_out=1,
                          hidden=[n_in // 2, n_in // 4],
                          activation_fn='torch.nn.GELU',
                          dropout=0.1)

        self.norm = nn.LayerNorm(n_in)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        esp = self.refine(x)
        return esp


# Additional transformation modules
class OrthogonalFeatureTransformer(nn.Module):
    """Transform features using learned orthogonal transformations to separate property-specific information."""

    def __init__(self, aim_dim: int, num_bases: int = 4):
        super().__init__()
        # Create property-specific orthogonal transformations
        self.property_bases = nn.Parameter(
            torch.randn(num_bases, aim_dim, aim_dim))
        self.property_weights = nn.Parameter(torch.ones(num_bases) / num_bases)

    def forward(self, x: Tensor) -> Tensor:
        # Orthogonalize the bases using batch-wise processing
        q_bases = []
        for i in range(self.property_bases.shape[0]):
            # Use QR decomposition for orthogonalization
            q, r = torch.linalg.qr(self.property_bases[i])
            q_bases.append(q)
        q_bases = torch.stack(q_bases)

        # Apply weighted orthogonal transformations
        weights = F.softmax(self.property_weights, dim=0)
        transformed = torch.zeros_like(x)
        for i in range(q_bases.shape[0]):
            # Batch matrix multiplication for all atoms in the batch
            batch_size, num_atoms, _ = x.shape
            x_reshaped = x.view(-1, x.size(-1))
            q_applied = torch.matmul(x_reshaped, q_bases[i])
            q_applied = q_applied.view(batch_size, num_atoms, -1)
            transformed = transformed + weights[i] * q_applied

        return transformed


class TaskAdaptiveResidual(nn.Module):
    """Apply task-adaptive residual connections to better transform the features."""

    def __init__(self, aim_dim: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(aim_dim), nn.Linear(aim_dim, aim_dim),
                          nn.SiLU(), nn.Linear(aim_dim, aim_dim))
            for _ in range(num_layers)
        ])
        self.gate_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(aim_dim, aim_dim), nn.Sigmoid())
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        original = x
        for layer, gate in zip(self.layers, self.gate_layers):
            # Compute residual
            residual = layer(x)

            # Compute adaptive gate
            gate_value = gate(x)

            # Apply gated residual connection
            x = x + gate_value * residual

        # Final skip connection to original input
        return x + original


class MultiResolutionTransformer(nn.Module):
    """Transform features at multiple resolutions to capture both local and global information."""

    def __init__(self, aim_dim: int):
        super().__init__()
        self.local_transform = nn.Linear(aim_dim, aim_dim // 2)
        self.global_pooling = nn.Sequential(nn.Linear(aim_dim, aim_dim // 2),
                                            nn.SiLU())
        self.output_transform = nn.Linear(aim_dim, aim_dim)

    def forward(self,
                aim_vectors: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        # Local transformation
        local_features = self.local_transform(aim_vectors)

        batch_size, num_atoms, feat_dim = aim_vectors.shape

        # Global pooling with masking if needed
        if mask is not None:
            # Apply mask to get valid atoms only
            valid_vectors = aim_vectors.masked_fill(mask.unsqueeze(-1), 0.0)
            norm = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
            global_context = self.global_pooling(
                valid_vectors.sum(dim=1) / norm)
        else:
            global_context = self.global_pooling(aim_vectors.mean(dim=1))

        # Reshape and expand global context to all atoms
        global_features = global_context.unsqueeze(1).expand(
            batch_size, num_atoms, global_context.size(-1))

        # Concatenate local and global features
        combined = torch.cat([local_features, global_features], dim=-1)

        # Final transformation
        return self.output_transform(combined)


class MolecularMultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for molecular property prediction."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        self.scaling = self.head_dim**-0.5
        self.embed_dim = embed_dim

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, N, E = x.shape

        # QKV transform
        qkv = self.qkv(x)  # (B, N, 3*E)

        # Split heads
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scaling

        if mask is not None:
            # Make sure mask has the right shape
            mask = mask.unsqueeze(1).unsqueeze(2) if mask.dim() < 3 else mask
            attn = attn.masked_fill(mask, float('-inf'))

        # Create new tensor for softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, E)

        return self.out_proj(out)


class EnhancedFeatureTransformationPipeline(nn.Module):
    """
    Enhanced feature transformation pipeline combining orthogonal transformations,
    task-adaptive residuals, multi-resolution context, and multi-head attention.
    """

    def __init__(self, aim_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Define components individually
        self.orthogonal_transformer = OrthogonalFeatureTransformer(aim_dim)
        self.task_adaptive = TaskAdaptiveResidual(aim_dim)
        self.multi_resolution = MultiResolutionTransformer(aim_dim)

        # Add multi-head attention
        self.attention = MolecularMultiHeadAttention(embed_dim=aim_dim,
                                                     num_heads=num_heads,
                                                     dropout=dropout)

        # Final layer normalization
        self.norm = nn.LayerNorm(aim_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Transform through orthogonal and task-adaptive components
        transformed = self.orthogonal_transformer(x)
        transformed = self.task_adaptive(transformed)

        # Apply attention with residual connection
        attn_mask = mask.unsqueeze(1).unsqueeze(
            2) if mask is not None else None
        attn_output = self.attention(transformed, attn_mask)
        transformed = transformed + attn_output

        # Apply multi-resolution transformation
        transformed = self.multi_resolution(transformed, mask)

        # Apply layer normalization
        transformed = self.norm(transformed)

        return transformed
