"""
attn_conformer_diffusion.py — EQGAT-diff Style Attention-Enhanced EGNN

Experiment B: Adds dot-product attention over neighbour messages in EGNN
(inspired by EQGAT-diff, Le et al. ICLR 2024).

Research motivation:
  Standard EGNN aggregates messages with unweighted sum-pooling:
    h_i ← h_i + sum_j m_ij

  This treats all neighbours equally, which is wrong:
  - A carbonyl oxygen should attend more strongly to its C=O partner
  - An aromatic ring should couple across the pi-system
  - Chiral centres need to distinguish their 4 neighbours

  EQGAT-diff adds a learnable attention weight per message:
    a_ij = softmax_j( LeakyReLU(w^T [h_i || h_j || rbf_ij]) )
    h_i ← h_i + sum_j a_ij * m_ij

  This gives the model "where to look" per atom, improving:
  - Torsion angle accuracy (MAT-R)
  - Chiral centre discrimination
  - Long-range conjugation (planarity of aromatic systems)

References:
  Le et al. "EQGAT-diff: a novel equivariant graph attention model for
  molecular 3D generation" ICLR 2024. arXiv:2306.01916.

  Veličković et al. "Graph Attention Networks" ICLR 2018. arXiv:1710.10903.
  (Original GAT — foundation for attention in GNNs)

Usage:
  from models.attn_conformer_diffusion import AttnConformerDiffusion
  model = AttnConformerDiffusion(hidden_dim=256, num_layers=6)
  loss_dict = model.get_loss(coords, atom_types, edge_index, bond_types, batch_idx)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict

from models.conformer_diffusion import (
    cosine_beta_schedule, sinusoidal_embedding, remove_com, rbf_features,
    ConformerDiffusion  # we inherit the full diffusion wrapper
)


# =============================================================================
# ATTENTION-ENHANCED EQUIVARIANT LAYER (EQGAT-diff style)
# =============================================================================

class AttnEquivariantLayer(nn.Module):
    """
    E(3)-equivariant message passing layer with attention-weighted aggregation.

    Architecture change vs. standard EGNN (EquivariantLayer):
      Standard: h_i ← LayerNorm(h_i + MLP([h_i, sum_j m_ij]))
      Attention: h_i ← LayerNorm(h_i + MLP([h_i, sum_j a_ij * m_ij]))

    where a_ij = softmax_over_j(LeakyReLU(W_attn [h_i || h_j || rbf_ij]))

    The coordinate update is UNCHANGED (already equivariant):
      x_i ← x_i + (1/deg_i) * sum_j phi_x(m_ij) * unit_vec_ij

    Dropout is applied to attention weights (Srivastava 2014 — prevents
    over-reliance on specific neighbours, important for chiral centres).
    """

    def __init__(self, hidden_dim: int, num_rbf: int = 20,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        # --- Edge message MLP (same as EGNN) ---
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # --- Multi-head attention scoring (EQGAT-diff) ---
        # Input: [h_i || h_j || rbf] → attention logit per head
        self.attn_gate = nn.Linear(hidden_dim * 2 + num_rbf, num_heads, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # --- Coordinate update (equivariant scalar per edge) ---
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

        # --- Node update MLP (operates on attention-weighted message) ---
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                h: torch.Tensor,           # (N, hidden_dim)
                x: torch.Tensor,           # (N, 3)
                edge_index: torch.Tensor,  # (2, E)
                bond_embed: torch.Tensor   # (E, hidden_dim)
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index          # row=source, col=destination
        N = x.size(0)
        E = row.size(0)

        # ── Pairwise geometry ─────────────────────────────────────────────
        diff = x[row] - x[col]                                         # (E, 3)
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6) # (E, 1)
        unit_vec = diff / dist                                          # (E, 3)
        rbf = rbf_features(dist, num_rbf=self.num_rbf)                # (E, num_rbf)

        # ── Edge messages (same as EGNN) ──────────────────────────────────
        edge_input = torch.cat([h[row], h[col], rbf], dim=-1)         # (E, 2H+rbf)
        m_ij = self.edge_mlp(edge_input)                               # (E, H)
        m_ij = m_ij + bond_embed                                       # residual bond info

        # ── EQGAT-diff Multi-Head Attention ───────────────────────────────
        # Compute attention logits: (E, num_heads)
        attn_input = torch.cat([h[row], h[col], rbf], dim=-1)
        attn_logits = self.attn_gate(attn_input)                       # (E, num_heads)
        attn_logits = F.leaky_relu(attn_logits, negative_slope=0.2)

        # Softmax over incoming edges per destination node, per head
        # Use scatter softmax: for each col (dest), normalize logits over all row (src)
        # Implementation: subtract max per-node for numerical stability, then exp/sum
        attn_weights = self._scatter_softmax(attn_logits, col, N)     # (E, num_heads)
        attn_weights = self.attn_dropout(attn_weights)                 # (E, num_heads)

        # Weight messages by attention: broadcast heads over hidden_dim
        # m_ij: (E, H), attn_weights: (E, num_heads)
        # Reshape m_ij to (E, num_heads, head_dim), weight, reshape back
        m_ij_reshaped = m_ij.view(E, self.num_heads, self.head_dim)   # (E, H, D)
        attn_m = (m_ij_reshaped * attn_weights.unsqueeze(-1))         # (E, H, D)
        attn_m = attn_m.view(E, self.hidden_dim)                      # (E, H*D)

        # ── Coordinate update (unchanged — equivariant) ───────────────────
        coord_weight = self.coord_mlp(m_ij)                            # (E, 1)
        coord_update = coord_weight * unit_vec                         # (E, 3)

        x_agg = torch.zeros_like(x)
        x_agg.scatter_add_(0, col.unsqueeze(-1).expand(-1, 3), coord_update)

        # Degree normalization: bidirectional graph → divide by 2 (FIX-1 from v5)
        degree = torch.zeros(N, 1, device=x.device)
        degree.scatter_add_(0, col.unsqueeze(-1),
                            torch.ones(E, 1, device=x.device))
        degree = (degree / 2.0).clamp(min=1.0)
        x_new = x + x_agg / degree

        # ── Node update with attention-weighted aggregation ───────────────
        m_agg = torch.zeros_like(h)
        m_agg.scatter_add_(0, col.unsqueeze(-1).expand(-1, self.hidden_dim), attn_m)

        h_new = self.node_mlp(torch.cat([h, m_agg], dim=-1))
        h_new = self.layer_norm(h + h_new)

        return h_new, x_new

    @staticmethod
    def _scatter_softmax(logits: torch.Tensor, index: torch.Tensor,
                         num_nodes: int) -> torch.Tensor:
        """
        Scatter softmax: for each destination node, normalize attention
        logits over all incoming source nodes.

        logits: (E, num_heads)
        index:  (E,) destination node indices (col)
        Returns: (E, num_heads) attention weights summing to 1 per node per head
        """
        E, H = logits.shape

        # Subtract max per node for numerical stability
        max_logits = torch.full((num_nodes, H), float('-inf'), device=logits.device)
        max_logits.scatter_reduce_(0, index.unsqueeze(-1).expand(-1, H),
                                   logits, reduce='amax', include_self=True)
        shifted = logits - max_logits[index]  # (E, H)

        exp_logits = shifted.exp()

        # Sum exponentials per destination node
        exp_sum = torch.zeros(num_nodes, H, device=logits.device)
        exp_sum.scatter_add_(0, index.unsqueeze(-1).expand(-1, H), exp_logits)

        # Normalize
        attn = exp_logits / (exp_sum[index] + 1e-8)   # (E, H)
        return attn


# =============================================================================
# ATTENTION CONFORMER DENOISER
# =============================================================================

class AttnConformerDenoiser(nn.Module):
    """
    Drop-in replacement for ConformerDenoiser using AttnEquivariantLayer.

    Identical interface, identical output shape — only internal aggregation
    changes from sum-pooling to attention-weighted pooling.
    """

    def __init__(self,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_atom_types: int = 10,
                 num_bond_types: int = 5,
                 num_rbf: int = 20,
                 time_dim: int = 128,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.time_dim = time_dim

        self.atom_embed = nn.Embedding(54, hidden_dim)
        self.bond_embed = nn.Embedding(num_bond_types + 1, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Use AttnEquivariantLayer instead of EquivariantLayer
        self.layers = nn.ModuleList([
            AttnEquivariantLayer(hidden_dim, num_rbf=num_rbf,
                                 num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output head: predict x_0 (x_0 parameterization, same as v5)
        self.coord_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)
        )

    def forward(self,
                x_noisy: torch.Tensor,
                t: torch.Tensor,
                atom_types: torch.Tensor,
                edge_index: torch.Tensor,
                bond_types: torch.Tensor,
                batch_idx: torch.Tensor) -> torch.Tensor:
        h = self.atom_embed(atom_types.clamp(0, 53))
        t_emb = sinusoidal_embedding(t.float(), self.time_dim)
        t_emb = self.time_mlp(t_emb)
        h = h + t_emb[batch_idx]
        bond_feat = self.bond_embed(bond_types.clamp(0, 5))
        x = x_noisy
        for layer in self.layers:
            h, x = layer(h, x, edge_index, bond_feat)
        delta_x = self.coord_pred(h)
        x_0_pred = x + delta_x
        return x_0_pred


# =============================================================================
# ATTENTION CONFORMER DIFFUSION (full model wrapper)
# =============================================================================

class AttnConformerDiffusion(nn.Module):
    """
    E(3)-equivariant diffusion model using EQGAT-diff attention EGNN.

    Drop-in replacement for ConformerDiffusion — identical get_loss()
    and ddim_sample() interface, so mol_train.py needs only one import swap.
    """

    def __init__(self,
                 num_timesteps: int = 1000,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_rbf: int = 20,
                 time_dim: int = 128,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.num_timesteps = num_timesteps

        betas = cosine_beta_schedule(num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1 - alphas_cumprod))

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance',
                             torch.log(posterior_variance.clamp(min=1e-20)))

        snr = alphas_cumprod / (1 - alphas_cumprod)
        self.register_buffer('snr', snr)

        # Use the attention-enhanced denoiser
        self.denoiser = AttnConformerDenoiser(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            time_dim=time_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        from models.geometry_constraints import GeometryConstraints
        self.geometry = GeometryConstraints(
            bond_weight=10.0,
            angle_weight=3.0,
            torsion_weight=1.0,
            repulsion_weight=5.0,
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor,
                 batch_idx: torch.Tensor) -> torch.Tensor:
        return a[t][batch_idx].unsqueeze(-1)

    def q_sample(self, x_0, t, batch_idx, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        noise = remove_com(noise, batch_idx)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, batch_idx)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, batch_idx)
        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        x_t = remove_com(x_t, batch_idx)
        return x_t, noise

    def get_loss(self, x_0, atom_types, edge_index, bond_types, batch_idx,
                 geometry_weight=1.0, epoch=1, max_epochs=300,
                 min_snr_gamma=5.0) -> Dict:
        """Identical interface to ConformerDiffusion.get_loss()."""
        device = x_0.device
        B = int(batch_idx.max().item()) + 1
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        x_t, noise = self.q_sample(x_0, t, batch_idx)

        # x_0 parameterization: predict clean coords directly
        x_0_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)
        x_0_pred = remove_com(x_0_pred, batch_idx)

        # Derive epsilon from x_0_pred
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, batch_idx)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, batch_idx)
        noise_pred = (x_t - sqrt_alpha * x_0_pred) / sqrt_one_minus.clamp(min=1e-6)

        mse_per_atom = ((noise_pred - noise) ** 2).sum(-1)

        # Per-molecule MSE (FIX-6 from v5: avoid large-mol bias)
        mse_per_mol = torch.zeros(B, device=device)
        mol_counts = torch.zeros(B, device=device)
        mse_per_mol.scatter_add_(0, batch_idx, mse_per_atom)
        mol_counts.scatter_add_(0, batch_idx, torch.ones_like(mse_per_atom))
        mse_per_mol = mse_per_mol / mol_counts.clamp(min=1)

        # Min-SNR weighting
        snr_t = self.snr[t]
        snr_weight = torch.minimum(snr_t, torch.full_like(snr_t, min_snr_gamma)) \
                     / snr_t.clamp(min=1e-8)
        mse_loss = (snr_weight * mse_per_mol).mean()

        # Geometry loss (same as ConformerDiffusion.get_loss FIX-4: all timesteps)
        geo_loss = self._compute_geometry_loss(
            x_0_pred, atom_types, edge_index, bond_types, batch_idx,
            include_angles=True, include_torsions=False
        )

        total_loss = mse_loss + geometry_weight * geo_loss
        return {
            'total': total_loss,
            'mse':   mse_loss.detach(),
            'geo':   geo_loss.detach() if isinstance(geo_loss, torch.Tensor)
                     else torch.tensor(0.0),
        }

    def _compute_geometry_loss(self, pos, atom_types, edge_index, bond_types,
                                batch_idx, include_angles=True,
                                include_torsions=False) -> torch.Tensor:
        total, _ = self.geometry.compute_total_loss(
            pos, atom_types, edge_index, bond_types, batch_idx,
            include_angles=include_angles,
            include_torsions=include_torsions,
        )
        return total

    @torch.no_grad()
    def ddim_sample(self, atom_types, edge_index, bond_types, batch_idx,
                    num_steps=50, eta=0.0) -> torch.Tensor:
        """DDIM sampling — identical to ConformerDiffusion.ddim_sample()."""
        device = atom_types.device
        N = atom_types.size(0)
        B = int(batch_idx.max().item()) + 1

        step_size = self.num_timesteps // num_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=device).flip(0)

        x_t = remove_com(torch.randn(N, 3, device=device), batch_idx)

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val.item(), dtype=torch.long, device=device)

            x_0_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)
            x_0_pred = remove_com(x_0_pred, batch_idx)

            alpha_t = self.alphas_cumprod[t][batch_idx].unsqueeze(-1)

            if i == len(timesteps) - 1:
                x_t = x_0_pred
            else:
                t_next_val = timesteps[i + 1].item()
                t_next = torch.full((B,), t_next_val, dtype=torch.long, device=device)
                alpha_next = self.alphas_cumprod[t_next][batch_idx].unsqueeze(-1)

                sqrt_one_minus_at = torch.sqrt(1.0 - alpha_t).clamp(min=1e-6)
                noise_pred = (x_t - torch.sqrt(alpha_t) * x_0_pred) / sqrt_one_minus_at

                ratio = (alpha_t / alpha_next.clamp(min=1e-8)).clamp(max=1.0)
                sigma = eta * torch.sqrt(
                    (1.0 - alpha_next) / (1.0 - alpha_t).clamp(min=1e-8)
                ) * torch.sqrt((1.0 - ratio).clamp(min=0.0))

                direction = torch.sqrt((1.0 - alpha_next - sigma ** 2).clamp(min=0.0)) \
                            * noise_pred
                noise = remove_com(torch.randn_like(x_t), batch_idx) if eta > 0 else 0.0
                x_t = torch.sqrt(alpha_next) * x_0_pred + direction + sigma * noise

        return x_t
