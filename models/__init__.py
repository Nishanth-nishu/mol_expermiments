"""
models/__init__.py — Model registry for NExT-Mol Gen

Available models:
  - ConformerDiffusion: DDPM-based E(3)-equivariant diffusion (baseline + experiments A, D)
  - FlowMatchingConformer: Conditional Flow Matching (experiment C)
  - AttnConformerDiffusion: EQGAT-diff attention EGNN (experiment B)
"""

from models.conformer_diffusion import ConformerDiffusion, ConformerDenoiser, remove_com
from models.flow_matching import FlowMatchingConformer

__all__ = [
    "ConformerDiffusion",
    "ConformerDenoiser",
    "FlowMatchingConformer",
    "remove_com",
]
