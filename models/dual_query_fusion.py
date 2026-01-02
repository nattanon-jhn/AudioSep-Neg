# models/dual_query_fusion.py

import torch
import torch.nn as nn


class DualQueryFusion(nn.Module):
    """
    Fuse positive & negative CLAP embeddings
    """

    def __init__(self, emb_dim=512, out_dim=512):
        super().__init__()
        self.proj = nn.Linear(emb_dim * 2, out_dim)

    def forward(self, e_pos, e_neg):
        """
        e_pos, e_neg: (B, 512)
        """
        x = torch.cat([e_pos, e_neg], dim=-1)
        return self.proj(x)
