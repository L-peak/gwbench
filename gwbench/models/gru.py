"""A compact global GRU for multi-site groundwater forecasting."""

from __future__ import annotations

import torch
from torch import nn


class GroundwaterGRU(nn.Module):
    """GRU with a learned site embedding.

    The same model is trained across all wells; the site embedding lets it
    learn site-specific offsets and dynamics without training a separate model
    per well.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        num_sites: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        site_embedding_dim: int = 8,
        use_site_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.use_site_embedding = use_site_embedding
        if use_site_embedding:
            self.site_embedding = nn.Embedding(num_sites, site_embedding_dim)
            projection_input = input_dim + site_embedding_dim
        else:
            self.site_embedding = None
            projection_input = input_dim
        self.input_projection = nn.Linear(projection_input, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, site_id: torch.Tensor) -> torch.Tensor:
        # x: [batch, lookback, input_dim]
        if self.site_embedding is not None:
            site = self.site_embedding(site_id).unsqueeze(1)
            site = site.expand(-1, x.size(1), -1)
            h = torch.cat([x, site], dim=-1)
        else:
            h = x
        h = torch.relu(self.input_projection(h))
        output, _ = self.gru(h)
        return self.head(output[:, -1, :]).squeeze(-1)
