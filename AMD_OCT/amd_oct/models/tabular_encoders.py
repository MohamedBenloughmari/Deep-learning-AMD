import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """Simple MLP tabular encoder: (B, tab_dim) -> (B, d_model)."""

    def __init__(self, tab_dim: int, d_model: int, dropout: float = 0.3, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tab_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tab: torch.Tensor) -> torch.Tensor:
        return self.mlp(tab)


class TabularAttention(nn.Module):
    """Transformer-block tabular encoder treating each feature as a token."""

    def __init__(self, tab_dim: int, d_model: int, n_heads: int = 6, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.pool = nn.Linear(tab_dim * d_model, d_model)

    def forward(self, tab: torch.Tensor) -> torch.Tensor:
        x = tab.unsqueeze(-1)
        x = self.input_proj(x)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        x = x.flatten(1)
        x = self.pool(x)
        return x


def build_tabular_encoder(cfg, tab_dim: int, d_model: int) -> nn.Module:
    name = str(cfg.get("name", "mlp")).lower()
    dropout = float(cfg.get("dropout", 0.3))
    if name in ("mlp", "tabular_mlp"):
        return TabularMLP(tab_dim, d_model, dropout=dropout, hidden=int(cfg.get("hidden", 256)))
    if name in ("attention", "tabular_attention"):
        return TabularAttention(tab_dim, d_model, n_heads=int(cfg.get("n_heads", 6)), dropout=dropout)
    raise ValueError(f"Unknown tabular encoder: {name}")
