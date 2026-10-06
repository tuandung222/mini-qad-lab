"""
Lightweight model architectures designed for quick training and QAD benchmarking.
Includes a native PyTorch MiniTransformerLM that runs in seconds on Apple Silicon.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN attention
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out

        # Pre-LN MLP
        x = x + self.mlp(self.ln2(x))
        return x


class MiniTransformerLM(nn.Module):
    """
    Compact Causal Transformer Language Model.
    Designed for fast, zero-dependency experimentation and QAD recovery demonstrations.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        max_seq_len: Maximum sequence length
        d_ff: Hidden dimension of feed-forward network (default: 4 * d_model)
    """

    def __init__(
        self,
        vocab_size: int = 2000,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 128,
        d_ff: Optional[int] = None,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        d_ff = d_ff or (4 * d_model)

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, seq_len = input_ids.size()
        assert (
            seq_len <= self.max_seq_len
        ), f"Sequence length {seq_len} exceeds max {self.max_seq_len}"

        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.tok_embed(input_ids) + self.pos_embed(positions)
        x = self.drop(x)

        # Causal mask (upper triangular)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
