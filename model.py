import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1

class SelfAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_per_head = d_model // num_heads
        
        assert self.d_per_head * num_heads == self.d_model, "d_model must be divisible by num_heads"
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_per_head).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_per_head).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_per_head).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_per_head ** 0.5)
        
        if mask is not None:
            # Paper uses -1e9 to mask out future positions
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(out)


class FFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU() # Strict paper uses ReLU
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))


class Decoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        # Sublayer 1: Masked Multi-Head Self-Attention
        self.self_attn = SelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Sublayer 2: Position-wise Feed-Forward Network
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, tgt_mask=None):
        # 1. Masked Self-Attention (Post-LayerNorm: Add -> Norm)
        attn_out = self.self_attn(x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # 2. Feed-Forward Network (Post-LayerNorm: Add -> Norm)
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.register_buffer(
            "positional_encoding", self._sinusoidal_encoding(config.max_seq_len, config.d_model)
        )
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [
                Decoder(config.d_model, config.num_heads, config.d_ff, config.dropout)
                for _ in range(config.num_layers)
            ]
        )

        self.norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying between token embedding and output projection
        self.output_proj.weight = self.token_embedding.weight

    @staticmethod
    def _sinusoidal_encoding(max_seq_len: int, d_model: int) -> torch.Tensor:
        """Pre-compute the sinusoidal positional encodings (Vaswani et al., 2017)."""
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_seq_len, d_model)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate a causal (lower-triangular) attention mask."""
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) token indices.
        Returns:
            logits: (batch_size, seq_len, vocab_size) raw predictions.
        """
        seq_len = input_ids.size(1)

        x = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x)

        mask = self._causal_mask(seq_len, x.device)

        for layer in self.layers:
            x = layer(x, tgt_mask=mask)

        x = self.norm(x)
        logits = self.output_proj(x)
        return logits
