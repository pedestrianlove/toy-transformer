import torch
from torch import nn
import torch.nn.functional as F

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
