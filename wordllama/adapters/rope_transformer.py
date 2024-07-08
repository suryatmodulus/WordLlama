import torch
from torch import nn
import torch.nn.functional as F


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to the query and key tensors."""
    cos = cos.unsqueeze(1)  # Expand for the batch dimension
    sin = sin.unsqueeze(1)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot


class RotaryPositionalEncoding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Normalization introduced in
    https://arxiv.org/pdf/1910.07467.pdf.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): input tensor to normalize

        Returns:
            Tensor: The output tensor after applying RMSNorm.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return (x_normed * self.scale).to(dtype=x.dtype)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.05):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        # self.norm1 = nn.LayerNorm(embed_dim)
        # self.norm2 = nn.LayerNorm(embed_dim)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.rope = RotaryPositionalEncoding(embed_dim // num_heads)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, position_ids=None):
        # Apply norm first in the forward pass
        src = self.norm1(src)

        batch_size, seq_len, embed_dim = src.size()
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads

        # Prepare position ids if not provided
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=src.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )

        cos, sin = self.rope(src, position_ids)

        # Reshape src to (batch_size, seq_len, num_heads, head_dim)
        q = k = src.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Apply rotary positional embeddings
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        # Reshape back to (batch_size, seq_len, embed_dim)
        q_rot = q_rot.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        k_rot = k_rot.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)

        src2 = self.self_attn(
            q_rot, k_rot, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


## Example usage
# embed_dim = 128
# num_heads = 8
# ff_dim = 512
# batch_size = 32
# seq_len = 16
## position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
#
## Initialize the transformer encoder layer
# encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, ff_dim)
#
## Example input (batch_size, seq_len, embed_dim)
# src = torch.randn(batch_size, seq_len, embed_dim)
#
## Forward pass with position ids
# output = encoder_layer(src)
#
# print(output.shape)  # Expected: torch.Size([32, 16, 128])
