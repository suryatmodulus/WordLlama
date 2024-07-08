import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from .local_encoder import LocalTransformerEncoder

import safetensors.torch as st


# class AlibiPositionalEncoding(nn.Module):
#     def __init__(self, num_heads, max_len=1024):
#         super().__init__()
#         slopes = torch.Tensor(self._get_slopes(num_heads))
#         alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_len).unsqueeze(
#             0
#         ).unsqueeze(0).expand(num_heads, -1, -1)
#         alibi = alibi.view(num_heads, 1, max_len)
#         self.register_buffer("alibi", alibi)
#
#     def _get_slopes(self, num_heads):
#         def get_slopes_power_of_2(n):
#             start = 2 ** (-(2 ** -(math.log2(n) - 3)))
#             ratio = start
#             return [start * ratio**i for i in range(n)]
#
#         if math.log2(num_heads).is_integer():
#             return get_slopes_power_of_2(num_heads)
#         else:
#             closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
#             return (
#                 get_slopes_power_of_2(closest_power_of_2)
#                 + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
#                     : num_heads - closest_power_of_2
#                 ]
#             )
#
#     def forward(self, x):
#         return self.alibi[:, :, : x.size(1)]


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=1024):
#         super().__init__()
#
#         # Create a long enough positional encoding
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         pe = pe.unsqueeze(0).transpose(0, 1)
#
#         # Register the positional encoding as a buffer
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]
# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model: int, dropout: float = 0.05, max_len: int = 1024):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
#         )
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer("pe", pe)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         x = x + self.pe[: x.size(0)]
#         return self.dropout(x)


# class PositionalEncoding(nn.Module):
#    def __init__(self, d_model, max_len=1024):
#        super(PositionalEncoding, self).__init__()
#        position = torch.arange(0, max_len).unsqueeze(1)
#        div_term = torch.exp(
#            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
#        )
#        encoding = torch.zeros(max_len, d_model)
#        encoding[:, 0::2] = torch.sin(position * div_term)
#        encoding[:, 1::2] = torch.cos(position * div_term)
#        self.register_buffer("encoding", encoding.unsqueeze(0))
#
#    def forward(self, x):
#        return x + self.encoding[:, : x.size(1)]
# def prepare_mask(attention_mask):
#     # Convert 0s to -inf and 1s to 0s
#     mask = (
#         attention_mask.float()
#         .masked_fill(attention_mask == 0, float("-inf"))
#         .masked_fill(attention_mask == 1, float(0.0))
#     )
#
#     # Expand dimensions to (batch_size, 1, seq_len, seq_len)
#     mask = mask.unsqueeze(1).unsqueeze(2)
#
#     # Create a square mask
#     mask = mask.expand(-1, -1, mask.size(-1), -1)
#
#     return mask
def prepare_mask(attention_mask):
    # Ensure the mask is a boolean tensor
    bool_mask = attention_mask.bool()

    # Expand dimensions to (batch_size, 1, 1, seq_len)
    expanded_mask = bool_mask.unsqueeze(1).unsqueeze(2)

    return expanded_mask


class AttentionReducer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=8,
        ff_dim=512,
        num_layers=3,
        dim_head=48,
        window_size=64,
        # n_steps=4,
        input_dim=1024,
        # reduce=4,
    ):
        super().__init__()
        # layer = nn.TransformerEncoderLayer(
        #    embed_dim,
        #    num_heads,
        #    dim_feedforward=ff_dim,
        #    activation="gelu",
        #    batch_first=True,
        #    norm_first=True,
        # )
        # self.n_steps = n_steps
        self.embed_dim = embed_dim
        # self.num_heads = num_heads
        # self.reduce = reduce
        # self.window = 4
        self.encoder = LocalTransformerEncoder(
            dim=embed_dim,
            depth=num_layers,
            local_attn_window_size=window_size,
            heads=num_heads,
            dim_head=dim_head,
            use_xpos=True,
            ff_dropout=0.02,
            attn_dropout=0.02,
        )
        # self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        # self.encoder = TransformerEncoderLayer(embed_dim, num_heads, ff_dim)
        # self.to_reducer_batch = Rearrange("b (s2 s1 s) e -> (b s2) (s1 s) e", s=reduce, s1=self.window)
        # self.to_pooler_batch = Rearrange("b2 (s1 s) e -> (b2 s1) s e", s=reduce, s1=self.window)
        # self.half_seq = Rearrange("b (s1 s) e -> (b s1) s e", s1=2)
        # self.pos_encoder = PositionalEncoding(embed_dim)
        # self.rmsnorm = RMSNorm(input_dim)
        # self.alibi = AlibiPositionalEncoding(num_heads)
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(embed_dim, input_dim)

    # def avg_pool_masked(self, batch_size, pooler_x, pooler_mask):
    #     pooler_x = (pooler_x * pooler_mask).sum(
    #         dim=1
    #     ) / pooler_mask.sum(dim=1)
    #     pooler_mask = pooler_mask.any(dim=1).squeeze()
    #     pooler_x = rearrange(pooler_x, "(b s2 s1) e -> b (s2 s1) e", s1=4, b=batch_size)
    #     return pooler_x, pooler_mask
    # def avg_pool_masked(self, batch_size, pooler_x, pooler_mask):
    #     # Calculate the sum of elements within each pool
    #     sum_pooler_x = (pooler_x * pooler_mask).sum(dim=1)
    #
    #     # Calculate the number of elements in each pool
    #     num_elements = pooler_mask.sum(dim=1)
    #
    #     # Avoid division by zero by setting zero sums to one (or another small non-zero number)
    #     # This modification ensures that we do not divide by zero
    #     # The result where num_elements is zero will be zero anyway because sum_pooler_x is zero in those cases
    #     safe_num_elements = num_elements.clone()
    #     safe_num_elements[safe_num_elements == 0] = 1
    #
    #     # Perform the division
    #     avg_pooler_x = sum_pooler_x / safe_num_elements
    #
    #     # Calculate the mask indicating where any value was true (pooled over non-zero elements)
    #     pooler_mask = pooler_mask.any(dim=1).squeeze()
    #
    #     # Rearrange back to the original batch and sequence layout
    #     avg_pooler_x = rearrange(avg_pooler_x, "(b s2 s1) e -> b (s2 s1) e", s1=self.window, b=batch_size)
    #
    #     return avg_pooler_x, pooler_mask

    def forward(self, tensors):
        x = tensors["x"]
        mask = tensors["attention_mask"]
        bs = x.size(0)
        seq = x.size(1)

        reducer_x = x[:, :, 0 : self.embed_dim]
        reducer_mask = prepare_mask(mask)
        reducer_x = self.encoder(reducer_x, reducer_mask)

        # reducer_mask = tensors["attention_mask"]
        # reducer_x = x[:,:,0:self.embed_dim]
        ##reducer_x = self.rmsnorm(reducer_x + self.pos_encoder(reducer_x))
        ##reducer_x = reducer_x * math.sqrt(self.embed_dim)
        ##reducer_x = self.pos_encoder(reducer_x)

        # for _ in range(self.n_steps):
        #    # reshape to small sequences
        #    #reducer_x = self.to_reducer_batch(reducer_x)
        #    #reducer_mask = self.to_reducer_batch(mask.unsqueeze(-1))
        #    valid_idx = reducer_mask.squeeze(-1).any(dim=-1)
        #    valid_x = reducer_x[valid_idx]
        #    valid_mask = reducer_mask[valid_idx]
        #    #position_ids = torch.arange(valid_x.size(1), device=x.device).unsqueeze(0).repeat(valid_x.size(0), 1)

        #    ## # Generate Alibi bias
        #    ## seq_len = valid_x.size(1)
        #    ## alibi_bias = self.alibi(valid_x)  # Shape: (num_heads, 1, seq_len)
        #    ##
        #    ## # Expand alibi_bias to match the batch size and sequence length
        #    ## alibi_bias = alibi_bias.unsqueeze(0).expand(valid_x.size(0), -1, seq_len, -1)  # (N, num_heads, S, S)
        #    ## #alibi_bias = alibi_bias.expand(-1, -1, seq_len, -1)  # (N, num_heads, S, S)
        #    ##
        #    ## # Apply padding mask to alibi_bias
        #    ## padding_mask = valid_mask.logical_not().float().unsqueeze(1).unsqueeze(2)  # (N, 1, 1, S)
        #    ## padding_mask = padding_mask.expand(-1, self.num_heads, seq_len, -1)  # (N, num_heads, S, S)
        #    ## alibi_bias = alibi_bias.masked_fill(padding_mask.bool(), float('-inf'))
        #    ##
        #    ## # Reshape for the encoder
        #    ## alibi_bias = alibi_bias.reshape(-1, seq_len, seq_len)  # Shape: (N*num_heads, S, S)
        #    ##
        #    ## # encode
        #    ## valid_x = self.encoder(
        #    ##     src=valid_x,
        #    ##     mask=alibi_bias,
        #    ## )

        #    ## reducer_x[valid_idx] = valid_x

        #    # encode
        #    valid_x = self.encoder(
        #        src=valid_x,
        #        src_key_padding_mask=valid_mask.logical_not(),
        #        #position_ids=position_ids
        #    )
        #    reducer_x[valid_idx] = valid_x.to(dtype=reducer_x.dtype)

        #    # rearrange to have seq double batch
        #    reducer_x = self.half_seq(reducer_x)
        #    reducer_mask = self.half_seq(reducer_mask.unsqueeze(-1)).squeeze(-1)

        ##reducer_x = rearrange(reducer_x, "(b s2) (s1 s) e -> b (s2 s1 s) e", s=self.reduce, s1=self.window, b=bs)
        # reducer_x = rearrange(reducer_x, "(b s1) s e -> b (s1 s) e", b=bs)
        tensors.update({"x": self.norm(x + self.proj(reducer_x))})
        return tensors

        # x = self.to_pooler_batch(x)
        # pooler_x = self.to_pooler_batch(reducer_x)
        # pooler_mask = self.to_pooler_batch(reducer_mask)

        # # average pool both
        # #x, _ = self.avg_pool_masked(bs, x, pooler_mask)
        # pooler_x, pooler_mask = self.avg_pool_masked(bs, pooler_x, pooler_mask)
        # # pooler_x = (pooler_x * pooler_mask).sum(
        # #     dim=1
        # # ) / pooler_mask.sum(dim=1)
        # # pooler_mask = pooler_mask.any(dim=1).squeeze()
        # # pooler_x = rearrange(pooler_x, "(b s2 s1) e -> b (s2 s1) e", s1=4, b=bs)
        # # sum as residual and norm
        # #x = self.norm(x + self.proj(pooler_x))
        # x = self.proj(pooler_x)
        # mask = rearrange(pooler_mask, "(b s2 s1) -> b (s2 s1)", s1=self.window, b=bs)
        # tensors.update({"x": x, "attention_mask": mask})
        # return tensors

    def save(self, filepath: str, **kwargs):
        """Save the model's state_dict using safetensors.

        Args:
            filepath (str): The path where the model should be saved.
        """
        # Ensure tensors are on CPU and converted to the required format for safetensors
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        metadata = {
            "model": "ReducedSequenceAttention",
        }
        st.save_model(
            model=self,
            filename=os.path.join(filepath, "reduced_seq_attn.safetensors"),
            metadata=metadata,
        )
