import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

import safetensors.torch as st


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding[:, : x.size(1)]


class ReducedSequenceAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, input_dim=1024):
        super(ReducedSequenceAttention, self).__init__()
        self.embed_dim = embed_dim
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.feature_transform = nn.Sequential(
            Rearrange("b s e -> b e s"),
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=embed_dim,
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            Rearrange("b e s -> b s e"),
        )
        self.qk = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.sequence_expansion = nn.Sequential(
            Rearrange("b s e -> b e s"),
            nn.ConvTranspose1d(
                in_channels=embed_dim,
                out_channels=input_dim,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.GELU(),
            Rearrange("b e s -> b s e"),
        )
        self.output_layer_norm = nn.LayerNorm(input_dim)

    def forward(self, tensors: dict):
        x = tensors["x"]
        mask = tensors["attention_mask"]

        x_reduced = self.feature_transform(x)
        x_norm = self.layer_norm(self.pos_encoder(x_reduced))

        if mask is not None:
            mask_reduced = (
                F.max_pool1d(
                    mask.float().unsqueeze(1), kernel_size=2, stride=2, padding=0
                )
                .squeeze(1)
                .bool()
            )
        else:
            mask_reduced = None

        qk = self.qk(x_norm)
        attention_output, _ = self.attention(
            qk, qk, self.v(x_norm), key_padding_mask=mask_reduced.logical_not()
        )
        x_expanded = self.sequence_expansion(attention_output)

        tensors["x"] = self.output_layer_norm(x + x_expanded)
        return tensors

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
