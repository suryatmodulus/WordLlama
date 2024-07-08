import os
import torch
from torch import nn
from einops.layers.torch import Rearrange

import safetensors.torch as st


class CnnNgramGLU(nn.Module):

    def __init__(self, dims=128, vector_dim=2048, n_gram=3):
        super().__init__()
        self.dims = dims
        self.cnn = nn.Sequential(
            nn.LayerNorm(dims),
            Rearrange("b n d -> b d n"),
            nn.Conv1d(
                in_channels=dims,
                out_channels=vector_dim,
                kernel_size=n_gram,
                padding="same",
            ),
            nn.GLU(),
            Rearrange("b d n -> b n d"),
        )

    def forward(self, tensors):
        mask = tensors["attention_mask"].unsqueeze(dim=-1)
        x = self.cnn(tensors["x"][:, :, 0 : self.dims]) * mask
        x = tensors["x"] * token_weights
        tensors.update({"token_weights": token_weights, "x": x})
        return tensors

    def save(self, filepath: str, **kwargs):
        """Save the model's state_dict using safetensors.

        Args:
            filepath (str): The path where the model should be saved.
        """
        # Ensure tensors are on CPU and converted to the required format for safetensors
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        metadata = {
            "model": "CnnNgramWeight",
        }
        st.save_model(
            model=self,
            filename=os.path.join(filepath, "cnn_weights.safetensors"),
            metadata=metadata,
        )
