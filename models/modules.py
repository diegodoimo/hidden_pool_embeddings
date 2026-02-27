import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def mean_pool(hidden_states, attention_mask):
    """
    Perform masked mean pooling over sequence dimension.

    Args:
        hidden_states: (batch_size, seq_len, hidden_dim)
        attention_mask: (batch_size, seq_len)

    Returns:
        pooled: (batch_size, hidden_dim)
    """
    # Expand mask to match hidden_states dimensions
    mask = attention_mask.unsqueeze(-1)  # (B, L, 1)

    # Compute masked sum
    masked_sum = (hidden_states * mask).sum(dim=1)  # (B, H)

    # Compute mask sum with numerical stability
    mask_sum = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)

    # Compute mean
    return masked_sum / mask_sum


class EncoderWithPooling(nn.Module):
    """
    Wraps an encoder model to perform pooling and L2 normalization in the forward pass.
    The wrapped model's forward should return an object with `last_hidden_state` (e.g. BaseModelOutput).
    """

    def __init__(self, model, pool_fn):
        super().__init__()
        self.model = model
        self.pool_fn = pool_fn

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = self.pool_fn(output.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=1)


def add_pooling_layers(model, pool_fn):
    """Wrap a model so its forward pass includes pooling and L2 normalization."""
    return EncoderWithPooling(model, pool_fn)


class Normalize(nn.Module):
    """Normalizes embeddings to unit length (L2 norm)"""

    def forward(self, embeddings: Tensor) -> Tensor:
        return F.normalize(embeddings, p=2, dim=1)


class Projection(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.up = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)
        self.down = nn.Linear(
            in_features=hidden_dim, out_features=input_dim, bias=False
        )

    def forward(self, hidden_states):
        hidden_states = self.up(hidden_states)
        hidden_states = self.down(hidden_states)
        return hidden_states


class MeanPooling(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states, attention_mask):
        """
        Perform masked mean pooling over sequence dimension.

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            attention_mask: (batch_size, seq_len)

        Returns:
            pooled: (batch_size, hidden_dim)
        """
        # Expand mask to match hidden_states dimensions
        mask = attention_mask.unsqueeze(-1)  # (B, L, 1)

        # Compute masked sum
        masked_sum = (hidden_states * mask).sum(dim=1)  # (B, H)

        # Compute mask sum with numerical stability
        mask_sum = mask.sum(dim=1).clamp(min=self.eps)  # (B, 1)

        # Compute mean
        return masked_sum / mask_sum