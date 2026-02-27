
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