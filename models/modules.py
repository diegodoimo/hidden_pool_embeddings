import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def mean_pool(last_hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
    masked_sum = (last_hidden_states * mask).sum(dim=1)  # (B, H)
    mask_sum = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
    return masked_sum / mask_sum


class AttentionPooling(nn.Module):
    """Multi-head self-attention pooling over a sequence of hidden-layer
    representations into a single vector.

    Applies standard multi-head self-attention (Q, K, V all projected from the
    input), then mean-pools over the sequence dimension.

    Args:
        hidden_size: Dimension of each hidden representation.
        num_attention_heads: Number of attention heads.
            ``head_dim`` is computed as ``hidden_size // num_attention_heads``.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        assert hidden_size % num_attention_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_attention_heads ({num_attention_heads})"
        )
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.hidden_size = hidden_size

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, K, D) where K is the number of layer representations.

        Returns:
            pooled: (B, D) — the mean-pooled output.
            attn_weights: (B, num_heads, K, K) — attention weights.
        """
        hidden_states = hidden_states.to(self.q_proj.weight.dtype)
        B, K, D = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, K, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, K, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, K, self.num_attention_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

        attn_output = torch.matmul(attn_weights, v)                    # (B, H, K, hd)
        attn_output = attn_output.transpose(1, 2).reshape(B, K, D)    # (B, K, D)
        attn_output = self.o_proj(attn_output)                         # (B, K, D)

        pooled = attn_output.mean(dim=1)                               # (B, D)
        return pooled, attn_weights


class GatedAttention(nn.Module):
    """Multi-head self-attention with Qwen3-style headwise gating for pooling.

    Like :class:`AttentionPooling`, but the Q projection outputs additional
    per-head gate scalars.  After computing attention, the output of each head
    is multiplied by ``sigmoid(gate_score)``, following the Qwen3 headwise
    attention output gating approach.

    Reference:
        https://huggingface.co/QwQZh/gated_attention/blob/main/1B_gate_headwise/modeling_qwen3.py

    Args:
        hidden_size: Dimension of each hidden representation.
        num_attention_heads: Number of attention heads.
            ``head_dim`` is computed as ``hidden_size // num_attention_heads``.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        assert hidden_size % num_attention_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_attention_heads ({num_attention_heads})"
        )
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.hidden_size = hidden_size

        # Q proj outputs extra num_heads gate scalars (Qwen3 headwise gating)
        self.q_proj = nn.Linear(hidden_size, hidden_size + num_attention_heads, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, K, D) where K is the number of layer representations.

        Returns:
            pooled: (B, D) — the mean-pooled, gated output.
            attn_weights: (B, num_heads, K, K) — attention weights.
        """
        hidden_states = hidden_states.to(self.q_proj.weight.dtype)
        B, K, D = hidden_states.shape

        q = self.q_proj(hidden_states)                                  # (B, K, D + H)
        q, gate_score = q.split([self.hidden_size, self.num_attention_heads], dim=-1)
        gate_score = gate_score.view(B, K, self.num_attention_heads, 1) # (B, K, H, 1)

        q = q.view(B, K, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, K, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, K, self.num_attention_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

        attn_output = torch.matmul(attn_weights, v)                    # (B, H, K, hd)
        attn_output = attn_output.transpose(1, 2)                      # (B, K, H, hd)

        # Qwen3-style headwise gating
        attn_output = attn_output * torch.sigmoid(gate_score)          # (B, K, H, hd)

        attn_output = attn_output.reshape(B, K, D)                    # (B, K, D)
        attn_output = self.o_proj(attn_output)                         # (B, K, D)

        pooled = attn_output.mean(dim=1)                               # (B, D)
        return pooled, attn_weights


class EncoderWithPooling(nn.Module):
    """
    Wraps an encoder model to perform pooling and L2 normalization in the forward pass.
    The wrapped model's forward should return an object with `last_hidden_state` (e.g. BaseModelOutput).

    If *projection_layers* is provided it is applied between pooling and
    normalization.  This is needed for models like ``embeddinggemma-300m``
    whose SentenceTransformer checkpoint ships two Dense layers (768→3072→768)
    on top of mean-pooling.
    """

    def __init__(self, model, pool_fn, projection_layers=None):
        super().__init__()
        self.model = model
        self.pool_fn = pool_fn
        self.projection = projection_layers

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def config(self):
        """Expose inner model config (torch.compile / Dynamo and eval scripts expect it)."""
        return self.model.config

    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        pooled = self.pool_fn(output.last_hidden_state, attention_mask)
        if self.projection is not None:
            pooled = self.projection(pooled)
        return F.normalize(pooled, p=2, dim=1)


def add_pooling_layers(model, pool_fn, projection_layers=None):
    """Wrap a model so its forward pass includes pooling and L2 normalization.

    *projection_layers* (optional ``nn.Module``) is inserted between pooling
    and L2-norm — use :func:`load_st_dense_layers` to build one for models
    that ship Dense projections in their SentenceTransformer checkpoint.
    """
    return EncoderWithPooling(model, pool_fn, projection_layers=projection_layers)


def load_st_dense_layers(model_name_or_path: str, dtype=None):
    """Load extra Dense layers from a SentenceTransformer checkpoint.

    Models like ``google/embeddinggemma-300m`` include one or more Dense
    projection modules (``2_Dense``, ``3_Dense``, …) that sit between mean
    pooling and L2-normalisation.  When the base transformer is loaded with
    ``AutoModel.from_pretrained`` these layers are absent — this helper
    restores them.

    Returns ``None`` if the checkpoint has no Dense layers.
    """
    import json
    from pathlib import Path
    from safetensors.torch import load_file as safe_load

    local_dir = Path(model_name_or_path)
    if not local_dir.is_dir():
        from huggingface_hub import snapshot_download

        local_dir = Path(snapshot_download(model_name_or_path, local_files_only=True))

    modules_path = local_dir / "modules.json"
    if not modules_path.exists():
        return None

    with open(modules_path) as f:
        modules = json.load(f)

    dense_modules = [
        m for m in modules if m["type"] == "sentence_transformers.models.Dense"
    ]
    if not dense_modules:
        return None

    layers = []
    for m in dense_modules:
        cfg_path = local_dir / m["path"] / "config.json"
        wt_path = local_dir / m["path"] / "model.safetensors"
        with open(cfg_path) as f:
            cfg = json.load(f)

        linear = nn.Linear(
            cfg["in_features"], cfg["out_features"], bias=cfg.get("bias", True)
        )
        state = safe_load(str(wt_path))
        mapped = {}
        for k, v in state.items():
            new_key = k.removeprefix("linear.")
            mapped[new_key] = v
        linear.load_state_dict(mapped)
        layers.append(linear)

    proj = nn.Sequential(*layers)
    if dtype is not None:
        proj = proj.to(dtype)
    return proj


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
        hidden_states = hidden_states.to(self.up.weight.dtype)
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
        # Expand mask to match hidden_states dimensions, cast to same dtype
        # to avoid float32 promotion when attention_mask is int64
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, L, 1)

        # Compute masked sum
        masked_sum = (hidden_states * mask).sum(dim=1)  # (B, H)

        # Compute mask sum with numerical stability
        mask_sum = mask.sum(dim=1).clamp(min=self.eps)  # (B, 1)

        # Compute mean
        return masked_sum / mask_sum
