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


def gated_attention_forward(
    query_weight: torch.Tensor,
    U_states: torch.Tensor,
    V_states: torch.Tensor,
    hidden_states: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gated attention pooling over a sequence of hidden states.

    Args:
        query_weight: (1, head_dim) learned query vector per head
        U_states: (B, num_heads, seq_len, head_dim) - sigmoid gate
        V_states: (B, num_heads, seq_len, head_dim) - tanh gate
        hidden_states: (B, num_heads, seq_len, head_dim) - values
        scaling: attention scaling factor
        dropout: dropout probability

    Returns:
        attn_output: (B, seq_len, num_heads * head_dim)
        attn_weights: (B, num_heads, 1, seq_len)
    """
    gating_mechanism = (
        torch.tanh(V_states.float()) * torch.sigmoid(U_states.float())
    ).to(hidden_states.dtype)

    # query_weight: (1, head_dim), gating: (B, heads, seq, head_dim)
    # matmul: (B, heads, 1, head_dim) @ (B, heads, head_dim, seq) -> (B, heads, 1, seq)
    attn_weights = (
        torch.matmul(
            query_weight.unsqueeze(0).unsqueeze(0), gating_mechanism.transpose(2, 3)
        )
        * scaling
    )

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        hidden_states.dtype
    )
    attn_weights = nn.functional.dropout(attn_weights, p=dropout)
    # (B, heads, 1, seq) @ (B, heads, seq, head_dim) -> (B, heads, 1, head_dim)
    attn_output = torch.matmul(attn_weights, hidden_states)
    # (B, heads, 1, head_dim) -> (B, 1, heads * head_dim)
    attn_output = attn_output.squeeze(2).transpose(1, 2).contiguous()
    batch_size = attn_output.shape[0]
    attn_output = attn_output.reshape(batch_size, -1)  # (B, heads * head_dim)
    return attn_output, attn_weights


class GatedAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 1,
        head_dim: int = None,
    ):
        """
        Gated Attention mechanism for pooling a sequence of hidden-layer
        representations into a single vector.

        Args:
            hidden_size (int): The dimension of each hidden representation.
            num_attention_heads (int): Number of attention heads (default 1).
            head_dim (int): Per-head dimension.  Defaults to hidden_size // num_attention_heads.
        """
        super().__init__()

        self.num_attention_heads = num_attention_heads
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        self.head_dim = head_dim

        # Gating projections
        self.U = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.V = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)

        # Value projection
        self.W = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)

        # Learned query vector per head: (1, head_dim)
        self.w = nn.Parameter(torch.randn(1, head_dim))

        # Output projection back to hidden_size
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pool a sequence of representations into a single vector via gated attention.

        Args:
            hidden_states: (B, K, D) where K is the number of layer representations.

        Returns:
            pooled: (B, D) - the aggregated representation.
            attn_weights: (B, num_heads, 1, K) - attention weights over layers.
        """
        hidden_states = hidden_states.to(self.U.weight.dtype)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.num_attention_heads, self.head_dim)

        # Project and reshape to (B, num_heads, K, head_dim)
        U_states = self.U(hidden_states).view(hidden_shape).transpose(1, 2)
        V_states = self.V(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.W(hidden_states).view(hidden_shape).transpose(1, 2)

        attn_output, attn_weights = gated_attention_forward(
            query_weight=self.w,
            U_states=U_states,
            V_states=V_states,
            hidden_states=value_states,
            scaling=self.head_dim**-0.5,
            dropout=0.0,
        )

        # Project back to hidden_size
        attn_output = self.o_proj(attn_output)  # (B, D)
        return attn_output, attn_weights


class CLSCrossAttentionAdapter(nn.Module):
    """Low-rank cross-attention adapter that computes a CLS representation
    by attending to frozen encoder hidden states.

    At each encoder layer this module takes:
      - ``cls_query``     : (B, 1, H) — a learnable CLS embedding.
      - ``hidden_states`` : (B, L, H) — frozen hidden states from that layer.
      - ``attention_mask``: (B, L)    — 1 for real tokens, 0 for padding.

    It produces a (B, H) CLS representation for that layer via low-rank
    query/key/value projections (rank ``r`` << H), making it parameter-
    efficient while still allowing the CLS token to selectively gather
    information from the frozen backbone.
    """

    def __init__(self, hidden_size: int, rank: int = 64):
        super().__init__()
        self.rank = rank
        self.q_proj = nn.Linear(hidden_size, rank, bias=False)
        self.k_proj = nn.Linear(hidden_size, rank, bias=False)
        self.v_proj = nn.Linear(hidden_size, rank, bias=False)
        self.o_proj = nn.Linear(rank, hidden_size, bias=False)
        self.scale = rank ** -0.5

        # Initialise output projection near zero so that the adapter starts
        # close to a no-op (residual-friendly).
        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        cls_query: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            cls_query:      (B, 1, H)
            hidden_states:  (B, L, H)
            attention_mask:  (B, L)  — 1/True = attend, 0/False = ignore.

        Returns:
            (B, H) — adapted CLS representation for this layer.
        """
        q = self.q_proj(cls_query)                       # (B, 1, r)
        k = self.k_proj(hidden_states)                   # (B, L, r)
        v = self.v_proj(hidden_states)                   # (B, L, r)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, 1, L)

        if attention_mask is not None:
            attn = attn.masked_fill(
                ~attention_mask.unsqueeze(1).bool(), float("-inf")
            )

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

        out = torch.matmul(attn, v)    # (B, 1, r)
        out = self.o_proj(out)         # (B, 1, H)
        return out.squeeze(1)          # (B, H)


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
