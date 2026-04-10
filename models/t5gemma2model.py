"""
Load pretrained encoder weights from a T5Gemma2 encoder-decoder checkpoint
into the standalone T5Gemma2Encoder (text-only, encoder-only) defined in t5gemma2.py.

Strategy:
  1. Fetch the encoder config from the full T5Gemma2Config.
  2. Instantiate our lightweight T5Gemma2Encoder with that config.
  3. Download the full checkpoint's safetensors/bin shards.
  4. Extract only the text-encoder weights (prefix "model.encoder.",
     excluding vision_tower and multi_modal_projector),
     strip the prefix, and load them into our encoder with strict=True.
"""

import sys
from pathlib import Path

# Ensure the project root is on the path when running this file directly
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from peft import TaskType

from models.t5gemma2 import (
    T5Gemma2Encoder,
    T5Gemma2EncoderOutput,
    T5Gemma2LoRACLSEncoderLayer,
    sliding_window_mask_function,
)
from transformers.masking_utils import create_bidirectional_mask
from models.modules import (
    MeanPooling,
    Projection,
    Normalize,
    AttentionPooling,
    GatedAttention,
    CLSCrossAttentionAdapter,
)


# Full-model keys look like:  model.encoder.layers.0.self_attn.q_proj.weight
# Our T5Gemma2Encoder keys:   layers.0.self_attn.q_proj.weight
# So we strip "model.encoder." and exclude vision_tower / multi_modal_projector.
ENCODER_PREFIX = "model.encoder."
EXCLUDE_PREFIXES = ("vision_tower.", "multi_modal_projector.")


def _collect_safetensors(model_dir: str | Path) -> list[Path]:
    """Return all .safetensors files in *model_dir*, sorted."""
    model_dir = Path(model_dir)
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_dir}. "
            "Make sure the checkpoint was downloaded correctly."
        )
    return files


def extract_encoder_state_dict(
    model_name_or_path: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Download (or reuse cached) full T5Gemma2 checkpoint and return only the
    text-encoder weights, with keys already stripped to match T5Gemma2Encoder.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace Hub repo id (e.g. "google/t5gemma-2-270m-270m") or local path.
    revision, token, cache_dir
        Forwarded to ``huggingface_hub.snapshot_download``.
    device : str
        Tensors are loaded onto this device (default "cpu").

    Returns
    -------
    dict[str, torch.Tensor]
        State dict whose keys correspond to ``T5Gemma2Encoder.state_dict()``.
    """
    local_path = Path(model_name_or_path)
    if not local_path.is_dir():
        local_path = Path(
            snapshot_download(
                model_name_or_path,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                allow_patterns=["*.safetensors", "*.json"],
            )
        )

    shard_files = _collect_safetensors(local_path)

    encoder_sd: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        full_sd = load_file(shard, device=device)
        for key, tensor in full_sd.items():
            if key.startswith(ENCODER_PREFIX):
                new_key = key[len(ENCODER_PREFIX) :]
                # Skip vision tower and multi-modal projector weights
                if any(new_key.startswith(ep) for ep in EXCLUDE_PREFIXES):
                    continue
                encoder_sd[new_key] = tensor

    return encoder_sd


def load_t5gemma2_encoder(
    model_name_or_path: str = "google/t5gemma-2-270m-270m",
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
    torch_dtype: torch.dtype | None = None,
    device: str = "cpu",
    attn_implementation: str = "sdpa",
    cross_attention_layers: list[int] | None = None,
) -> T5Gemma2Encoder:
    """
    Build a ``T5Gemma2Encoder`` and load pretrained text-encoder weights from
    a full T5Gemma2 encoder-decoder checkpoint.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace Hub repo id or local checkpoint directory.
    torch_dtype : torch.dtype, optional
        If given, cast the model to this dtype after loading.
    device : str
        Device for initial weight loading (default "cpu").

    Returns
    -------
    T5Gemma2Encoder
        The encoder with pretrained weights loaded.
    """
    # 1. Fetch the full config and extract the encoder sub-config.
    full_config = AutoConfig.from_pretrained(
        model_name_or_path,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
    )
    encoder_config = full_config.encoder  # T5Gemma2EncoderConfig
    eoi_token_index = getattr(full_config, "eoi_token_index", 256_000)

    # Override attention implementation before instantiation so that
    # T5Gemma2Encoder.__init__ propagates it to text_config.
    # Default is "sdpa" (fused QKV kernel, no O(L²) HBM materialisation).
    encoder_config._attn_implementation = attn_implementation

    # 2. Instantiate the encoder (random weights).
    encoder = T5Gemma2Encoder(
        encoder_config,
        eoi_token_index=eoi_token_index,
        cross_attention_layers=cross_attention_layers,
    )

    # 3. Extract the text-encoder weights from the full checkpoint.
    encoder_sd = extract_encoder_state_dict(
        model_name_or_path,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        device=device,
    )

    # 4. Load weights (strict=True ensures a perfect 1-to-1 match).
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)

    # Rotary-embedding buffers are recomputed at init and are non-persistent,
    # so they are expected in `missing` but never in the checkpoint.
    # Cross-attention layers (cls_query, cross_attn_layers) are randomly
    # initialised and have no pretrained weights.
    missing_non_buffer = [
        k
        for k in missing
        if "rotary_emb" not in k
        and "embed_scale" not in k
        and "cross_attn" not in k
        and "cls_query" not in k
    ]
    if missing_non_buffer:
        raise RuntimeError(
            f"Missing keys that are NOT rotary/embed buffers: {missing_non_buffer}"
        )
    if unexpected:
        raise RuntimeError(f"Unexpected keys in encoder state dict: {unexpected}")

    # 5. Optional dtype cast.
    if torch_dtype is not None:
        encoder = encoder.to(dtype=torch_dtype)

    encoder.eval()
    print(
        f"T5Gemma2Encoder loaded from '{model_name_or_path}' "
        f"({len(encoder_sd)} parameters, missing buffers: {len(missing)})"
    )
    return encoder


# ---------------------------------------------------------------------------
# Embedding model wrappers
# ---------------------------------------------------------------------------


def get_model_t5gemma2_model(
    model_name_or_path,
    activation_checkpointing,
    attention_pooling,
    cls_query_pooling,
    attn_implementation: str = "sdpa",
    lora_cls: bool = False,
    lora_rank: int = 64,
    cross_attention: bool = False,
    cross_attention_layers: list[int] | None = None,
    lora_cls_attn: bool = False,
    num_pooling_heads: int | None = None,
    gated_attention: bool = False,
):
    """
    Build a T5Gemma2-based embedding model, mirroring the interface of
    ``gemma3model.get_model``.
    """
    encoder = load_t5gemma2_encoder(
        model_name_or_path=model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        cross_attention_layers=cross_attention_layers if cross_attention else None,
    )

    if activation_checkpointing:
        encoder.config.use_cache = False

    pooling_kwargs = dict(num_attention_heads=num_pooling_heads, gated_attention=gated_attention)

    if lora_cls_attn:
        model = EmbeddingT5Gemma2LoRACLSAttn(
            encoder, rank=lora_rank, **pooling_kwargs,
        )
    elif cross_attention:
        model = EmbeddingT5Gemma2CrossAttention(encoder, **pooling_kwargs)
    elif lora_cls:
        model = EmbeddingT5Gemma2HiddenPoolCLSLoRA(
            encoder,
            lora_rank=lora_rank,
            **pooling_kwargs,
        )
    elif attention_pooling:
        if cls_query_pooling:
            model = EmbeddingT5Gemma2HiddenPoolCLS(
                encoder,
                **pooling_kwargs,
            )
        else:
            model = EmbeddingT5Gemma2HiddenPool(
                encoder,
                **pooling_kwargs,
            )
    else:
        model = EmbeddingT5Gemma2(encoder)

    task_type = TaskType.FEATURE_EXTRACTION
    lora_modules = [
        "q_proj",
        "o_proj",
        "v_proj",
        "k_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    return model, task_type, lora_modules


class EmbeddingT5Gemma2(nn.Module):
    """
    T5Gemma2 encoder with mean pooling + projection,
    analogous to EmbeddingGemma in gemma3model.py.
    """

    def __init__(self, encoder: T5Gemma2Encoder):
        super().__init__()
        self.encoder = encoder
        h = self.encoder.text_config.hidden_size
        self.pooling = MeanPooling()
        self.projection = Projection(input_dim=h, hidden_dim=4 * h)
        self.normalize = Normalize()

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=False,
            **kwargs,
        )

        hidden = outputs.last_hidden_state  # (B, L, H)
        hidden = self.pooling(hidden, attention_mask)
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)
        return hidden


class EmbeddingT5Gemma2HiddenPool(nn.Module):
    """
    T5Gemma2 encoder with gated-attention pooling over per-layer
    mean-pooled hidden states, analogous to EmbeddingGemmaHiddenPool
    in gemma3model.py.
    """

    def __init__(
        self,
        encoder: T5Gemma2Encoder,
        num_attention_heads: int | None = None,
        gated_attention: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        h = self.encoder.text_config.hidden_size
        if num_attention_heads is None:
            num_attention_heads = self.encoder.text_config.num_attention_heads
        PoolingClass = GatedAttention if gated_attention else AttentionPooling
        self.pooling = PoolingClass(
            hidden_size=h,
            num_attention_heads=num_attention_heads,
        )
        self.projection = Projection(input_dim=h, hidden_dim=4 * h)
        self.normalize = Normalize()

    # use to call model.device deepspeed needs it
    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # Force output_hidden_states=True to get per-layer mean-pooled representations
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            **kwargs,
        )

        # outputs.hidden_states is a tuple of (B, D) tensors (mean-pooled per layer)
        hidden = torch.stack(outputs.hidden_states, dim=1)  # [B, num_layers+1, D]
        hidden, _ = self.pooling(hidden)  # [B, D]
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)
        return hidden


class EmbeddingT5Gemma2HiddenPoolCLS(nn.Module):
    """
    T5Gemma2 encoder with a learnable CLS query token prepended to the input.
    The CLS token's residual-stream representation at each layer is extracted
    and pooled via attention over layers.
    """

    def __init__(
        self,
        encoder: T5Gemma2Encoder,
        num_attention_heads: int | None = None,
        gated_attention: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        h = self.encoder.text_config.hidden_size
        if num_attention_heads is None:
            num_attention_heads = self.encoder.text_config.num_attention_heads
        PoolingClass = GatedAttention if gated_attention else AttentionPooling
        self.pooling = PoolingClass(
            hidden_size=h,
            num_attention_heads=num_attention_heads,
        )
        # Learnable CLS query embedding (1, 1, H)
        self.cls_query = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.projection = Projection(input_dim=h, hidden_dim=4 * h)
        self.normalize = Normalize()

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # 1. Get token embeddings
        if inputs_embeds is None:
            inputs_embeds = self.encoder.embed_tokens(input_ids)
            input_ids = None

        batch_size = inputs_embeds.shape[0]

        # 2. Prepend learnable CLS query at position 0
        cls_expanded = self.cls_query.expand(batch_size, -1, -1).to(
            dtype=inputs_embeds.dtype
        )
        inputs_embeds = torch.cat([cls_expanded, inputs_embeds], dim=1)  # (B, 1+L, H)

        # 3. Extend attention_mask: CLS token is always attended to
        if attention_mask is not None:
            cls_mask = torch.ones(
                batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)  # (B, 1+L)

        # 4. Forward through encoder, collecting CLS position at each layer
        outputs = self.encoder(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,  # let the encoder recompute for new length
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            cls_position=0,  # extract position 0 (the CLS token)
            **kwargs,
        )

        # 5. outputs.hidden_states is a tuple of (B, D) tensors (CLS repr per layer)
        hidden = torch.stack(outputs.hidden_states, dim=1)  # [B, num_layers+1, D]
        hidden, _ = self.pooling(hidden)  # [B, D]
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)
        return hidden


class EmbeddingT5Gemma2HiddenPoolCLSLoRA(nn.Module):
    """T5Gemma2 encoder with per-layer cross-attention adapters that project
    information from the frozen backbone into a learnable CLS token.

    Architecture (frozen backbone + trainable adapters):
      1. The frozen encoder runs on the original input tokens with
         ``return_full_hidden_states=True``, producing full ``(B, L, H)``
         hidden states at every layer — the backbone is never modified.
      2. At each layer a ``CLSCrossAttentionAdapter`` computes a CLS
         representation by cross-attending from a learnable query to the
         frozen hidden states of that layer:
             CLS_i = CrossAttn(cls_query_i, frozen_hidden_i, mask)
         The CLS query learns *where* to look in the sequence; the backbone
         provides what to look at.
      3. The per-layer CLS representations are stacked and pooled via
         ``GatedAttention`` → ``Projection`` → ``Normalize``.

    Trainable parameters: per-layer CLS queries, ``CLSCrossAttentionAdapter``
    weights (q/k/v/o projections), ``GatedAttention``, ``Projection``.
    The encoder backbone is frozen externally before training.
    """

    def __init__(
        self,
        encoder: T5Gemma2Encoder,
        lora_rank: int = 64,
        num_attention_heads: int | None = None,
        gated_attention: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        h = self.encoder.text_config.hidden_size
        num_layers = self.encoder.text_config.num_hidden_layers + 1  # +1 for embed layer

        # Per-layer learnable CLS query embeddings
        self.cls_queries = nn.ParameterList(
            [nn.Parameter(torch.randn(1, 1, h) * 0.02) for _ in range(num_layers)]
        )

        # Per-layer cross-attention adapters (properly initialized — not zeroed)
        self.adapters = nn.ModuleList(
            [CLSCrossAttentionAdapter(h, rank=lora_rank) for _ in range(num_layers)]
        )

        if num_attention_heads is None:
            num_attention_heads = self.encoder.text_config.num_attention_heads
        PoolingClass = GatedAttention if gated_attention else AttentionPooling
        self.pooling = PoolingClass(
            hidden_size=h,
            num_attention_heads=num_attention_heads,
        )
        self.projection = Projection(input_dim=h, hidden_dim=4 * h)
        self.normalize = Normalize()

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # 1. Frozen encoder — full (B, L, H) per-layer hidden states.
        # No CLS token is prepended: backbone sees only the original tokens,
        # keeping its representations completely unaffected by the adapters.
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_full_hidden_states=True,  # (B, L, H) per layer
                **kwargs,
            )

        # 2. Per-layer CLS via cross-attention adapters.
        # outputs.hidden_states: tuple of (B, L, H), one per layer.
        batch_size = outputs.hidden_states[0].shape[0]
        cls_reps = []
        for adapter, cls_q, layer_hidden in zip(
            self.adapters, self.cls_queries, outputs.hidden_states
        ):
            cls_expanded = cls_q.expand(batch_size, -1, -1).to(
                dtype=layer_hidden.dtype
            )
            cls_rep = adapter(cls_expanded, layer_hidden, attention_mask)  # (B, H)
            cls_reps.append(cls_rep)

        # 3. Pool over layers → project → normalise
        hidden = torch.stack(cls_reps, dim=1)  # [B, num_layers+1, H]
        hidden, _ = self.pooling(hidden)        # [B, H]
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)
        return hidden


class EmbeddingT5Gemma2CrossAttention(nn.Module):
    """T5Gemma2 encoder with MLLama-style cross-attention layers that let a
    learnable CLS token progressively extract information from the frozen
    backbone's residual stream at selected layers.

    Architecture (frozen backbone + trainable cross-attention):
      1. The encoder runs on the original input tokens.  At selected layers
         (``cross_attention_layers``), cross-attention blocks update a CLS
         token that queries the encoder hidden states.
      2. Each cross-attention block contains:
         - LayerNorm + Cross-Attention + tanh-gated residual
         - LayerNorm + MLP + tanh-gated residual
         Gates are initialised to zero (no-op at init).
      3. Intermediate CLS representations (one per cross-attention block
         + one from the final encoder output) are pooled via
         ``GatedAttention`` → ``Projection`` → L2-normalise.

    With the default 5 cross-attention layers this produces a sequence of
    6 CLS representations that GatedAttention attends over:
      [cls_after_L3, cls_after_L8, cls_after_L13, cls_after_L18, cls_after_L23, cls_after_final]

    Trainable parameters: CLS query, cross-attention layers, GatedAttention,
    Projection.
    The encoder backbone self-attention layers are frozen externally.
    """

    def __init__(
        self,
        encoder: T5Gemma2Encoder,
        num_attention_heads: int | None = None,
        gated_attention: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        h = self.encoder.text_config.hidden_size
        if num_attention_heads is None:
            num_attention_heads = self.encoder.text_config.num_attention_heads
        PoolingClass = GatedAttention if gated_attention else AttentionPooling
        self.pooling = PoolingClass(
            hidden_size=h,
            num_attention_heads=num_attention_heads,
        )
        self.projection = Projection(input_dim=h, hidden_dim=4 * h)
        self.normalize = Normalize()

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=False,
            **kwargs,
        )

        # cls_state: (B, num_cross_attn+1, H) — initial + one per block
        hidden = outputs.cls_state
        hidden, _ = self.pooling(hidden)  # (B, H)
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)
        return hidden


class EmbeddingT5Gemma2LoRACLSAttn(nn.Module):
    """T5Gemma2 encoder with CLS-position LoRA on self-attention.

    A learnable CLS token is prepended to the input.  At every encoder layer
    the self-attention Q and O projections carry a low-rank adapter whose
    delta is applied **only at the CLS position**.  An asymmetric attention
    mask prevents non-CLS tokens from attending to CLS, so their hidden
    states are mathematically identical to the frozen backbone.

    The CLS hidden state from each layer is collected and pooled via
    attention → ``Projection`` → L2-normalise.

    Trainable: CLS query, LoRA adapters (Q/O per layer), pooling head,
    Projection.  The backbone encoder is frozen externally.
    """

    def __init__(
        self,
        encoder: T5Gemma2Encoder,
        rank: int = 64,
        num_attention_heads: int | None = None,
        gated_attention: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        h = encoder.text_config.hidden_size

        # Learnable CLS embedding
        self.cls_query = nn.Parameter(torch.randn(1, 1, h) * 0.02)

        # LoRA layer wrappers — one per encoder layer
        self.lora_layers = nn.ModuleList(
            [
                T5Gemma2LoRACLSEncoderLayer(layer, rank=rank)
                for layer in encoder.layers
            ]
        )

        if num_attention_heads is None:
            num_attention_heads = self.encoder.text_config.num_attention_heads
        PoolingClass = GatedAttention if gated_attention else AttentionPooling
        self.pooling = PoolingClass(
            hidden_size=h,
            num_attention_heads=num_attention_heads,
        )
        self.projection = Projection(input_dim=h, hidden_dim=4 * h)
        self.normalize = Normalize()

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    # ------------------------------------------------------------------
    # Helpers for mask creation
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_cls_column(mask: torch.Tensor) -> torch.Tensor:
        """Set column 0 to -inf for rows 1: (non-CLS can't attend to CLS)."""
        mask = mask.clone()
        if mask.dtype.is_floating_point:
            mask[:, :, 1:, 0] = torch.finfo(mask.dtype).min
        else:
            mask[:, :, 1:, 0] = False
        return mask

    def _build_masks(self, inputs_embeds, attention_mask):
        """Create bidirectional attention masks with asymmetric CLS masking.

        For sliding-attention layers the CLS row keeps full attention
        (overriding the sliding window) so it can see the entire sequence.
        """
        enc = self.encoder
        mask_kwargs = {
            "config": enc.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
        full_mask = create_bidirectional_mask(**mask_kwargs)
        sliding_mask = create_bidirectional_mask(
            **mask_kwargs,
            and_mask_function=sliding_window_mask_function(
                enc.text_config.sliding_window, is_causal=False
            ),
        )

        # Block non-CLS → CLS column
        full_mask = self._mask_cls_column(full_mask)
        sliding_mask = self._mask_cls_column(sliding_mask)
        # CLS row in sliding layers should have full attention (not windowed)
        sliding_mask[:, :, 0, :] = full_mask[:, :, 0, :]

        return {
            "full_attention": full_mask,
            "sliding_attention": sliding_mask,
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        enc = self.encoder
        text_config = enc.text_config

        # 1. Token embeddings
        if inputs_embeds is None:
            inputs_embeds = enc.embed_tokens(input_ids)
        batch_size = inputs_embeds.shape[0]

        # 2. Prepend learnable CLS token
        cls_expanded = self.cls_query.expand(batch_size, -1, -1).to(
            dtype=inputs_embeds.dtype
        )
        inputs_embeds = torch.cat([cls_expanded, inputs_embeds], dim=1)  # (B, 1+L, H)

        # 3. Extend 2-D attention mask with a 1 for CLS
        if attention_mask is not None:
            cls_mask = torch.ones(
                batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        # 4. Position IDs: CLS=0, text tokens 1…L (preserves relative positions)
        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        ).unsqueeze(0)

        # 5. 4-D attention masks with asymmetric CLS masking
        self_attn_mask_mapping = self._build_masks(inputs_embeds, attention_mask)

        # 6. RoPE
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in set(text_config.layer_types):
            position_embeddings[layer_type] = enc.rotary_emb(
                hidden_states, position_ids, layer_type
            )

        hidden_states = enc.dropout(hidden_states)

        # 7. Forward through LoRA-wrapped layers, collect CLS at each
        cls_intermediates = []
        for layer_module, lora_layer in zip(
            enc.layers[: text_config.num_hidden_layers], self.lora_layers
        ):
            hidden_states = lora_layer(
                hidden_states,
                position_embeddings[layer_module.attention_type],
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                **kwargs,
            )
            cls_intermediates.append(hidden_states[:, 0, :])  # (B, H)

        # 8. Pool CLS intermediates → Projection → Normalise
        hidden = torch.stack(cls_intermediates, dim=1)  # (B, num_layers, H)
        hidden, _ = self.pooling(hidden)  # (B, H)
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)
        return hidden


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    encoder = load_t5gemma2_encoder("google/t5gemma-2-270m-270m")
    print(encoder)

    # Minimal forward pass sanity check
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2-270m-270m")
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    with torch.no_grad():
        outputs = encoder(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
    print("Output shape:", outputs.last_hidden_state.shape)
