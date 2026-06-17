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

import contextlib

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
    T5Gemma2CrossAttentionEncoder,
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
    ProcrustesAlignment,
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
    if cross_attention_layers is not None:
        encoder = T5Gemma2CrossAttentionEncoder(
            encoder_config,
            eoi_token_index=eoi_token_index,
            cross_attention_layers=cross_attention_layers,
        )
    else:
        encoder = T5Gemma2Encoder(
            encoder_config,
            eoi_token_index=eoi_token_index,
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
    attn_implementation: str = "sdpa",
    lora_rank: int = 64,
    cross_attention: bool = False,
    cross_attention_layers: list[int] | None = None,
    lora_cls_attn: bool = False,
    num_pooling_heads: int | None = None,
    gated_attention: bool = False,
    pooling_mode: str = "mean",
    procrustes_alignment: bool = False,
    procrustes_init: str = "identity",
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
    procrustes_kwargs = dict(
        procrustes_alignment=procrustes_alignment, procrustes_init=procrustes_init,
    )

    if lora_cls_attn:
        model = EmbeddingT5Gemma2LoRACLSAttn(
            encoder, rank=lora_rank, **pooling_kwargs, **procrustes_kwargs,
        )
    elif cross_attention:
        model = EmbeddingT5Gemma2CrossAttention(
            encoder, **pooling_kwargs, **procrustes_kwargs,
        )
    elif attention_pooling:
        model = EmbeddingT5Gemma2HiddenPool(
            encoder,
            pooling_mode=pooling_mode,
            **pooling_kwargs,
            **procrustes_kwargs,
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
    """T5Gemma2 encoder with attention pooling over per-layer hidden states.

    Depending on ``pooling_mode`` it extracts per-layer representations and
    pools them via the attention head:

    * ``"mean"`` — mean-pooled token representations per layer, no CLS
      token → ``(num_layers+1)`` vectors.
    * ``"cls"`` — a learnable CLS token is prepended; its hidden state is
      extracted at each layer → ``(num_layers+1)`` vectors.
    * ``"both"`` — CLS token prepended; for each layer, collects both the
      CLS hidden state and the mean-pooled non-CLS tokens →
      ``2*(num_layers+1)`` vectors.

    CLS/mean extraction uses the encoder's ``cls_position`` /
    ``cls_and_mean_pool`` arguments so that full ``(B, L, H)`` tensors are
    never materialised.
    """

    def __init__(
        self,
        encoder: T5Gemma2Encoder,
        num_attention_heads: int | None = None,
        gated_attention: bool = False,
        pooling_mode: str = "mean",
        procrustes_alignment: bool = False,
        procrustes_init: str = "identity",
    ):
        super().__init__()
        assert pooling_mode in ("both", "cls", "mean"), (
            f"pooling_mode must be 'both', 'cls', or 'mean', got '{pooling_mode}'"
        )
        self.pooling_mode = pooling_mode
        self.encoder = encoder
        h = self.encoder.text_config.hidden_size
        if num_attention_heads is None:
            num_attention_heads = self.encoder.text_config.num_attention_heads
        PoolingClass = GatedAttention if gated_attention else AttentionPooling
        self.pooling = PoolingClass(
            hidden_size=h,
            num_attention_heads=num_attention_heads,
        )
        # Learnable CLS query embedding — only used for "cls" and "both" modes
        if pooling_mode in ("cls", "both"):
            self.cls_query = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        # Optional per-layer orthogonal alignment applied before pooling.
        if procrustes_alignment:
            num_layers = self.encoder.text_config.num_hidden_layers
            num_views = (num_layers + 1) * (2 if pooling_mode == "both" else 1)
            self.procrustes = ProcrustesAlignment(
                num_views=num_views, hidden_size=h, init=procrustes_init,
            )
        else:
            self.procrustes = None
        # When True and pooling_mode=="mean", run the encoder under
        # torch.no_grad() — safe because no trainable parameter (e.g.
        # cls_query) flows through the encoder in mean mode.  Set by the
        # training script for stage-1 (frozen encoder) to skip graph
        # construction and save memory/compute.
        self.encoder_no_grad = False
        self.projection = Projection(input_dim=h, hidden_dim=4 * h)
        self.normalize = Normalize()

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def _prepend_cls(self, input_ids, inputs_embeds, attention_mask):
        """Embed tokens, prepend the CLS query, and extend the mask."""
        if inputs_embeds is None:
            inputs_embeds = self.encoder.embed_tokens(input_ids)
        batch_size = inputs_embeds.shape[0]
        cls_expanded = self.cls_query.expand(batch_size, -1, -1).to(
            dtype=inputs_embeds.dtype
        )
        inputs_embeds = torch.cat([cls_expanded, inputs_embeds], dim=1)  # (B, 1+L, H)
        if attention_mask is not None:
            cls_mask = torch.ones(
                batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)  # (B, 1+L)
        return inputs_embeds, attention_mask

    def _encode_views(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """Run the encoder and return the (B, K, D) per-view stack
        consumed by the pooling head — before any Procrustes rotation."""
        if self.pooling_mode == "mean":
            # No CLS token — mean-pool per layer directly.
            # When encoder is frozen (stage 1), skip autograd graph
            # construction since no trainable param flows through it.
            ctx = torch.no_grad() if self.encoder_no_grad else contextlib.nullcontext()
            with ctx:
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=True,
                    **kwargs,
                )
                return torch.stack(outputs.hidden_states, dim=1)  # [B, L+1, D]

        # Prepend CLS token
        inputs_embeds, attention_mask = self._prepend_cls(
            input_ids, inputs_embeds, attention_mask
        )
        if self.pooling_mode == "both":
            outputs = self.encoder(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=None,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                cls_position=0,
                cls_and_mean_pool=True,
                **kwargs,
            )
            return torch.stack(outputs.hidden_states, dim=1)  # (B, 2*(L+1), H)
        # "cls"
        outputs = self.encoder(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            cls_position=0,
            **kwargs,
        )
        return torch.stack(outputs.hidden_states, dim=1)  # (B, L+1, H)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        gpa_loss: bool = False,
        **kwargs,
    ):
        if gpa_loss:
            return self.compute_gpa_loss(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs,
            )
        hidden = self._encode_views(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if self.procrustes is not None:
            hidden = self.procrustes(hidden)
        hidden, _ = self.pooling(hidden)  # [B, D]
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)
        return hidden

    def compute_gpa_loss(self, input_ids=None, attention_mask=None, **kwargs):
        """GPA loss over the per-view stack. Encoder runs under no_grad —
        only the Procrustes matrices receive gradients."""
        if self.procrustes is None:
            raise RuntimeError(
                "compute_gpa_loss requires procrustes_alignment=True at construction"
            )
        with torch.no_grad():
            hidden = self._encode_views(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs,
            )
        return self.procrustes.gpa_loss(hidden)


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
        encoder: T5Gemma2CrossAttentionEncoder,
        num_attention_heads: int | None = None,
        gated_attention: bool = False,
        procrustes_alignment: bool = False,
        procrustes_init: str = "identity",
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
        # Optional orthogonal alignment over the (num_cross_attn+1) CLS states.
        if procrustes_alignment:
            num_views = len(self.encoder.cross_attention_layer_ids) + 1
            self.procrustes = ProcrustesAlignment(
                num_views=num_views, hidden_size=h, init=procrustes_init,
            )
        else:
            self.procrustes = None
        self.projection = Projection(input_dim=h, hidden_dim=4 * h)
        self.normalize = Normalize()

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def _encode_views(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """Run the cross-attention encoder and return the (B, K, D) stack of
        per-block CLS states (pre-Procrustes)."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=False,
            **kwargs,
        )
        # cls_state: (B, num_cross_attn+1, H) — initial + one per block
        return outputs.cls_state

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        gpa_loss: bool = False,
        **kwargs,
    ):
        if gpa_loss:
            return self.compute_gpa_loss(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs,
            )
        hidden = self._encode_views(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if self.procrustes is not None:
            hidden = self.procrustes(hidden)
        hidden, _ = self.pooling(hidden)  # (B, H)
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)
        return hidden

    def compute_gpa_loss(self, input_ids=None, attention_mask=None, **kwargs):
        """GPA loss over the per-block CLS states (encoder under no_grad)."""
        if self.procrustes is None:
            raise RuntimeError(
                "compute_gpa_loss requires procrustes_alignment=True at construction"
            )
        with torch.no_grad():
            hidden = self._encode_views(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs,
            )
        return self.procrustes.gpa_loss(hidden)


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
        procrustes_alignment: bool = False,
        procrustes_init: str = "identity",
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
        # Optional orthogonal alignment over the per-layer CLS intermediates.
        if procrustes_alignment:
            num_views = encoder.text_config.num_hidden_layers
            self.procrustes = ProcrustesAlignment(
                num_views=num_views, hidden_size=h, init=procrustes_init,
            )
        else:
            self.procrustes = None
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
    def _encode_views(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """Forward through the LoRA-wrapped encoder and stack the CLS
        intermediates into a (B, num_layers, H) per-view tensor."""
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

        return torch.stack(cls_intermediates, dim=1)  # (B, num_layers, H)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        gpa_loss: bool = False,
        **kwargs,
    ):
        if gpa_loss:
            return self.compute_gpa_loss(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs,
            )
        hidden = self._encode_views(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if self.procrustes is not None:
            hidden = self.procrustes(hidden)
        hidden, _ = self.pooling(hidden)  # (B, H)
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)
        return hidden

    def compute_gpa_loss(self, input_ids=None, attention_mask=None, **kwargs):
        """GPA loss over the per-layer CLS intermediates (encoder no_grad)."""
        if self.procrustes is None:
            raise RuntimeError(
                "compute_gpa_loss requires procrustes_alignment=True at construction"
            )
        with torch.no_grad():
            hidden = self._encode_views(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs,
            )
        return self.procrustes.gpa_loss(hidden)


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
