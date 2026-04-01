"""
T5Gemma2 decoder adapted as a standalone causal embedding model.

Strategy:
  1. Fetch the decoder config (T5Gemma2DecoderConfig) from the full T5Gemma2Config.
  2. Instantiate T5Gemma2CausalDecoder which reuses T5Gemma2EncoderLayer
     (weight-compatible: identical Q/K/V/O projection shapes) with causal masks.
  3. Load the decoder weights (prefix "model.decoder.") from the full checkpoint.
  4. Wrap with EOS-token pooling (like Qwen3-Embedding) + Projection + Normalize.

The decoder weights from T5Gemma2MergedAttention (merged self+cross attention)
load directly into T5Gemma2SelfAttention because both use the same q/k/v/o_proj
and q/k_norm parameter names and shapes.  Cross-attention was an *additive* path
(K/V concatenation) so removing it does not alter the self-attention weights.
"""

import sys
from pathlib import Path
from typing import Optional

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from peft import TaskType

from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2DecoderConfig

from models.t5gemma2 import (
    T5Gemma2EncoderLayer,
    T5Gemma2TextScaledWordEmbedding,
    T5Gemma2RMSNorm,
    T5Gemma2RotaryEmbedding,
    T5Gemma2PreTrainedModel,
)
from models.modules import (
    Projection,
    Normalize,
    GatedAttention,
    last_token_pool,
)
from models.t5gemma2model import _collect_safetensors


DECODER_PREFIX = "model.decoder."


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def extract_decoder_state_dict(
    model_name_or_path: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Download (or reuse cached) full T5Gemma2 checkpoint and return only the
    decoder weights, with keys stripped of the ``model.decoder.`` prefix."""
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

    decoder_sd: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        full_sd = load_file(shard, device=device)
        for key, tensor in full_sd.items():
            if key.startswith(DECODER_PREFIX):
                decoder_sd[key[len(DECODER_PREFIX) :]] = tensor

    return decoder_sd


def load_t5gemma2_decoder(
    model_name_or_path: str = "google/t5gemma-2-270m-270m",
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
    torch_dtype: torch.dtype | None = None,
    device: str = "cpu",
    attn_implementation: str = "sdpa",
) -> "T5Gemma2CausalDecoder":
    """Build a ``T5Gemma2CausalDecoder`` and load pretrained decoder weights."""
    full_config = AutoConfig.from_pretrained(
        model_name_or_path,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
    )
    decoder_config = full_config.decoder  # T5Gemma2DecoderConfig
    eoi_token_index = getattr(full_config, "eoi_token_index", 256_000)

    decoder_config._attn_implementation = attn_implementation

    decoder = T5Gemma2CausalDecoder(decoder_config, eoi_token_index=eoi_token_index)

    decoder_sd = extract_decoder_state_dict(
        model_name_or_path,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        device=device,
    )

    missing, unexpected = decoder.load_state_dict(decoder_sd, strict=False)

    missing_non_buffer = [
        k for k in missing if "rotary_emb" not in k and "embed_scale" not in k
    ]
    if missing_non_buffer:
        raise RuntimeError(
            f"Missing keys that are NOT rotary/embed buffers: {missing_non_buffer}"
        )
    if unexpected:
        raise RuntimeError(f"Unexpected keys in decoder state dict: {unexpected}")

    if torch_dtype is not None:
        decoder = decoder.to(dtype=torch_dtype)

    decoder.eval()
    print(
        f"T5Gemma2CausalDecoder loaded from '{model_name_or_path}' "
        f"({len(decoder_sd)} parameters, missing buffers: {len(missing)})"
    )
    return decoder


# ---------------------------------------------------------------------------
# Standalone causal decoder
# ---------------------------------------------------------------------------


class T5Gemma2CausalDecoder(T5Gemma2PreTrainedModel):
    """Standalone causal decoder extracted from a T5Gemma2 encoder-decoder model.

    Architecturally identical to the T5Gemma2 text encoder stack
    (``T5Gemma2EncoderLayer`` with ``T5Gemma2SelfAttention``) but uses:

    * **Causal** attention masks (full-attention and sliding-window variants)
      instead of bidirectional masks.
    * **EOS-token extraction** for ``output_hidden_states`` (instead of
      per-layer mean pooling), making it directly suitable for EOS-pooled
      contrastive learning.

    Decoder weights from ``T5Gemma2MergedAttention`` load without modification
    because the Q/K/V/O projections share the same names and shapes.
    """

    def __init__(
        self,
        config: T5Gemma2DecoderConfig,
        eoi_token_index: int = 256_000,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = T5Gemma2TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
            eoi_token_index=eoi_token_index,
        )
        self.norm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList(
            [
                T5Gemma2EncoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.dropout = nn.Dropout(getattr(config, "dropout_rate", 0.0))
        self.rotary_emb = T5Gemma2RotaryEmbedding(config)

        self.text_config = config

        if not getattr(config, "_attn_implementation", None):
            config._attn_implementation = "sdpa"

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )

        kwargs.pop("past_key_values", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                0, inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)

        # ---- Causal masks (key difference from the bidirectional encoder) ----
        if not isinstance(self_attn_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }
            self_attn_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds

        position_embeddings = {}
        for layer_type in self.text_config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(
                hidden_states, position_ids, layer_type
            )

        hidden_states = self.dropout(hidden_states)

        # Resolve the raw 2-D attention_mask for EOS-token extraction
        raw_attention_mask = (
            attention_mask if not isinstance(attention_mask, dict) else None
        )
        if raw_attention_mask is None:
            raw_attention_mask = torch.ones(
                hidden_states.shape[:2],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        all_hidden_states = () if output_hidden_states else None

        for layer_module in self.layers[: self.text_config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (
                    last_token_pool(hidden_states, raw_attention_mask),
                )

            hidden_states = layer_module(
                hidden_states,
                position_embeddings[layer_module.attention_type],
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                **kwargs,
            )

        if output_hidden_states:
            all_hidden_states += (
                last_token_pool(hidden_states, raw_attention_mask),
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


# ---------------------------------------------------------------------------
# Embedding model wrappers
# ---------------------------------------------------------------------------


class EmbeddingT5Gemma2Decoder(nn.Module):
    """T5Gemma2 decoder with EOS-token pooling + projection + L2 normalize.

    Analogous to ``EmbeddingT5Gemma2`` (encoder, mean pooling) but uses
    ``last_token_pool`` to extract the EOS/last-token representation,
    matching the Qwen3-Embedding approach.
    """

    def __init__(self, decoder: T5Gemma2CausalDecoder):
        super().__init__()
        self.encoder = decoder  # named 'encoder' for activation-checkpointing compat
        h = self.encoder.text_config.hidden_size
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
        hidden = last_token_pool(hidden, attention_mask)  # (B, H)
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)
        return hidden


class EmbeddingT5Gemma2DecoderHiddenPool(nn.Module):
    """T5Gemma2 decoder with gated-attention pooling over per-layer
    EOS-token representations.

    At each layer the EOS-token hidden state is extracted (via
    ``last_token_pool``).  The resulting ``(B, num_layers+1, D)`` stack is
    then pooled with ``GatedAttention`` → ``Projection`` → ``Normalize``.
    """

    def __init__(
        self,
        decoder: T5Gemma2CausalDecoder,
        attention_dim: int = None,
    ):
        super().__init__()
        self.encoder = decoder
        h = self.encoder.text_config.hidden_size
        self.pooling = GatedAttention(
            hidden_size=h,
            num_attention_heads=1,
            head_dim=attention_dim,
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
            output_hidden_states=True,
            **kwargs,
        )

        # outputs.hidden_states: tuple of (B, D) tensors (EOS repr per layer)
        hidden = torch.stack(outputs.hidden_states, dim=1)  # [B, num_layers+1, D]
        hidden, _ = self.pooling(hidden)  # [B, D]
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)
        return hidden


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_model_t5gemma2_decoder(
    model_name_or_path,
    activation_checkpointing,
    attention_pooling,
    attention_dim,
    attn_implementation: str = "sdpa",
):
    """Build a T5Gemma2 decoder-based embedding model.

    Mirrors the interface of ``get_model_t5gemma2_model`` (encoder variant).
    """
    decoder = load_t5gemma2_decoder(
        model_name_or_path=model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )

    if activation_checkpointing:
        decoder.config.use_cache = False

    if attention_pooling:
        model = EmbeddingT5Gemma2DecoderHiddenPool(
            decoder,
            attention_dim=attention_dim,
        )
    else:
        model = EmbeddingT5Gemma2Decoder(decoder)

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
