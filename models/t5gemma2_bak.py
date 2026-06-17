"""Cross-attention and LoRA-CLS variants of the T5Gemma2 encoder.

Lifted out of ``models/t5gemma2.py`` to keep the main module focused on the
mean / CLS / both attention-pooling path. Two architectures live here:

* :class:`T5Gemma2CrossAttentionEncoder` — MLLama-style cross-attention
  blocks at selected encoder layers update a learnable CLS query that
  reads from the frozen backbone's residual stream.
* :class:`T5Gemma2LoRACLSEncoderLayer` — wraps a frozen
  :class:`models.t5gemma2.T5Gemma2EncoderLayer` with CLS-position LoRA on
  the Q/O projections of self-attention.

These are re-importable but no longer wired into
``models/t5gemma2model.py`` or ``train_f2llm_repro.py``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.models.t5gemma2.configuration_t5gemma2 import (
    T5Gemma2EncoderConfig,
    T5Gemma2TextConfig,
)

from models.t5gemma2 import (
    T5Gemma2Encoder,
    T5Gemma2EncoderLayer,
    T5Gemma2MLP,
    T5Gemma2RMSNorm,
    T5Gemma2SelfAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    mean_pool,
    repeat_kv,
)


@dataclass
class T5Gemma2CrossAttentionEncoderOutput(BaseModelOutput):
    """BaseModelOutput extended with the per-block CLS state stack."""

    cls_state: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# Cross-attention: a learnable CLS query reads from the backbone residual
# stream at selected layers via tanh-gated blocks (MLLama pattern).
# ---------------------------------------------------------------------------


class T5Gemma2CrossAttention(nn.Module):
    """Cross-attention module: Q from CLS token, K/V from encoder hidden states.

    Follows the MLLama MllamaTextCrossAttention pattern with Q/K RMSNorm,
    GQA support, and optional logit softcapping.
    """

    def __init__(self, config: T5Gemma2TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = T5Gemma2RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = T5Gemma2RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.attn_logit_softcapping = config.attn_logit_softcapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: CLS state (B, 1, H) — the query source.
            encoder_hidden_states: Encoder output (B, L, H) — key/value source.
            attention_mask: (B, L) padding mask (1 = real, 0 = pad).
        """
        bsz, q_len, _ = hidden_states.shape
        _, kv_len, _ = encoder_hidden_states.shape

        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(encoder_hidden_states)
            .view(bsz, kv_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(encoder_hidden_states)
            .view(bsz, kv_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # GQA: repeat KV heads to match query heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # (B, heads, 1, head_dim) @ (B, heads, head_dim, L) -> (B, heads, 1, L)
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )

        if self.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping

        # Mask padding positions
        if attention_mask is not None:
            # (B, L) -> (B, 1, 1, L)
            mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)  # (B, heads, 1, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


class T5Gemma2CrossAttentionEncoderLayer(GradientCheckpointingLayer):
    """Cross-attention encoder layer with tanh-gated residual connections.

    Following the MLLama ``MllamaCrossAttentionDecoderLayer`` pattern:
      residual + tanh(gate) * cross_attn(...)
      residual + tanh(gate) * mlp(...)

    Gates are initialised to zero so the layer is a no-op at init,
    preserving the pretrained backbone behaviour until training opens them.
    """

    def __init__(self, config: T5Gemma2TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.cross_attn = T5Gemma2CrossAttention(config=config, layer_idx=layer_idx)
        self.input_layernorm = T5Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.cross_attn_attn_gate = nn.Parameter(torch.zeros(1))

        self.mlp = T5Gemma2MLP(config)
        self.post_attention_layernorm = T5Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.cross_attn_mlp_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: CLS state (B, 1, H).
            encoder_hidden_states: Backbone output at this layer (B, L, H).
            attention_mask: (B, L) padding mask.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# LoRA CLS-in-Attention: CLS token participates in self-attention with
# LoRA adapters masked to position 0.  Non-CLS representations stay frozen.
# ---------------------------------------------------------------------------


class LoRACLSProjection(nn.Module):
    """Low-rank adapter that computes a delta for position 0 (CLS) only.

    ``up`` is zero-initialised so the adapter is a no-op at init (LoRA-style).
    """

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return LoRA delta for position 0.  (B, 1, out_features)"""
        return self.up(self.down(x[:, 0:1, :]))


class T5Gemma2LoRACLSSelfAttention(nn.Module):
    """Wraps a frozen ``T5Gemma2SelfAttention`` with CLS-position LoRA on Q/O.

    Only position 0 (CLS) receives the LoRA delta.  Because K and V are
    untouched and only the CLS query changes, non-CLS rows of the attention
    matrix — and therefore non-CLS hidden states — are mathematically
    identical to the frozen backbone.
    """

    def __init__(self, frozen_attn: "T5Gemma2SelfAttention", rank: int = 64):
        super().__init__()
        self.attn = frozen_attn  # reference — frozen externally
        config = frozen_attn.config
        h = config.hidden_size
        qo_dim = config.num_attention_heads * frozen_attn.head_dim
        self.lora_q = LoRACLSProjection(h, qo_dim, rank)
        self.lora_o = LoRACLSProjection(qo_dim, h, rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn = self.attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        # Q with LoRA at CLS position
        query_states = attn.q_proj(hidden_states)
        lora_q_delta = self.lora_q(hidden_states)  # (B, 1, qo_dim)
        query_states = torch.cat(
            [query_states[:, 0:1, :] + lora_q_delta, query_states[:, 1:, :]], dim=1
        )
        query_states = query_states.view(hidden_shape).transpose(1, 2)

        # K, V — frozen (no LoRA)
        key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = attn.q_norm(query_states)
        key_states = attn.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        attention_interface: Callable = eager_attention_forward
        if attn.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                attn.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            attn,  # pass frozen module for property access (head_dim, etc.)
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=attn.attention_dropout if attn.training else 0.0,
            scaling=attn.scaling,
            sliding_window=attn.sliding_window,
            **kwargs,
        )

        # O with LoRA at CLS position
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        o_output = attn.o_proj(attn_output)
        lora_o_delta = self.lora_o(attn_output)  # (B, 1, H)
        o_output = torch.cat(
            [o_output[:, 0:1, :] + lora_o_delta, o_output[:, 1:, :]], dim=1
        )
        return o_output, attn_weights


class T5Gemma2LoRACLSEncoderLayer(GradientCheckpointingLayer):
    """Wraps a frozen ``T5Gemma2EncoderLayer`` with CLS-position LoRA.

    Reuses all frozen layer components (layernorms, MLP, dropout) directly.
    Only the self-attention is replaced with the LoRA variant.
    """

    def __init__(self, frozen_layer: "T5Gemma2EncoderLayer", rank: int = 64):
        super().__init__()
        self.frozen_layer = frozen_layer
        self.lora_self_attn = T5Gemma2LoRACLSSelfAttention(
            frozen_layer.self_attn, rank
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        fl = self.frozen_layer

        residual = hidden_states
        hidden_states = fl.pre_self_attn_layernorm(hidden_states)
        hidden_states, _ = self.lora_self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = fl.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + fl.dropout(hidden_states)

        residual = hidden_states
        hidden_states = fl.pre_feedforward_layernorm(hidden_states)
        # Non-CLS tokens never depend on trainable params (asymmetric mask
        # blocks non-CLS → CLS attention), so detach them before the MLP to
        # avoid storing large intermediate activations for the full sequence.
        cls_h = fl.mlp(hidden_states[:, 0:1, :])             # (B, 1, H) — grad
        with torch.no_grad():
            rest_h = fl.mlp(hidden_states[:, 1:, :])         # (B, L, H) — no grad
        hidden_states = torch.cat([cls_h, rest_h], dim=1)
        hidden_states = fl.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + fl.dropout(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Cross-attention encoder: T5Gemma2Encoder with cross-attention blocks at
# selected layers and a final block after the last encoder layer.
# ---------------------------------------------------------------------------


class T5Gemma2CrossAttentionEncoder(T5Gemma2Encoder):
    """T5Gemma2 encoder with MLLama-style cross-attention layers.

    At selected encoder layers a learnable CLS token queries the encoder
    hidden states via cross-attention blocks.  The intermediate CLS
    representations are collected and returned in ``cls_state``.
    """

    def __init__(
        self,
        config: T5Gemma2EncoderConfig,
        eoi_token_index: int = 256_000,
        cross_attention_layers: list[int] | None = None,
    ):
        super().__init__(config, eoi_token_index=eoi_token_index)
        text_config = config.text_config

        if cross_attention_layers is None:
            raise ValueError(
                "T5Gemma2CrossAttentionEncoder requires cross_attention_layers"
            )
        self.cross_attention_layer_ids = cross_attention_layers
        self._cross_attention_layer_set = set(cross_attention_layers)
        self.cls_query = nn.Parameter(
            torch.randn(1, 1, text_config.hidden_size) * 0.02
        )
        self.cross_attn_layers = nn.ModuleList(
            [
                T5Gemma2CrossAttentionEncoderLayer(text_config, layer_idx=idx)
                for idx in cross_attention_layers
            ]
        )
        self.cross_attn_final = T5Gemma2CrossAttentionEncoderLayer(
            text_config, layer_idx=text_config.num_hidden_layers
        )

        # Re-run post_init to initialise cross-attention weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> T5Gemma2CrossAttentionEncoderOutput:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )

        hidden_states, position_ids, position_embeddings, self_attn_mask_mapping, raw_attention_mask = (
            self._prepare_forward(input_ids, attention_mask, position_ids, inputs_embeds, kwargs)
        )

        # Initialise CLS state for cross-attention
        cls_state = self.cls_query.expand(
            hidden_states.shape[0], -1, -1
        ).to(dtype=hidden_states.dtype)
        cls_intermediates = []
        cross_attn_idx = 0

        all_hidden_states = () if output_hidden_states else None

        for layer_idx, layer_module in enumerate(
            self.layers[: self.text_config.num_hidden_layers]
        ):
            if output_hidden_states:
                all_hidden_states += (mean_pool(hidden_states, raw_attention_mask),)

            hidden_states = layer_module(
                hidden_states,
                position_embeddings[layer_module.attention_type],
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                **kwargs,
            )

            if layer_idx in self._cross_attention_layer_set:
                cls_state = self.cross_attn_layers[cross_attn_idx](
                    cls_state, hidden_states, raw_attention_mask
                )
                cross_attn_idx += 1
                cls_intermediates.append(cls_state.squeeze(1))  # (B, H)

        if output_hidden_states:
            all_hidden_states += (mean_pool(hidden_states, raw_attention_mask),)

        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Cross-attend to the final (normed) encoder output
        cls_state = self.cross_attn_final(
            cls_state, hidden_states, raw_attention_mask
        )
        cls_intermediates.append(cls_state.squeeze(1))  # (B, H)
        cls_state = torch.stack(cls_intermediates, dim=1)  # (B, num_cross_attn+1, H)

        return T5Gemma2CrossAttentionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            cls_state=cls_state,
        )


__all__ = [
    "T5Gemma2CrossAttention",
    "T5Gemma2CrossAttentionEncoder",
    "T5Gemma2CrossAttentionEncoderLayer",
    "T5Gemma2CrossAttentionEncoderOutput",
    "T5Gemma2LoRACLSEncoderLayer",
    "T5Gemma2LoRACLSSelfAttention",
    "LoRACLSProjection",
]
