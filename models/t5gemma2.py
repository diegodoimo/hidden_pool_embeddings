# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Stripped-down version: encoder-only, text-only embedding model.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.integrations import use_kernel_func_from_hub, use_kernelized_func
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.generic import maybe_autocast
from transformers.utils.output_capturing import OutputRecorder
from transformers.models.t5gemma2.configuration_t5gemma2 import (
    T5Gemma2EncoderConfig,
    T5Gemma2TextConfig,
)


def mean_pool(hidden_states, attention_mask):
    """
    Perform masked mean pooling over sequence dimension.

    Args:
        hidden_states: (batch_size, seq_len, hidden_dim)
        attention_mask: (batch_size, seq_len)

    Returns:
        pooled: (batch_size, hidden_dim)
    """
    mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
    masked_sum = (hidden_states * mask).sum(dim=1)  # (B, H)
    mask_sum = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
    return masked_sum / mask_sum


class T5Gemma2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst T5Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class T5Gemma2MLP(nn.Module):
    def __init__(self, config: T5Gemma2TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        hidden_states = self.dropout(hidden_states)
        down_proj = self.down_proj(hidden_states)
        return down_proj


class T5Gemma2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: T5Gemma2TextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.layer_types = list(set(config.layer_types))
        self.rope_type = {}
        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None:
                continue

            self.rope_type[layer_type] = rope_params["rope_type"]
            rope_init_fn: Callable = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            curr_inv_freq, curr_attention_scaling = rope_init_fn(
                self.config, device, layer_type=layer_type
            )
            self.register_buffer(
                f"{layer_type}_inv_freq", curr_inv_freq, persistent=False
            )
            self.register_buffer(
                f"{layer_type}_original_inv_freq",
                curr_inv_freq.clone(),
                persistent=False,
            )
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[T5Gemma2TextConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
        layer_type: Optional[str] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        inv_freq_expanded = (
            inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


@use_kernelized_func(apply_rotary_pos_emb)
class T5Gemma2SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: T5Gemma2TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = (
            config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        )
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = False  # Only used by the encoder

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.sliding_window = (
            config.sliding_window if self.layer_type == "sliding_attention" else None
        )
        self.is_sliding = self.layer_type == "sliding_attention"

        self.q_norm = T5Gemma2RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = T5Gemma2RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class T5Gemma2EncoderLayer(GradientCheckpointingLayer):
    """Encoder sub-layer."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]

        self.self_attn = T5Gemma2SelfAttention(
            config=config,
            layer_idx=layer_idx,
        )
        self.pre_self_attn_layernorm = T5Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_self_attn_layernorm = T5Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.mlp = T5Gemma2MLP(config)
        self.pre_feedforward_layernorm = T5Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = T5Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor,]:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            **kwargs,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


@dataclass
class T5Gemma2EncoderOutput(BaseModelOutput):
    """BaseModelOutput extended with an optional CLS state from cross-attention."""

    cls_state: Optional[torch.FloatTensor] = None


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


class T5Gemma2TextScaledWordEmbedding(nn.Embedding):
    """T5Gemma2 Embedding: override to add eoi token embedding separately."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
        eoi_token_index: int = 256_000,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)
        self.eoi_token_index = eoi_token_index
        self.eoi_embedding = nn.Parameter(torch.zeros(self.embedding_dim))

    def forward(self, input_ids: torch.Tensor):
        # Use the plain Python float instead of the tensor buffer to avoid
        # torch.compile version-tracking the scalar across multiple forward
        # calls in the same backward graph.
        input_embeddings = super().forward(input_ids) * self.scalar_embed_scale
        # input_embeddings = super().forward(input_ids) * self.embed_scale.to(
        #     self.weight.dtype
        # )
        # input_embeddings[input_ids == self.eoi_token_index] = self.eoi_embedding.to(
        #     input_embeddings.dtype
        # )
        mask = (input_ids == self.eoi_token_index).unsqueeze(-1)
        eoi = self.eoi_embedding.to(input_embeddings.dtype)
        input_embeddings = torch.where(mask, eoi, input_embeddings)
        return input_embeddings


@auto_docstring
class T5Gemma2PreTrainedModel(PreTrainedModel):
    config: T5Gemma2EncoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    _no_split_modules = [
        "T5Gemma2EncoderLayer",
        "T5Gemma2CrossAttentionEncoderLayer",
    ]
    _skip_keys_device_placement = ["past_key_values"]

    # Mask creation is incompatible
    # FA due to non-default creation / SWA
    _supports_flash_attn = False
    _supports_sdpa = True
    # Flex due to custom masks not compatible to be merged after creation
    _supports_flex_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": [T5Gemma2EncoderLayer],
        "attentions": [
            OutputRecorder(T5Gemma2SelfAttention, index=1, layer_name="self_attn"),
        ],
    }
    input_modalities = ("text",)

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, T5Gemma2TextScaledWordEmbedding):
            init.zeros_(module.eoi_embedding)
            init.constant_(module.embed_scale, module.scalar_embed_scale)
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif "RMSNorm" in module.__class__.__name__:
            init.zeros_(module.weight)
        elif isinstance(module, T5Gemma2RotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(
                    getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq
                )


def sliding_window_mask_function(sliding_window: int, is_causal=True) -> Callable:
    """
    This creates uni/bidirectional attention mask with sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        if is_causal:
            left_window_size, right_window_size = sliding_window, 0
        else:
            left_window_size, right_window_size = (
                (sliding_window + 1) // 2,
                (sliding_window) // 2 + 1,
            )

        dist = q_idx - kv_idx
        left_mask = (dist >= 0) & (dist < left_window_size)
        right_mask = (dist < 0) & (-dist < right_window_size)
        return left_mask | right_mask

    return inner_mask


class T5Gemma2Encoder(T5Gemma2PreTrainedModel):
    config: T5Gemma2EncoderConfig
    _can_record_outputs = {
        "attentions": T5Gemma2SelfAttention,
        "hidden_states": T5Gemma2EncoderLayer,
    }

    def __init__(
        self,
        config: T5Gemma2EncoderConfig,
        eoi_token_index: int = 256_000,
        cross_attention_layers: list[int] | None = None,
    ):
        super().__init__(config)
        text_config = config.text_config
        self.padding_idx = getattr(config, "pad_token_id", text_config.pad_token_id)
        self.vocab_size = text_config.vocab_size

        self.embed_tokens = T5Gemma2TextScaledWordEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            self.padding_idx,
            embed_scale=text_config.hidden_size**0.5,
            eoi_token_index=eoi_token_index,
        )
        self.norm = T5Gemma2RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList(
            [
                T5Gemma2EncoderLayer(text_config, layer_idx)
                for layer_idx in range(text_config.num_hidden_layers)
            ]
        )
        self.dropout = nn.Dropout(text_config.dropout_rate)
        self.rotary_emb = T5Gemma2RotaryEmbedding(text_config)

        self.text_config = text_config

        # ---- Cross-attention layers for CLS token extraction (MLLama-style) ----
        self.cross_attention_layer_ids = (
            cross_attention_layers if cross_attention_layers else None
        )
        if self.cross_attention_layer_ids is not None:
            self._cross_attention_layer_set = set(self.cross_attention_layer_ids)
            self.cls_query = nn.Parameter(
                torch.randn(1, 1, text_config.hidden_size) * 0.02
            )
            # One block per selected layer + one for the final encoder output
            self.cross_attn_layers = nn.ModuleList(
                [
                    T5Gemma2CrossAttentionEncoderLayer(text_config, layer_idx=idx)
                    for idx in self.cross_attention_layer_ids
                ]
            )
            self.cross_attn_final = T5Gemma2CrossAttentionEncoderLayer(
                text_config, layer_idx=text_config.num_hidden_layers
            )

        # Propagate attn implementation from the outer config to text_config,
        # so that T5Gemma2SelfAttention (which receives text_config) can find it.
        if not getattr(text_config, "_attn_implementation", None):
            text_config._attn_implementation = getattr(
                config, "_attn_implementation", "eager"
            )

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring(
        custom_args="""
        cls_position (`int`, *optional*, defaults to `None`):
            If set and `output_hidden_states=True`, extract hidden states at this position index instead of mean pooling.
            Useful when a CLS token is prepended (e.g. position 0).
        return_full_hidden_states (`bool`, *optional*, defaults to `False`):
            If `True` and `output_hidden_states=True`, return the full hidden state tensors `(B, L, H)` for each layer
            instead of pooling them to a single vector."""
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        cls_position: Optional[int] = None,
        return_full_hidden_states: bool = False,
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

        # As we want to pass `past_key_values=None` explicitly everywhere, we need to pop them from kwargs if present
        kwargs.pop("past_key_values", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                0, inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)

        if not isinstance(self_attn_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }
            self_attn_mask_mapping = {
                "full_attention": create_bidirectional_mask(**mask_kwargs),
                "sliding_attention": create_bidirectional_mask(
                    **mask_kwargs,
                    and_mask_function=sliding_window_mask_function(
                        self.text_config.sliding_window, is_causal=False
                    ),
                ),
            }

        # input layer
        hidden_states = inputs_embeds

        # global and local position embeddings
        position_embeddings = {}
        for layer_type in self.text_config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(
                hidden_states, position_ids, layer_type
            )

        # dropout
        hidden_states = self.dropout(hidden_states)

        # Resolve the raw attention_mask for mean pooling
        # (attention_mask may have been replaced by the dict above)
        raw_attention_mask = (
            attention_mask if not isinstance(attention_mask, dict) else None
        )
        if raw_attention_mask is None:
            # Fallback: all ones
            raw_attention_mask = torch.ones(
                hidden_states.shape[:2],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        # Initialise CLS state for cross-attention (MLLama-style)
        cls_state = None
        cls_intermediates = None
        cross_attn_idx = 0
        if self.cross_attention_layer_ids is not None:
            cls_state = self.cls_query.expand(
                hidden_states.shape[0], -1, -1
            ).to(dtype=hidden_states.dtype)
            cls_intermediates = []

        all_hidden_states = () if output_hidden_states else None

        for layer_idx, layer_module in enumerate(
            self.layers[: self.text_config.num_hidden_layers]
        ):
            if output_hidden_states:
                if return_full_hidden_states:
                    all_hidden_states += (hidden_states,)  # (B, L, H)
                elif cls_position is not None:
                    all_hidden_states += (hidden_states[:, cls_position, :],)
                else:
                    all_hidden_states += (mean_pool(hidden_states, raw_attention_mask),)

            hidden_states = layer_module(
                hidden_states,
                position_embeddings[layer_module.attention_type],
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                **kwargs,
            )

            # Cross-attention: CLS queries encoder hidden states at this layer
            if (
                cls_state is not None
                and layer_idx in self._cross_attention_layer_set
            ):
                cls_state = self.cross_attn_layers[cross_attn_idx](
                    cls_state, hidden_states, raw_attention_mask
                )
                cross_attn_idx += 1
                # Save intermediate CLS representation after each cross-attn block
                cls_intermediates.append(cls_state.squeeze(1))  # (B, H)

        if output_hidden_states:
            if return_full_hidden_states:
                all_hidden_states += (hidden_states,)  # (B, L, H)
            elif cls_position is not None:
                all_hidden_states += (hidden_states[:, cls_position, :],)
            else:
                all_hidden_states += (mean_pool(hidden_states, raw_attention_mask),)

        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Stack intermediate CLS representations: (B, num_cross_attn+1, H)
        # 5 from mid-network cross-attention blocks + 1 from final output
        if cls_intermediates is not None:
            # Cross-attend to the final (normed) encoder output
            cls_state = self.cross_attn_final(
                cls_state, hidden_states, raw_attention_mask
            )
            cls_intermediates.append(cls_state.squeeze(1))  # (B, H)
            cls_state = torch.stack(cls_intermediates, dim=1)  # (B, 6, H)

        return T5Gemma2EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            cls_state=cls_state,
        )


__all__ = [
    "T5Gemma2PreTrainedModel",
    "T5Gemma2Encoder",
    "T5Gemma2EncoderOutput",
    "T5Gemma2CrossAttention",
    "T5Gemma2CrossAttentionEncoderLayer",
]
