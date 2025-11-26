import torch
from torch import nn
from transformers import Gemma3PreTrainedModel, Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3TextScaledWordEmbedding,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
    Gemma3DecoderLayer,
    _bidirectional_window_overlay,
)
from typing import Optional

# from ...cache_utils import Cache, DynamicCache
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import TransformersKwargs, logging
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.processing_utils import Unpack
# from transformers.utils.generic import check_model_inputs

logger = logging.get_logger(__name__)


# @dataclass
# @auto_docstring(
#     custom_intro="""
#     Base class for Gemma3 outputs, with hidden states and attentions.
#     """
# )


# ***************************************************************


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
    masked_sum = (hidden_states * mask).sum(dim=1, keepdim=True)  # (B, 1, H)

    # Compute mask sum with numerical stability
    mask_sum = mask.sum(dim=1, keepdim=1).clamp(min=1e-9)  # (B, 1)

    # Compute mean
    return masked_sum / mask_sum


#@auto_docstring
class Gemma3TextModel(Gemma3PreTrainedModel):
    config: Gemma3TextConfig
    input_modalities = "text"

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    #@check_model_inputs()
    #@auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            sliding_mask_kwargs = mask_kwargs.copy()

            if self.config.use_bidirectional_attention:
                mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(True, dtype=torch.bool)
                sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(
                    self.config.sliding_window
                )

            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(
                hidden_states, position_ids, layer_type
            )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                # here all hidden states should be of shape [B x 1  x D]
                all_hidden_states += (mean_pool(hidden_states, attention_mask),)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = self.norm(hidden_states)

        # if output_hidden_states:
        #     all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
