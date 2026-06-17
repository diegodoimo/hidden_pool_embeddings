"""Embedding-model wrappers for the cross-attention and LoRA-CLS variants.

Mirrors :mod:`models.t5gemma2_bak`: lifted out of
``models/t5gemma2model.py`` because the main module now only supports the
mean / CLS / both attention-pooling head. The two classes here keep the
Procrustes alignment plumbing in place so that a follow-up can re-attach
them to the training scripts if the variants are revived.

Build helpers (encoder loading, weight extraction) intentionally stay in
``models.t5gemma2model``: pass them an already-loaded
:class:`T5Gemma2CrossAttentionEncoder` (built via
``load_t5gemma2_encoder_with_cross_attention`` below) into
:class:`EmbeddingT5Gemma2CrossAttention`.
"""

import torch
import torch.nn as nn
from transformers.masking_utils import create_bidirectional_mask

from models.t5gemma2 import (
    T5Gemma2Encoder,
    sliding_window_mask_function,
)
from models.t5gemma2_bak import (
    T5Gemma2CrossAttentionEncoder,
    T5Gemma2LoRACLSEncoderLayer,
)
from models.modules import (
    AttentionPooling,
    GatedAttention,
    Normalize,
    ProcrustesAlignment,
    Projection,
)


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


__all__ = [
    "EmbeddingT5Gemma2CrossAttention",
    "EmbeddingT5Gemma2LoRACLSAttn",
]
