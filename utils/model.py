import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import AutoConfig
from peft import TaskType
import torch.nn.functional as F
from torch import Tensor

import torch
import torch.nn as nn
from transformers import (
    Gemma3Config,
    Gemma3Model,  # Decoder
)

# modified version with mean pool
from gemma3textmodel import Gemma3TextModel

from collections import defaultdict
import numpy as np


def get_model(args, model_config):

    config = model_config
    if model_config is None:
        config = AutoConfig.from_pretrained(args.model_name_or_path)

    # when we need to overrid the config we need to load the config in from pretrained below
    if args.activation_checkpointing:
        config.use_cache = False
        # by default is true: about use cache
        # https://github.com/huggingface/transformers/issues/28499

    encoder = convert_gemma_decoder_to_encoder(
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        verbose=True,
    )

    if args.attention_pooling:
        model = EmbeddingGemmaHiddenPool(
            encoder,
            attention_dim=args.attention_dim,
        )
    else:
        model = EmbeddingGemma(encoder)

    if args.freeze_encoder:
        """Freeze the pretrained Gemma encoder (useful for initial training)"""
        for param in model.encoder.parameters():
            param.requires_grad = False

        # """Unfreeze the pretrained Gemma encoder (for fine-tuning)"""
        # for param in model.encoder.parameters():
        #     param.requires_grad = True

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


def eager_attention_forward(
    query_weight: torch.Tensor,
    U_states: torch.Tensor,
    V_states: torch.Tensor,
    hidden_states: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:

    gating_mechanism = (torch.tanh(V_states.float()) * torch.sigmoid(U_states.float())).to(
        hidden_states.dtype
    )

    attn_weights = torch.matmul(query_weight, gating_mechanism.transpose(2, 3)) * scaling

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        hidden_states.dtype
    )
    attn_weights = nn.functional.dropout(attn_weights, p=dropout)
    attn_output = torch.matmul(attn_weights, hidden_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class GatedAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 1,
        head_dim: int = None,
    ):
        """
        Initializes the Gated Attention mechanism.

        Args:
            d_model (int): The dimension of the hidden representations (d).
            d_attn (int): The dimension of the attention space (L).
        """
        super().__init__()

        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        self.head_dim = head_dim
        # Initialize the projection matrices V and U (d_attn x d_model)
        self.U = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.V = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)

        # Initialize the aggregation vector w (d_attn x 1)
        self.w = nn.Parameter(torch.randn(head_dim, 1))

        # nn.init.xavier_uniform_(self.V.weight)
        # nn.init.xavier_uniform_(self.U.weight)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the attention weights a_k and the final context vector z.

        Args:
            H (torch.Tensor): The hidden representations, of shape (B x K, d).
                              B is the batch size, K is the number of instances, d is d_model.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The context vector z (1, d) and
                                               the attention weights a_k (K, 1).
        """
        # --- 1. Gated Attention Score Calculation ---

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        U_states = self.U(hidden_states).view(hidden_shape).transpose(1, 2)
        V_states = self.V(hidden_states).view(hidden_shape).transpose(1, 2)

        attn_output, attn_weights = eager_attention_forward(
            query_weight=self.w,
            U_states=U_states,
            V_states=V_states,
            hidden_states=hidden_states,
            scaling=self.head_dim**0.5,
            dropout=0.0,
        )

        return attn_output, attn_weights


class Normalize(nn.Module):
    """Normalizes embeddings to unit length (L2 norm)"""

    def forward(self, embeddings: Tensor) -> Tensor:
        return F.normalize(embeddings, p=2, dim=1)


class Projection(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        self.up = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)
        self.down = nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=False)

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


def convert_gemma_decoder_to_encoder(
    config,
    torch_dtype,
    attn_implementation,
    verbose=True,
):
    """
    Converts a Gemma-3 pretrained causal decoder into
    a bidirectional encoder suitable for embedding training.
    """

    if verbose:
        print(f"Loading pretrained decoder model")

    # 1. Load the pretrained decoder
    decoder = Gemma3Model.from_pretrained(
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )

    # 2. Modify config to enable bidirectional attention
    cfg: Gemma3Config = decoder.config
    cfg.text_config.use_bidirectional_attention = True

    # (Optional) Reduce max context length for efficiency
    # cfg.text_config.max_position_embeddings = 1024

    if verbose:
        print("Creating encoder with bidirectional attention...")

    # 3. Create an encoder model with the same architecture
    encoder = Gemma3TextModel(
        cfg,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )

    # 4. Load compatible weights from decoder → encoder
    decoder_state = decoder.state_dict()
    encoder_state = encoder.state_dict()

    # Auto-filter incompatible keys: remove lm_head, etc.
    filtered_state = {
        k: v
        for k, v in decoder_state.items()
        if k in encoder_state and v.shape == encoder_state[k].shape
    }

    missing_in_decoder = [k for k in encoder_state.keys() if k not in filtered_state]
    if verbose:
        print(f"Loading {len(filtered_state)} matched weights.")
        print(
            f"Skipping {len(decoder_state) - len(filtered_state)} incompatible weights (e.g., lm_head)."
        )
        print(f"Encoder has {len(missing_in_decoder)} newly initialized params.")

    # Load (strict=False is now safe)
    encoder.load_state_dict(filtered_state, strict=False)
    return encoder


class EmbeddingGemma(nn.Module):
    """
    Gemma-3 encoder with mean pooling + projection,
    as described in the EmbeddingGemma paper.
    """

    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder
        h = self.encoder.config.text_config.hidden_size
        self.pooling = MeanPooling()
        self.projection = Projection(input_dim=h, hidden_dim=4 * h)
        self.normalize = Normalize()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, H)

        # Masked mean pooling
        hidden = self.pooling(hidden, attention_mask)
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)

        return hidden


class EmbeddingGemmaHiddenPool(nn.Module):
    """
    Gemma-3 encoder with mean pooling + projection,
    as described in the EmbeddingGemma paper.
    """

    def __init__(
        self,
        encoder,
        attention_dim: int = None,
    ):
        super().__init__()

        self.encoder = encoder
        h = self.encoder.config.text_config.hidden_size
        self.pooling = GatedAttention(d_model=h, d_attn=attention_dim)
        self.projection = Projection(input_dim=h, hidden_dim=4 * h)
        self.normalize = Normalize()

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # outputs.hidden_states should be a list of tensors with shapes [B x 1 x D]
        hidden = torch.cat(outputs.hidden_states, dim=1)  # [B, L, D]
        hidden = self.pooling(hidden, attention_mask)
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)

        return hidden
