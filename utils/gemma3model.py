import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig
from peft import TaskType

# modified version with mean pool
from utils.gemma3textmodel import Gemma3TextModel


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
        model_name_or_path=args.model_name_or_path,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        verbose=True,
    )

    if args.attention_pooling:
        if args.cls_query_pooling:
            model = EmbeddingGemmaHiddenPoolCLS(
                encoder,
                attention_dim=args.attention_dim,
            )
        else:
            model = EmbeddingGemmaHiddenPool(
                encoder,
                attention_dim=args.attention_dim,
            )
    else:
        model = EmbeddingGemma(encoder)

    # if args.freeze_encoder:
    #     """Freeze the pretrained Gemma encoder (useful for initial training)"""
    #     for param in model.encoder.parameters():
    #         param.requires_grad = False

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
    model_name_or_path,
    config,
    dtype,
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
    decoder = Gemma3TextModel.from_pretrained(
        model_name_or_path,
        config=config,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    cfg = decoder.config
    # 2. Modify config to enable bidirectional attention
    # text_cfg = full_cfg.text_config
    cfg.use_bidirectional_attention = True

    # (Optional) Reduce max context length for efficiency
    cfg.max_position_embeddings = 4096

    if verbose:
        print("Creating encoder with bidirectional attention...")

    # 3. Create an encoder model with the same architecture

    # by default the cfg.dtype is bfloat16
    encoder = Gemma3TextModel(cfg)

    # 4. Load compatible weights from decoder → encoder
    decoder_state = decoder.state_dict()
    # decoder_state = decoder.text_model.state_dict()    # IMPORTANT!
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
        h = self.encoder.config.hidden_size
        self.pooling = MeanPooling()
        self.projection = Projection(input_dim=h, hidden_dim=4 * h)
        self.normalize = Normalize()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        **kwargs,
    ):
        # Forward to encoder with the actual arguments, not literal `None`
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden = outputs.last_hidden_state  # (B, L, H)

        # Masked mean pooling
        hidden = self.pooling(hidden, attention_mask)
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)

        return hidden


class EmbeddingGemmaHiddenPool(nn.Module):
    """
    Gemma-3 encoder with hidden pooling
    """

    def __init__(
        self,
        encoder,
        attention_dim: int = None,
    ):
        super().__init__()

        self.encoder = encoder
        h = self.encoder.config.hidden_size
        self.pooling = GatedAttention(
            hidden_size=h, num_attention_heads=1, head_dim=attention_dim
        )
        self.projection = Projection(input_dim=h, hidden_dim=4 * h)
        self.normalize = Normalize()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        **kwargs,
    ):
        # Force output_hidden_states=True so we get per-layer mean-pooled representations
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            cache_position=cache_position,
            **kwargs,
        )

        # outputs.hidden_states is a tuple of (B, D) tensors (mean-pooled per layer)
        hidden = torch.stack(outputs.hidden_states, dim=1)  # [B, num_layers, D]
        hidden, _ = self.pooling(hidden)  # [B, D]
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)

        return hidden


class EmbeddingGemmaHiddenPoolCLS(nn.Module):
    """
    Gemma-3 encoder with a learnable CLS query token prepended to the input.
    The CLS token's residual-stream representation at each layer is extracted
    and pooled via gated attention over layers.
    """

    def __init__(
        self,
        encoder,
        attention_dim: int = None,
    ):
        super().__init__()

        self.encoder = encoder
        h = self.encoder.config.hidden_size
        # Learnable CLS query embedding (1, 1, H)
        self.cls_query = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.pooling = GatedAttention(
            hidden_size=h, num_attention_heads=1, head_dim=attention_dim
        )
        self.projection = Projection(input_dim=h, hidden_dim=4 * h)
        self.normalize = Normalize()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        **kwargs,
    ):
        # 1. Get token embeddings
        if inputs_embeds is None:
            inputs_embeds = self.encoder.embed_tokens(input_ids)
            input_ids = None  # pass inputs_embeds instead

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
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            cache_position=None,
            cls_position=0,  # extract position 0 (the CLS token)
            **kwargs,
        )

        # 5. outputs.hidden_states is a tuple of (B, D) tensors (CLS repr per layer)
        hidden = torch.stack(outputs.hidden_states, dim=1)  # [B, num_layers+1, D]
        hidden, _ = self.pooling(hidden)  # [B, D]
        hidden = self.projection(hidden)
        hidden = self.normalize(hidden)

        return hidden


# class ContrastiveLossEmbedding(nn.Module):
#     def __init__(self, model, loss_fn=None):
#         super().__init__()
#         self.embedder = model
#         self.loss_fn = loss_fn

#     def forward(self, queries, documents, doc_ids):

#         out_q = self.embedder(**queries)

#         out_d = self.embedder(**documents)

#         # loss = (out_q**2 + out_d**2).mean()
#         loss = self.loss_fn(
#             query_embeddings=out_q,
#             doc_embeddings=out_d,
#             doc_ids=doc_ids,
#         )

#         return loss
