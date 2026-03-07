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

from models.t5gemma2 import T5Gemma2Encoder
from models.modules import (
    MeanPooling,
    Projection,
    Normalize,
    GatedAttention,
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
    encoder = T5Gemma2Encoder(encoder_config, eoi_token_index=eoi_token_index)

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
    missing_non_buffer = [
        k for k in missing if "rotary_emb" not in k and "embed_scale" not in k
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
    attention_dim,
    attn_implementation: str = "sdpa",
):
    """
    Build a T5Gemma2-based embedding model, mirroring the interface of
    ``gemma3model.get_model``.
    """
    encoder = load_t5gemma2_encoder(
        model_name_or_path=model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )

    if activation_checkpointing:
        encoder.config.use_cache = False

    if attention_pooling:
        if cls_query_pooling:
            model = EmbeddingT5Gemma2HiddenPoolCLS(
                encoder,
                attention_dim=attention_dim,
            )
        else:
            model = EmbeddingT5Gemma2HiddenPool(
                encoder,
                attention_dim=attention_dim,
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
        attention_dim: int = None,
    ):
        super().__init__()
        self.encoder = encoder
        h = self.encoder.text_config.hidden_size
        self.pooling = GatedAttention(
            hidden_size=h,
            num_attention_heads=1,
            head_dim=attention_dim,
        )
        self.projection = Projection(input_dim=h, hidden_dim=4 * h)
        self.normalize = Normalize()

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
    and pooled via gated attention over layers.
    """

    def __init__(
        self,
        encoder: T5Gemma2Encoder,
        attention_dim: int = None,
    ):
        super().__init__()
        self.encoder = encoder
        h = self.encoder.text_config.hidden_size
        # Learnable CLS query embedding (1, 1, H)
        self.cls_query = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.pooling = GatedAttention(
            hidden_size=h,
            num_attention_heads=1,
            head_dim=attention_dim,
        )
        self.projection = Projection(input_dim=h, hidden_dim=4 * h)
        self.normalize = Normalize()

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
