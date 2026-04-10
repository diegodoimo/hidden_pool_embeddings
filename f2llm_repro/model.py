import sys
from pathlib import Path

# Ensure the project root is on the path when this file is imported from the
# f2llm_repro sub-directory so that `models.*` and `utils.*` are accessible.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from models.t5gemma2model import get_model_t5gemma2_model
from models.t5gemma2_decoder import get_model_t5gemma2_decoder


class F2LLM:
    def __init__(self, model_path, max_seq_length=512, args=None):

        self.args = args
        self.dtype = torch.bfloat16
        self.device = None  # set after accelerator.prepare
        self.lm = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
        )
        self.lm.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length

    def set_device(self):
        self.device = self.lm.device

    # def forward(self, batch):
    #     """OLD: Python for-loop extracts one last-token embedding at a time,
    #     creating bs*(2+num_hard_neg) intermediate tensors. Also wraps query
    #     and passage features in an unnecessary [bs,1,d] shape that every
    #     call site immediately squeeze(1)'s away."""
    #     bs = batch["bs"]
    #     num_hard_neg = int((len(batch["input_ids"]) - 2 * bs) / bs)
    #     outputs = self.lm(batch["input_ids"], batch["attention_mask"])
    #     passage_features_all_tokens = outputs.last_hidden_state
    #     return {
    #         "query_passage_features": torch.stack(
    #             [passage_features_all_tokens[i, [batch["seq_lens"][i] - 1]]
    #              for i in range(bs)]
    #         ),                                                  # [bs, 1, d]
    #         "passage_passage_features": torch.stack(
    #             [passage_features_all_tokens[i, [batch["seq_lens"][i] - 1]]
    #              for i in range(bs, 2 * bs)]
    #         ),                                                  # [bs, 1, d]
    #         "negative_passage_features": (
    #             None if num_hard_neg == 0
    #             else torch.stack(
    #                 [passage_features_all_tokens[i, [batch["seq_lens"][i] - 1]]
    #                  for i in range(2 * bs, len(batch["seq_lens"]))]
    #             ).view(bs, num_hard_neg, -1)
    #         ),                                                  # [bs, n, d]
    #     }

    def forward(self, batch):
        """Vectorized last-token pooling.

        Replaces the Python for-loop with a single advanced-index gather over
        all sequences at once.  Also returns query/passage features as [bs, d]
        instead of [bs, 1, d] — the extra dim was always immediately squeezed
        away by every call site.
        """
        bs = batch["bs"]
        num_hard_neg = int((len(batch["input_ids"]) - 2 * bs) / bs)

        outputs = self.lm(
            batch["input_ids"],
            batch["attention_mask"],
        )

        hidden = outputs.last_hidden_state          # [total_seqs, max_len, d]
        last_idx = batch["seq_lens"] - 1             # [total_seqs]
        seq_range = torch.arange(len(last_idx), device=last_idx.device)
        features = hidden[seq_range, last_idx]       # [total_seqs, d]

        return {
            "query_passage_features": features[:bs],                  # [bs, d]
            "passage_passage_features": features[bs : 2 * bs],       # [bs, d]
            "negative_passage_features": (
                None
                if num_hard_neg == 0
                else features[2 * bs :].view(bs, num_hard_neg, -1)   # [bs, n, d]
            ),
        }


class F2LLMT5Gemma2:
    """
    Drop-in replacement for F2LLM that uses EmbeddingT5Gemma2 (built by
    get_model_t5gemma2_model) as the backbone instead of a raw causal LM.

    Compared to the previous implementation this adds:
    - A learned Projection FFN between mean-pooled encoder output and L2
      normalisation, trained end-to-end through the contrastive loss.
    - Optional gated-attention pooling over per-layer hidden states
      (attention_pooling=True / cls_query_pooling=True via args).

    Interface contract:
    - self.lm   : the EmbeddingT5Gemma2 nn.Module, passed to
                  accelerator.prepare so the full encoder + projection +
                  normalise stack is moved to GPU / wrapped by DeepSpeed.
    - forward() : accepts the stacked batch dict, calls self.lm, and
                  returns the {query,passage,negative}_passage_features dict
                  expected by accelerate_train.
    - Eval      : use EmbeddingModelEvalWrapper (in f2llm_train.py) which
                  calls self.lm(input_ids, attention_mask) directly, since
                  EmbeddingT5Gemma2 already returns L2-normalised embeddings.
    """

    def __init__(
        self,
        model_path: str,
        max_seq_length: int = 512,
        args=None,
        attn_implementation: str = "sdpa",
    ):
        self.args = args
        self.device = None  # set after accelerator.prepare

        attention_pooling = getattr(args, "attention_pooling", False)
        cls_query_pooling = getattr(args, "cls_query_pooling", False)
        attn_implementation = getattr(args, "attn_implementation", attn_implementation)
        lora_cls = getattr(args, "lora_cls", False)
        lora_rank = getattr(args, "lora_rank", 64)
        cross_attention = getattr(args, "cross_attention", False)
        cross_attention_layers = getattr(args, "cross_attention_layers", None)
        lora_cls_attn = getattr(args, "lora_cls_attn", False)
        num_pooling_heads = getattr(args, "num_pooling_heads", None)
        gated_attention = getattr(args, "gated_attention", False)

        # Build EmbeddingT5Gemma2 (encoder + MeanPooling + Projection + Normalize)
        self._embedding_model, _, _ = get_model_t5gemma2_model(
            model_name_or_path=model_path,
            activation_checkpointing=False,  # enabled separately after accelerator.prepare
            attention_pooling=attention_pooling,
            cls_query_pooling=cls_query_pooling,
            attn_implementation=attn_implementation,
            lora_cls=lora_cls,
            lora_rank=lora_rank,
            cross_attention=cross_attention,
            cross_attention_layers=cross_attention_layers,
            lora_cls_attn=lora_cls_attn,
            num_pooling_heads=num_pooling_heads,
            gated_attention=gated_attention,
        )
        self._embedding_model.encoder.config.use_cache = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length

    # ------------------------------------------------------------------ compat
    @property
    def lm(self):
        """The EmbeddingT5Gemma2 nn.Module; passed to accelerator.prepare."""
        return self._embedding_model

    @lm.setter
    def lm(self, value):
        """Allows `model.lm, opt, sched = accelerator.prepare(model.lm, ...)` to
        update the reference after accelerate wraps it."""
        self._embedding_model = value

    # ------------------------------------------------------------------
    def set_device(self):
        self.device = next(iter(self._embedding_model.parameters())).device

    def forward(self, batch):
        bs = batch["bs"]
        num_hard_neg = int((len(batch["input_ids"]) - 2 * bs) / bs)

        # EmbeddingT5Gemma2.forward returns (total_batch, H) L2-normalised embeddings
        embeddings = self._embedding_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )  # (total_batch, H)

        # Return [bs, d] directly instead of [bs, 1, d] — the extra dim was
        # always immediately squeezed away by every call site.
        return {
            # "query_passage_features": embeddings[:bs].unsqueeze(1),           # OLD [bs, 1, H]
            # "passage_passage_features": embeddings[bs : 2 * bs].unsqueeze(1), # OLD [bs, 1, H]
            "query_passage_features": embeddings[:bs],                          # [bs, H]
            "passage_passage_features": embeddings[bs : 2 * bs],               # [bs, H]
            "negative_passage_features": (
                None
                if num_hard_neg == 0
                else embeddings[2 * bs :].view(bs, num_hard_neg, -1)  # (bs, n_neg, H)
            ),
        }


class F2LLMT5Gemma2Decoder:
    """Drop-in replacement for F2LLM that uses the T5Gemma2 *decoder* as a
    causal embedding model with EOS-token pooling (like Qwen3-Embedding).

    Interface contract identical to ``F2LLMT5Gemma2`` (encoder variant):
    - self.lm   : the EmbeddingT5Gemma2Decoder nn.Module.
    - forward() : returns {query,passage,negative}_passage_features.
    - Eval      : use EmbeddingModelEvalWrapper since forward already returns
                  L2-normalised embeddings.
    """

    def __init__(
        self,
        model_path: str,
        max_seq_length: int = 512,
        args=None,
        attn_implementation: str = "sdpa",
    ):
        self.args = args
        self.device = None

        attention_pooling = getattr(args, "attention_pooling", False)
        attn_implementation = getattr(args, "attn_implementation", attn_implementation)
        lora_cls = getattr(args, "lora_cls", False)
        lora_rank = getattr(args, "lora_rank", 64)
        num_pooling_heads = getattr(args, "num_pooling_heads", None)
        gated_attention = getattr(args, "gated_attention", False)

        self._embedding_model, _, _ = get_model_t5gemma2_decoder(
            model_name_or_path=model_path,
            activation_checkpointing=False,
            attention_pooling=attention_pooling,
            attn_implementation=attn_implementation,
            lora_cls=lora_cls,
            lora_rank=lora_rank,
            num_pooling_heads=num_pooling_heads,
            gated_attention=gated_attention,
        )
        self._embedding_model.encoder.config.use_cache = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length

    @property
    def lm(self):
        return self._embedding_model

    @lm.setter
    def lm(self, value):
        self._embedding_model = value

    def set_device(self):
        self.device = next(iter(self._embedding_model.parameters())).device

    def forward(self, batch):
        bs = batch["bs"]
        num_hard_neg = int((len(batch["input_ids"]) - 2 * bs) / bs)

        embeddings = self._embedding_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        return {
            "query_passage_features": embeddings[:bs],
            "passage_passage_features": embeddings[bs : 2 * bs],
            "negative_passage_features": (
                None
                if num_hard_neg == 0
                else embeddings[2 * bs :].view(bs, num_hard_neg, -1)
            ),
        }
