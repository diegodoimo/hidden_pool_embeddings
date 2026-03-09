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

from models.t5gemma2model import load_t5gemma2_encoder
from models.modules import mean_pool


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

    def forward(self, batch):
        bs = batch["bs"]
        num_hard_neg = int((len(batch["input_ids"]) - 2 * bs) / bs)

        outputs = self.lm(
            batch["input_ids"],
            batch["attention_mask"],
        )

        passage_features_all_tokens = outputs.last_hidden_state
        return {
            "query_passage_features": torch.stack(
                [
                    passage_features_all_tokens[i, [batch["seq_lens"][i] - 1]]
                    for i in range(bs)
                ]
            ),
            "passage_passage_features": torch.stack(
                [
                    passage_features_all_tokens[i, [batch["seq_lens"][i] - 1]]
                    for i in range(bs, 2 * bs)
                ]
            ),
            "negative_passage_features": (
                None
                if num_hard_neg == 0
                else torch.stack(
                    [
                        passage_features_all_tokens[i, [batch["seq_lens"][i] - 1]]
                        for i in range(2 * bs, len(batch["seq_lens"]))
                    ]
                ).view(bs, num_hard_neg, -1)
            ),
        }


class F2LLMT5Gemma2:
    """
    Drop-in replacement for F2LLM that uses the T5Gemma2 bidirectional encoder
    (loaded via load_t5gemma2_encoder) instead of a causal LM.

    The forward interface is identical to F2LLM: it accepts the same batch dict
    and returns the same dict of {query, passage, negative}_passage_features.

    Key differences vs F2LLM:
    - Mean pooling is used instead of last-token pooling, which is more
      appropriate for a bidirectional encoder.
    - `seq_lens` in the batch is not used (mean pooling uses attention_mask).
    - The underlying model is stored as `self.encoder` instead of `self.lm`.
      An `lm` property is provided for compatibility with F2LLMEvalWrapper.
    """

    def __init__(
        self,
        model_path,
        max_seq_length: int = 512,
        args=None,
        attn_implementation: str = "sdpa",
    ):
        self.args = args
        self.dtype = torch.bfloat16
        self.device = None  # set after accelerator.prepare

        self.encoder = load_t5gemma2_encoder(
            model_name_or_path=model_path,
            torch_dtype=self.dtype,
            attn_implementation=attn_implementation,
        )
        self.encoder.config.use_cache = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length

    # ------------------------------------------------------------------ compat
    @property
    def lm(self):
        """Alias so that F2LLMEvalWrapper (which expects `.lm`) still works."""
        return self.encoder

    # ------------------------------------------------------------------
    def set_device(self):
        self.device = next(self.encoder.parameters()).device

    def forward(self, batch):
        bs = batch["bs"]
        num_hard_neg = int((len(batch["input_ids"]) - 2 * bs) / bs)

        outputs = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        # Mean-pool the full token sequence (bidirectional encoder)
        hidden_states = outputs.last_hidden_state  # (total_batch, seq_len, H)
        pooled = mean_pool(hidden_states, batch["attention_mask"])  # (total_batch, H)

        # Unsqueeze to match F2LLM's (N, 1, H) convention for query/passage
        return {
            "query_passage_features": pooled[:bs].unsqueeze(1),  # (bs, 1, H)
            "passage_passage_features": pooled[bs : 2 * bs].unsqueeze(1),  # (bs, 1, H)
            "negative_passage_features": (
                None
                if num_hard_neg == 0
                else pooled[2 * bs :].view(bs, num_hard_neg, -1)  # (bs, n_neg, H)
            ),
        }
