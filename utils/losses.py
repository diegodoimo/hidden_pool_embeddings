import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embeddings: torch.Tensor, key_embeddings: torch.Tensor):
        """
        Args:
            query_embeddings: (batch_size, embedding_dim)
            key_embeddings: (batch_size, embedding_dim)
        Returns:
            loss: scalar
        """
        batch_size = query_embeddings.shape[0]

        # Compute similarity matrix
        # (batch_size, batch_size)
        logits = torch.matmul(query_embeddings, key_embeddings.T) / self.temperature

        # Labels: diagonal elements are positives
        labels = torch.arange(batch_size, device=logits.device)

        # Cross-entropy loss (symmetric)
        loss_q = F.cross_entropy(logits, labels)
        loss_k = F.cross_entropy(logits.T, labels)

        return (loss_q + loss_k) / 2


class EmbeddingGemmaLossDistributed(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    @staticmethod
    def pairwise_dot_squared(x, B):
        dots = x @ x.t()
        dots_sq = dots**2
        diag_mask = torch.eye(B, dtype=torch.bool, device=dots_sq.device)
        dots_sq = dots_sq.masked_fill(diag_mask, 0)
        return dots_sq.sum() / (B * (B - 1))

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        doc_ids: torch.Tensor,
    ):
        # --- 1. Distributed Gathering ---
        # thids should support backward
        all_queries = torch.distributed.nn.functional.all_gather(query_embeddings)
        all_queries = torch.cat(all_queries, dim=0)

        all_docs = torch.distributed.nn.functional.all_gather(doc_embeddings)
        all_docs = torch.cat(all_docs, dim=0)

        all_doc_ids = torch.distributed.nn.functional.all_gather(doc_ids)
        all_doc_ids = torch.cat(all_doc_ids, dim=0)

        batch_size = all_queries.size(0)

        # Spherical loss
        Ls = self.pairwise_dot_squared(all_queries, B=batch_size) + self.pairwise_dot_squared(
            all_docs, B=batch_size
        )

        # --- 2. Compute Logits ---
        logits = torch.matmul(all_queries, all_docs.T) / self.temperature
        labels = torch.arange(batch_size, device=logits.device)

        # --- 3. Combined Masking Strategy ---

        # Mask 1: Duplicate positives (same doc_id, excluding diagonal)
        doc_id_matches = all_doc_ids.unsqueeze(1) == all_doc_ids.unsqueeze(0)
        doc_id_matches.fill_diagonal_(False)

        # Mask 2: Documents with similarity >= positive similarity
        # Extract positive similarities (diagonal elements)
        positive_sims = logits.diagonal() + 0.1  # [B] #qwen3

        # Compare each logit to its corresponding positive similarity
        # similarity_mask[i, j] = True if logits[i, j] >= positive_sims[i]
        similarity_mask = logits >= positive_sims.unsqueeze(1)  # [B, B]

        # Don't mask the positive pair itself (diagonal)
        similarity_mask.fill_diagonal_(False)

        # Combine masks: mask if EITHER condition is true
        combined_mask = doc_id_matches | similarity_mask

        # Apply mask
        logits_masked = logits.masked_fill(combined_mask, float("-inf"))

        # --- 4. Symmetric Loss ---
        # For symmetric loss, we need to handle the transpose carefully
        # For doc-to-query direction, we compare against the same positives
        # logits_T_masked = logits_masked.T

        loss_q = F.cross_entropy(logits_masked, labels)
        # loss_k = F.cross_entropy(logits_T_masked, labels)

        return Ls + loss_q


class EmbeddingGemmaLossHardNegatives(nn.Module):
    def __init__(self, temperature: float = 0.07, num_hard_negatives: int = 7):
        super().__init__()
        self.temperature = temperature
        self.num_hard_negatives = num_hard_negatives

    @staticmethod
    def pairwise_dot_squared(x, B, ids=None):
        dots = x @ x.t()
        dots_sq = dots**2
        diag_mask = torch.eye(B, dtype=torch.bool, device=dots_sq.device)
        same_id = ids.unsqueeze(1) == ids.unsqueeze(0)
        same_id = same_id.masked_fill(diag_mask, False)
        exclude = diag_mask | same_id
        dots_sq = dots_sq.masked_fill(exclude, 0)
        num_pairs = (B * (B - 1) - same_id.float().sum()).clamp(min=1)
        return dots_sq.sum() / num_pairs

    def forward(
        self,
        query_embeddings: torch.Tensor,  # (B, D)
        doc_embeddings: torch.Tensor,  # (B, D) positives
        hard_neg_embeddings: torch.Tensor,  # (B, num_hard_negatives, D)
        doc_ids: torch.Tensor,  # (B,)
        query_ids: torch.Tensor,  # (B,)
    ):
        batch_size = query_embeddings.size(0)

        # --- 1. Spherical Loss ---
        Ls_queries = self.pairwise_dot_squared(query_embeddings, B=batch_size, ids=query_ids)
        Ls_docs = self.pairwise_dot_squared(doc_embeddings, B=batch_size, ids=doc_ids)
        Ls = Ls_queries + Ls_docs

        # --- 2. Combine positives and hard negatives ---
        # (B, 1+num_hard_negatives, D)
        all_docs = torch.cat(
            [doc_embeddings.unsqueeze(1), hard_neg_embeddings], dim=1
        )

        # (B*(1+num_hard_negatives), D)
        num_docs_per_query = 1 + self.num_hard_negatives
        all_docs_flat = all_docs.view(batch_size * num_docs_per_query, -1)

        # --- 3. Compute Logits ---
        # (B, B*num_docs_per_query)
        logits = torch.matmul(query_embeddings, all_docs_flat.T) / self.temperature

        # (B, B, num_docs_per_query)
        logits = logits.view(batch_size, batch_size, num_docs_per_query)

        # --- 4. Prepare labels ---
        positive_logits = logits[torch.arange(batch_size), torch.arange(batch_size), 0]
        logits_flat = logits.view(batch_size, -1)
        labels = torch.arange(batch_size, device=logits.device) * num_docs_per_query

        # --- 5. Masking for repeated queries / repeated positives ---
        # Duplicate positives: doc_ids[i] == doc_ids[j] means doc_j is
        # the same document as doc_i, so it's a positive for query_i.
        dup_doc = doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0)  # (B, B)
        dup_doc.fill_diagonal_(False)

        # Duplicate queries: query_ids[i] == query_ids[j] means query_j is
        # the same query, so doc_j (its positive) is also a positive for query_i.
        dup_query = query_ids.unsqueeze(1) == query_ids.unsqueeze(0)  # (B, B)
        dup_query.fill_diagonal_(False)

        # Either condition means the positive at position [i, j, 0] must not
        # act as a negative for query_i.  Hard negatives (positions 1..K) are
        # genuinely different documents and remain as negatives.
        positive_dup_mask = dup_doc | dup_query  # (B, B)

        dup_mask_3d = torch.zeros(
            batch_size, batch_size, num_docs_per_query,
            dtype=torch.bool, device=logits.device,
        )
        dup_mask_3d[:, :, 0] = positive_dup_mask
        dup_mask_flat = dup_mask_3d.view(batch_size, -1)  # (B, B*num_docs_per_query)

        # Similarity-based mask: mask negatives harder than the positive
        # similarity_mask = logits_flat >= (positive_logits.unsqueeze(1) + 0.1)
        # similarity_mask.scatter_(1, labels.unsqueeze(1), False)

        combined_mask = dup_mask_flat # | similarity_mask
        logits_masked = logits_flat.masked_fill(combined_mask, float("-inf"))

        # --- 6. Compute Loss ---
        loss_q = F.cross_entropy(logits_masked, labels)

        return Ls + loss_q
