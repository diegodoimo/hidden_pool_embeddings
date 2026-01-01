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
        # Zero out diagonal in-place (most efficient)
        dots_sq.fill_diagonal_(0)
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
    def pairwise_dot_squared(x, B):
        dots = x @ x.t()
        dots_sq = dots**2
        # Zero out diagonal in-place
        dots_sq.fill_diagonal_(0)
        return dots_sq.sum() / (B * (B - 1))

    def forward(
        self,
        query_embeddings: torch.Tensor,  # (B,)
        doc_embeddings: torch.Tensor,  # (B,) positives
        hard_neg_embeddings: torch.Tensor,  # (B, 7, D) hard negatives
        doc_ids: torch.Tensor,  # (B,) document IDs
    ):
        batch_size = query_embeddings.size(0)

        # --- 1. Combine positives and hard negatives ---
        # Shape: (B, 1+7, D) = (B, 8, D)
        all_docs = torch.cat(
            [doc_embeddings.unsqueeze(1), hard_neg_embeddings], dim=1  # (B, 1, D)  # (B, 7, D)
        )

        # Flatten for pairwise operations: (B*8, D)
        all_docs_flat = all_docs.view(batch_size * (1 + self.num_hard_negatives), -1)

        # --- 2. Spherical Loss ---
        # Only compute on unique embeddings (queries and all docs)
        Ls_queries = self.pairwise_dot_squared(query_embeddings, B=batch_size)
        Ls_docs = self.pairwise_dot_squared(all_docs_flat, B=all_docs_flat.size(0))
        Ls = Ls_queries + Ls_docs

        # --- 3. Compute Logits ---
        # queries: (B, D), all_docs_flat: (B*8, D)
        # logits: (B, B*8)
        logits = torch.matmul(query_embeddings, all_docs_flat.T) / self.temperature

        # Reshape to separate in-batch and hard negatives
        # (B, B*8) -> (B, B, 8)
        logits = logits.view(batch_size, batch_size, 1 + self.num_hard_negatives)

        # --- 4. Prepare labels and masks ---
        # The positive for query i is at position [i, 0] after reshaping
        # We need to flatten back but track which position is positive

        # Extract positives (diagonal of first slice): (B,)
        positive_logits = logits[torch.arange(batch_size), torch.arange(batch_size), 0]

        # Flatten logits back to (B, B*8) for cross-entropy
        logits_flat = logits.view(batch_size, -1)

        # Labels: positive is at index i * 8 for query i
        labels = torch.arange(batch_size, device=logits.device) * (1 + self.num_hard_negatives)

        # --- 5. Masking ---
        # Mask 1: Duplicate positives (same doc_id)
        # Expand doc_ids to match all_docs structure
        doc_ids_expanded = doc_ids.unsqueeze(1).expand(-1, 1 + self.num_hard_negatives)
        doc_ids_flat = doc_ids_expanded.reshape(-1)  # (B*8,)

        # Create mask: doc_ids_flat[j] == doc_ids[i] for each query i
        doc_id_matches = doc_ids.unsqueeze(1) == doc_ids_flat.unsqueeze(0)  # (B, B*8)

        # Don't mask the actual positive for each query
        positive_positions = labels.unsqueeze(1)  # (B, 1)

        positive_mask = (
            torch.arange(
                batch_size * (1 + self.num_hard_negatives), device=logits.device
            ).unsqueeze(0)
            == positive_positions
        )
        doc_id_matches = doc_id_matches & ~positive_mask

        # Mask 2: Documents with similarity >= positive similarity
        similarity_mask = logits_flat >= (positive_logits.unsqueeze(1) + 0.1)  # (B, B*8)
        similarity_mask.scatter_(1, labels.unsqueeze(1), False)  # Don't mask positives

        # Combine masks
        combined_mask = doc_id_matches | similarity_mask

        # Apply mask
        logits_masked = logits_flat.masked_fill(combined_mask, float("-inf"))

        # --- 6. Compute Loss ---
        loss_q = F.cross_entropy(logits_masked, labels)

        return Ls + loss_q
