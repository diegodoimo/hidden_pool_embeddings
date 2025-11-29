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


class EmbeddingGemmaLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def pairwise_dot_squared(x):
        B = x.size(0)
        dots = x @ x.t()
        dots_sq = dots**2
        # Zero out diagonal in-place (most efficient)
        dots_sq.fill_diagonal_(0)
        return dots_sq.sum() / (B * (B - 1))

    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor):

        # shperical loss
        Ls = self.pairwise_dot_squared(query_embeddings) + self.pairwise_dot_squared(doc_embeddings)

        # contrastive loss


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
        Ls = self.pairwise_dot_squared(all_queries, B = batch_size) + self.pairwise_dot_squared(all_docs, B = batch_size)

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


def contrastive_loss_local(query_features, doc_features, hard_neg_features, temperature=0.07):
    """
    Local contrastive loss (only in-batch negatives from this GPU).
    Suitable when you have strong hard negatives.

    Args:
        query_features: [local_batch_size, feature_dim]
        doc_features: [local_batch_size, feature_dim]
        hard_neg_features: [local_batch_size, num_hard_negs, feature_dim]
        temperature: temperature scaling

    Returns:
        loss: scalar loss value
    """
    assert False, "make it consistent with the class above"
    local_batch_size = query_features.shape[0]

    # Normalize features
    query_features = F.normalize(query_features, dim=1)
    doc_features = F.normalize(doc_features, dim=1)
    if hard_neg_features is not None:
        hard_neg_features = F.normalize(hard_neg_features, dim=2)

    # Positive scores
    pos_scores = torch.sum(query_features * doc_features, dim=1, keepdim=True)  # [B, 1]

    # Local in-batch negative scores
    inbatch_scores = query_features @ doc_features.T  # [B, B]

    # Combine scores
    if hard_neg_features is not None:
        # Hard negative scores
        hard_neg_scores = torch.bmm(
            query_features.unsqueeze(1), hard_neg_features.transpose(1, 2)
        ).squeeze(
            1
        )  # [B, num_hard_negs]

        all_scores = torch.cat([pos_scores, hard_neg_scores, inbatch_scores], dim=1)
    else:
        all_scores = torch.cat([pos_scores, inbatch_scores], dim=1)

    # Apply temperature
    all_scores = all_scores / temperature

    # Labels: positive at index 0
    labels = torch.zeros(local_batch_size, dtype=torch.long, device=query_features.device)

    loss = F.cross_entropy(all_scores, labels)

    return loss
