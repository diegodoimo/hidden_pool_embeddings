# class GatedMultiHeadAttention(nn.Module):
#     def __init__(self, d_model: int, d_attn: int, n_heads: int):
#         """
#         Initializes the Gated Multi-Head Attention mechanism.

#         Args:
#             d_model (int): The dimension of the hidden representations (d).
#             d_attn (int): The total dimension of the attention space (L).
#             n_heads (int): The number of attention heads (H).
#         """
#         super().__init__()

#         if d_attn % n_heads != 0:
#             raise ValueError("d_attn (L) must be divisible by n_heads (H)")

#         self.d_model = d_model
#         self.d_attn = d_attn
#         self.n_heads = n_heads
#         self.d_head = d_attn // n_heads  # L_h = L / H

#         # We use a single large linear layer and split the output for efficiency.
#         # This computes (V_1...V_H) * h_k_T and (U_1...U_H) * h_k_T
#         # Shape: (L x d) for each, so (L x d) combined.
#         self.V = nn.Linear(d_model, d_attn, bias=False)  # V: L x d
#         self.U = nn.Linear(d_model, d_attn, bias=False)  # U: L x d

#         # w is now concatenated for all heads: (L_h * H) x 1 = L x 1
#         self.w = nn.Parameter(torch.randn(d_attn, 1))

#         # Final output projection layer W_O: (L) x d
#         # This projects the concatenated context vectors back to the d_model dimension.
#         self.W_O = nn.Linear(d_attn, d_model, bias=False)

#         # Initialization
#         nn.init.xavier_uniform_(self.V.weight)
#         nn.init.xavier_uniform_(self.U.weight)
#         nn.init.xavier_uniform_(self.W_O.weight)

#     def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Computes the final context vector z using Multi-Head Gated Attention.

#         Args:
#             H (torch.Tensor): The hidden representations, of shape (K, d).
#                               K is the number of instances, d is d_model.

#         Returns:
#             tuple[torch.Tensor, torch.Tensor]: The final context vector z (1, d) and
#                                                the attention weights a_k (K, n_heads).
#         """
#         K, d = H.shape

#         # 1. Compute V*h_k_T and U*h_k_T for all heads simultaneously
#         # V_H_T and U_H_T are shape (K, L)
#         V_H_T = self.V(H)
#         U_H_T = self.U(H)

#         # 2. Reshape to split into heads: (K, H, L_h)
#         # This explicitly separates the L_h dimension for each head.
#         V_H_T_heads = V_H_T.view(K, self.n_heads, self.d_head)
#         U_H_T_heads = U_H_T.view(K, self.n_heads, self.d_head)

#         # 3. Apply non-linearities and gating (element-wise)
#         # Gating_Mechanism: shape (K, H, L_h)
#         Gating_Mechanism = torch.tanh(V_H_T_heads) * torch.sigmoid(U_H_T_heads)

#         # 4. Reshape w for matrix multiplication with heads: (L, 1) -> (H, L_h, 1)
#         w_heads = self.w.view(self.n_heads, self.d_head, 1)

#         # 5. Compute Attention Scores (E_k_logits) for all heads
#         # (K, H, L_h) @ (H, L_h, 1) -> This requires batched matmul over L_h
#         # We need to compute: (K, H, L_h) @ (H, L_h, 1)
#         # E_k_logits: shape (K, H, 1) - attention scores for each instance and each head
#         E_k_logits = torch.matmul(Gating_Mechanism, w_heads.transpose(0, 1))

#         # Squeeze the last dimension: (K, H)
#         E_k_logits = E_k_logits.squeeze(-1)

#         # 6. Normalization (Softmax)
#         # attention_weights (a_k): shape (K, H)
#         attention_weights = F.softmax(E_k_logits, dim=0)

#         # 7. Weighted Sum (Context Vectors z_i)
#         # H is (K, d). We need to tile/repeat H for H times: (K, H, d)
#         H_tiled = H.unsqueeze(1).expand(-1, self.n_heads, -1)

#         # attention_weights is (K, H). Expand to (K, H, 1) for broadcasting
#         attn_weights_expanded = attention_weights.unsqueeze(-1)

#         # (K, H, 1) * (K, H, d) -> (K, H, d)
#         weighted_H_heads = attn_weights_expanded * H_tiled

#         # Sum over instances (K): (K, H, d) -> (H, d)
#         z_heads = torch.sum(weighted_H_heads, dim=0)

#         # 8. Concatenate and Final Projection
#         # Concatenate: (H, d) -> (H * d) = (L)
#         z_concatenated = z_heads.view(1, -1)  # Flatten to (1, L)

#         # Final Projection: (1, L) @ (L, d) -> (1, d)
#         z_final = self.W_O(z_concatenated)

#         return z_final, attention_weights


# class GatedAttention(nn.Module):
#     def __init__(self, d_model: int, d_attn: int):
#         """
#         Initializes the Gated Attention mechanism.

#         Args:
#             d_model (int): The dimension of the hidden representations (d).
#             d_attn (int): The dimension of the attention space (L).
#         """
#         super().__init__()

#         # Initialize the projection matrices V and U (d_attn x d_model)
#         self.V = nn.Linear(d_model, d_attn, bias=False)
#         self.U = nn.Linear(d_model, d_attn, bias=False)

#         # Initialize the aggregation vector w (d_attn x 1)
#         self.w = nn.Parameter(torch.randn(d_attn, 1))

#         nn.init.xavier_uniform_(self.V.weight)
#         nn.init.xavier_uniform_(self.U.weight)

#     def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Computes the attention weights a_k and the final context vector z.

#         Args:
#             H (torch.Tensor): The hidden representations, of shape (B x K, d).
#                               B is the batch size, K is the number of instances, d is d_model.

#         Returns:
#             tuple[torch.Tensor, torch.Tensor]: The context vector z (1, d) and
#                                                the attention weights a_k (K, 1).
#         """
#         # --- 1. Gated Attention Score Calculation ---
#         V_H_T = self.V(H)  # [B x k x d_attn]
#         U_H_T = self.U(H)  # [B x K  x d_attn]

#         # Gating_Mechanism: tanh(V*h_k_T) ⊙ sigm(U*h_k_T), shape (B x K, d_attn)
#         Gating_Mechanism = torch.tanh(V_H_T) * torch.sigmoid(U_H_T)

#         # E_k_logits: Unnormalized attention scores w^T * Gating_Mechanism_T, shape (B x K, 1)
#         E_k_logits = Gating_Mechanism @ self.w

#         # --- 2. Normalization (Softmax) ---
#         # attention_weights (a_k): shape (B x K x  1)
#         attention_weights = F.softmax(E_k_logits, dim=1)

#         # --- 3. Weighted Sum (Context Vector z) ---
#         # Equation: z = sum(a_k * h_k)

#         # Step 3a: Element-wise multiplication: a_k * h_k.
#         # (K, 1) * (K, d) uses broadcasting to yield (K, d).
#         weighted_H = attention_weights * H

#         # Step 3b: Summation over the instances (K) dimension (dim=0).
#         # This collapses the (K, d) tensor to (1, d).
#         z = torch.sum(weighted_H, dim=0, keepdim=True)

#         return z, attention_weights


# # for group query attention
# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(
#         batch, num_key_value_heads, n_rep, slen, head_dim
#     )
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
