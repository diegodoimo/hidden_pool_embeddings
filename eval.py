# """
# Custom MTEB evaluation with distributed PyTorch dataloader and compiled model.
# This implementation uses MTEB's exact dataset preparation and metric evaluation
# while handling the embedding generation independently.
# """

# import torch
# import torch.distributed as dist
# from torch.utils.data import Dataset, DataLoader, DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# import numpy as np
# from typing import List, Dict, Any, Optional, Union
# import mteb
# from mteb.abstasks.AbsTask import AbsTask
# from tqdm import tqdm
# import os


# class MTEBTextDataset(Dataset):
#     """Dataset wrapper for MTEB texts compatible with PyTorch DataLoader."""

#     def __init__(self, texts: List[str], tokenizer=None):
#         self.texts = texts
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         return self.texts[idx]


# class CustomMTEBModel:
#     """
#     Wrapper that makes your model compatible with MTEB's interface
#     while using distributed PyTorch for embedding generation.
#     """

#     def __init__(
#         self,
#         model: torch.nn.Module,
#         tokenizer,
#         batch_size: int = 32,
#         max_length: int = 512,
#         normalize_embeddings: bool = True,
#         use_compile: bool = True,
#         distributed: bool = True,
#     ):
#         """
#         Args:
#             model: Your PyTorch embedding model
#             tokenizer: Tokenizer compatible with your model
#             batch_size: Batch size per GPU
#             max_length: Maximum sequence length
#             normalize_embeddings: Whether to L2 normalize embeddings
#             use_compile: Whether to use torch.compile
#             distributed: Whether to use distributed training
#         """
#         self.tokenizer = tokenizer
#         self.batch_size = batch_size
#         self.max_length = max_length
#         self.normalize_embeddings = normalize_embeddings
#         self.distributed = distributed

#         # Setup distributed
#         if distributed and not dist.is_initialized():
#             dist.init_process_group(backend="nccl")

#         self.rank = dist.get_rank() if distributed and dist.is_initialized() else 0
#         self.world_size = dist.get_world_size() if distributed and dist.is_initialized() else 1
#         self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")

#         # Move model to device
#         self.model = model.to(self.device)

#         # Wrap with DDP if distributed
#         if distributed and self.world_size > 1:
#             self.model = DDP(self.model, device_ids=[self.rank])

#         # Compile model if requested
#         if use_compile:
#             self.model = torch.compile(self.model)

#         self.model.eval()

#     def encode(
#         self,
#         sentences: Union[List[str], str],
#         batch_size: Optional[int] = None,
#         show_progress_bar: bool = True,
#         **kwargs,
#     ) -> np.ndarray:
#         """
#         Encode sentences using distributed PyTorch.
#         This method signature matches MTEB's expected interface.

#         Args:
#             sentences: List of sentences to encode
#             batch_size: Override default batch size
#             show_progress_bar: Whether to show progress

#         Returns:
#             numpy array of embeddings with shape (n_sentences, embedding_dim)
#         """
#         if isinstance(sentences, str):
#             sentences = [sentences]

#         batch_size = batch_size or self.batch_size

#         # Create dataset and distributed sampler
#         dataset = MTEBTextDataset(sentences, self.tokenizer)

#         if self.distributed and self.world_size > 1:
#             sampler = DistributedSampler(
#                 dataset,
#                 num_replicas=self.world_size,
#                 rank=self.rank,
#                 shuffle=False,
#                 drop_last=False,
#             )
#         else:
#             sampler = None

#         dataloader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             sampler=sampler,
#             shuffle=False,
#             num_workers=4,
#             pin_memory=True,
#             collate_fn=self._collate_fn,
#         )

#         all_embeddings = []

#         # Only show progress bar on rank 0
#         iterator = (
#             tqdm(dataloader, desc="Encoding")
#             if show_progress_bar and self.rank == 0
#             else dataloader
#         )

#         with torch.no_grad():
#             for batch in iterator:
#                 # Move batch to device
#                 batch = {k: v.to(self.device) for k, v in batch.items()}

#                 # Get embeddings from model
#                 embeddings = self._get_embeddings(batch)

#                 # Normalize if needed
#                 if self.normalize_embeddings:
#                     embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

#                 all_embeddings.append(embeddings.cpu())

#         # Concatenate all embeddings
#         all_embeddings = torch.cat(all_embeddings, dim=0)

#         # Gather from all processes if distributed
#         if self.distributed and self.world_size > 1:
#             gathered_embeddings = [torch.zeros_like(all_embeddings) for _ in range(self.world_size)]
#             dist.all_gather(gathered_embeddings, all_embeddings)
#             all_embeddings = torch.cat(gathered_embeddings, dim=0)

#         # Convert to numpy and trim to original length (in case of padding from distributed sampler)
#         embeddings_np = all_embeddings.numpy()[: len(sentences)]

#         return embeddings_np

#     def _collate_fn(self, batch: List[str]) -> Dict[str, torch.Tensor]:
#         """Tokenize batch of texts."""
#         encoded = self.tokenizer(
#             batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
#         )
#         return encoded

#     def _get_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """
#         Extract embeddings from model output.
#         Override this method if your model has a different output format.
#         """
#         # Default implementation for models that output embeddings directly
#         # or have a pooler_output attribute
#         outputs = self.model(**batch)

#         if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
#             return outputs.pooler_output
#         elif hasattr(outputs, "last_hidden_state"):
#             # Mean pooling over sequence
#             attention_mask = batch["attention_mask"]
#             token_embeddings = outputs.last_hidden_state
#             input_mask_expanded = (
#                 attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#             )
#             return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
#                 input_mask_expanded.sum(1), min=1e-9
#             )
#         else:
#             # Assume direct embedding output
#             return outputs if isinstance(outputs, torch.Tensor) else outputs[0]


# def evaluate_on_mteb(
#     model: CustomMTEBModel,
#     tasks: List[str],
#     output_folder: str = "results",
#     eval_splits: List[str] = ["test"],
#     **kwargs,
# ):
#     """
#     Evaluate model on MTEB tasks using MTEB's exact evaluation pipeline.
#     Only rank 0 performs metric evaluation and saves results to avoid duplication.

#     Args:
#         model: CustomMTEBModel instance
#         tasks: List of MTEB task names (e.g., ["STSBenchmark", "SICK"])
#         output_folder: Where to save results
#         eval_splits: Which splits to evaluate on
#     """

#     evaluation = mteb.MTEB(tasks=tasks)

#     # Run evaluation - MTEB will handle dataset preparation and metric calculation
#     # The model.encode() method will use all GPUs for embedding generation
#     results = evaluation.run(model, output_folder=output_folder, eval_splits=eval_splits, **kwargs)

#     return results


# # Example usage
# if __name__ == "__main__":
#     # Example: Load your model and tokenizer
#     from transformers import AutoModel, AutoTokenizer

#     model_name = "your-model-name"
#     model = AutoModel.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     # Wrap with custom MTEB model
#     mteb_model = CustomMTEBModel(
#         model=model,
#         tokenizer=tokenizer,
#         batch_size=32,
#         max_length=512,
#         normalize_embeddings=True,
#         use_compile=True,
#         distributed=True,
#     )

#     # Evaluate on MTEB tasks
#     tasks = ["STSBenchmark", "SICK", "STS12", "STS13", "STS14", "STS15", "STS16"]

#     results = evaluate_on_mteb(model=mteb_model, tasks=tasks, output_folder="mteb_results")

#     # Only rank 0 gets results
#     if mteb_model.rank == 0:
#         print("Evaluation complete!")
#         print(results)
