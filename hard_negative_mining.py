"""
Complete pipeline for MS-MARCO corpus search with FAISS and result storage.
Efficiently processes MS-MARCO training set (500k+ queries, 8.8M passages).
"""

import faiss
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle
import h5py
from tqdm import tqdm
from datasets import load_dataset
import torch
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MSMARCOFAISSPipeline:
    """End-to-end pipeline for MS-MARCO corpus search with FAISS."""

    def __init__(
        self,
        model,
        output_dir: str = "./msmarco_results",
        corpus_chunk_size: int = 50000,
        query_batch_size: int = 10000,
        use_gpu: bool = True,
        storage_format: str = "hdf5",  # "hdf5", "jsonl", "pickle", or "npz"
    ):
        """
        Args:
            model: Your embedding model
            output_dir: Directory to save results
            corpus_chunk_size: Batch size for corpus encoding
            query_batch_size: Batch size for query processing
            use_gpu: Use GPU for FAISS if available
            storage_format: Format for storing results
                - "hdf5": Memory efficient, fast random access (RECOMMENDED)
                - "jsonl": Human readable, streaming friendly
                - "pickle": Fast, good for small-medium datasets
                - "npz": Separate arrays per query (memory efficient)
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.corpus_chunk_size = corpus_chunk_size
        self.query_batch_size = query_batch_size
        self.use_gpu = use_gpu
        self.storage_format = storage_format

        self.index = None
        self.corpus_ids = []

    def load_msmarco_data(self):
        """Load MS-MARCO training data."""
        logger.info("Loading MS-MARCO dataset...")

        # Load corpus (passages)
        corpus = load_dataset("ms_marco", "v1.1", split="corpus")
        logger.info(f"Loaded {len(corpus)} corpus passages")

        # Load training queries
        queries = load_dataset("ms_marco", "v1.1", split="train")
        logger.info(f"Loaded {len(queries)} training queries")

        return corpus, queries

    def build_faiss_index(self, corpus, embedding_dim: Optional[int] = None):
        """Build FAISS index from corpus."""
        logger.info("Building FAISS index...")

        corpus_size = len(corpus)

        # Determine embedding dimension if not provided
        if embedding_dim is None:
            logger.info("Encoding sample to determine embedding dimension...")
            sample = corpus.select([0])
            sample_emb = self._encode_texts(sample["text"])
            embedding_dim = sample_emb.shape[1]
            logger.info(f"Embedding dimension: {embedding_dim}")

        # Create index based on corpus size
        if corpus_size < 100000:
            index = faiss.IndexFlatIP(embedding_dim)
            logger.info("Using Flat index")
        elif corpus_size < 1000000:
            nlist = min(4096, int(np.sqrt(corpus_size)))
            quantizer = faiss.IndexFlatIP(embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            logger.info(f"Using IVF-Flat with {nlist} clusters")
        else:
            # For MS-MARCO (8.8M docs): IVF-PQ
            nlist = 8192
            m = 64  # subquantizers
            nbits = 8
            quantizer = faiss.IndexFlatIP(embedding_dim)
            index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, nbits)
            logger.info(f"Using IVF-PQ with {nlist} clusters")

        # Train index if needed
        if isinstance(index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)):
            logger.info("Training index...")
            train_size = min(10 * 5, corpus_size // 10)
            train_embeddings = self._encode_corpus_sample(corpus, train_size)
            index.train(train_embeddings)
            logger.info("Training complete")

        logger.info("Moving index to GPU...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

        # Add corpus embeddings
        logger.info("Adding corpus to index...")
        for start_idx in tqdm(
            range(0, corpus_size, self.corpus_chunk_size), desc="Encoding corpus"
        ):
            end_idx = min(start_idx + self.corpus_chunk_size, corpus_size)
            batch = corpus.select(range(start_idx, end_idx))

            embeddings = self._encode_texts(batch["text"])
            index.add(embeddings)
            self.corpus_ids.extend(
                batch["id"] if "id" in batch.column_names else range(start_idx, end_idx)
            )

        logger.info(f"Index built with {index.ntotal} vectors")
        self.index = index

        # Save index
        self._save_index()

        return index

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        # Adjust this based on your model's API
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_numpy=True)

        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        return embeddings

    def _encode_corpus_sample(self, corpus, sample_size: int) -> np.ndarray:
        """Encode a random sample of corpus for training."""
        indices = np.random.choice(len(corpus), min(sample_size, len(corpus)), replace=False)
        embeddings_list = []

        for i in tqdm(range(0, len(indices), self.corpus_chunk_size), desc="Training sample"):
            batch_indices = indices[i : i + self.corpus_chunk_size]
            batch = corpus.select(batch_indices.tolist())
            emb = self._encode_texts(batch["text"])
            embeddings_list.append(emb)

        return np.vstack(embeddings_list)

    def search_queries(self, queries, top_k: int = 100, nprobe: int = 128):
        """Search all queries and return top-k results."""
        if self.index is None:
            raise ValueError("Index not built. Call build_faiss_index first.")

        # Set nprobe for IVF indices
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe
            logger.info(f"Searching with nprobe={nprobe}")

        num_queries = len(queries)
        logger.info(f"Searching {num_queries} queries...")

        all_results = {}

        # Process queries in batches
        for batch_start in tqdm(
            range(0, num_queries, self.query_batch_size), desc="Searching queries"
        ):
            batch_end = min(batch_start + self.query_batch_size, num_queries)
            batch = queries.select(range(batch_start, batch_end))

            # Encode queries
            query_embeddings = self._encode_texts(batch["query"])

            # Search
            scores, indices = self.index.search(query_embeddings, top_k)

            # Store results
            for i, query_idx in enumerate(range(batch_start, batch_end)):
                query_id = (
                    batch["query_id"][i] if "query_id" in batch.column_names else str(query_idx)
                )

                # Filter out invalid results (-1 indices)
                valid_mask = indices[i] >= 0
                valid_indices = indices[i][valid_mask]
                valid_scores = scores[i][valid_mask]

                all_results[query_id] = {
                    "query": batch["query"][i],
                    "doc_ids": [self.corpus_ids[idx] for idx in valid_indices],
                    "scores": valid_scores.tolist(),
                }

        return all_results

    def save_results(self, results: Dict, filename: str = "search_results"):
        """Save search results in specified format."""
        output_path = self.output_dir / filename

        logger.info(f"Saving results in {self.storage_format} format...")

        if self.storage_format == "hdf5":
            self._save_hdf5(results, f"{output_path}.h5")
        elif self.storage_format == "jsonl":
            self._save_jsonl(results, f"{output_path}.jsonl")
        elif self.storage_format == "pickle":
            self._save_pickle(results, f"{output_path}.pkl")
        elif self.storage_format == "npz":
            self._save_npz(results, output_path)
        else:
            raise ValueError(f"Unknown storage format: {self.storage_format}")

        logger.info(f"Results saved to {output_path}")

    def _save_hdf5(self, results: Dict, filepath: str):
        """Save to HDF5 format (RECOMMENDED for large datasets)."""
        with h5py.File(filepath, "w") as f:
            for query_id, data in tqdm(results.items(), desc="Saving to HDF5"):
                grp = f.create_group(str(query_id))
                grp.attrs["query"] = data["query"]
                grp.create_dataset("doc_ids", data=data["doc_ids"], compression="gzip")
                grp.create_dataset("scores", data=data["scores"], compression="gzip")

    def _save_jsonl(self, results: Dict, filepath: str):
        """Save to JSONL format (human readable)."""
        with open(filepath, "w") as f:
            for query_id, data in tqdm(results.items(), desc="Saving to JSONL"):
                record = {
                    "query_id": query_id,
                    "query": data["query"],
                    "doc_ids": data["doc_ids"],
                    "scores": data["scores"],
                }
                f.write(json.dumps(record) + "\n")

    def _save_pickle(self, results: Dict, filepath: str):
        """Save to pickle format (fast, not human readable)."""
        with open(filepath, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_npz(self, results: Dict, output_dir: Path):
        """Save as separate compressed numpy arrays."""
        npz_dir = output_dir / "npz"
        npz_dir.mkdir(exist_ok=True)

        # Save metadata
        metadata = {qid: data["query"] for qid, data in results.items()}
        with open(npz_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Save arrays
        for query_id, data in tqdm(results.items(), desc="Saving to NPZ"):
            np.savez_compressed(
                npz_dir / f"{query_id}.npz", doc_ids=data["doc_ids"], scores=data["scores"]
            )

    def _save_index(self):
        """Save FAISS index to disk."""
        index_path = self.output_dir / "faiss_index"

        # Move to CPU before saving
        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
        else:
            index_cpu = self.index

        faiss.write_index(index_cpu, str(index_path) + ".index")
        np.save(str(index_path) + "_ids.npy", np.array(self.corpus_ids))
        logger.info(f"Index saved to {index_path}")

    def load_index(self):
        """Load pre-built FAISS index."""
        index_path = self.output_dir / "faiss_index"

        self.index = faiss.read_index(str(index_path) + ".index")
        self.corpus_ids = np.load(str(index_path) + "_ids.npy").tolist()

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        logger.info("Index loaded successfully")

    def run_full_pipeline(self, top_k: int = 100, nprobe: int = 128):
        """Run complete pipeline: load data, build index, search, save results."""
        # Load data
        corpus, queries = self.load_msmarco_data()

        # Build or load index
        index_path = self.output_dir / "faiss_index.index"
        if index_path.exists():
            logger.info("Loading existing index...")
            self.load_index()
        else:
            logger.info("Building new index...")
            self.build_faiss_index(corpus)

        # Search queries
        results = self.search_queries(queries, top_k=top_k, nprobe=nprobe)

        # Save results
        self.save_results(results)

        logger.info("Pipeline complete!")
        return results


def load_results(filepath: str, format: str = "hdf5"):
    """Utility function to load saved results."""
    if format == "hdf5":
        results = {}
        with h5py.File(filepath, "r") as f:
            for query_id in f.keys():
                grp = f[query_id]
                results[query_id] = {
                    "query": grp.attrs["query"],
                    "doc_ids": grp["doc_ids"][:].tolist(),
                    "scores": grp["scores"][:].tolist(),
                }
        return results

    elif format == "jsonl":
        results = {}
        with open(filepath, "r") as f:
            for line in f:
                record = json.loads(line)
                results[record["query_id"]] = {
                    "query": record["query"],
                    "doc_ids": record["doc_ids"],
                    "scores": record["scores"],
                }
        return results

    elif format == "pickle":
        with open(filepath, "rb") as f:
            return pickle.load(f)

    else:
        raise ValueError(f"Unknown format: {format}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":

    # Initialize model (replace with your model)
    model = SentenceTransformer("")

    # Create pipeline
    pipeline = MSMARCOFAISSPipeline(
        model=model,
        output_dir="./msmarco_results",
        corpus_chunk_size=50000,
        query_batch_size=10000,
        use_gpu=True,
        storage_format="hdf5",  # Best for large datasets
    )

    # Run full pipeline
    results = pipeline.run_full_pipeline(
        top_k=100,
        nprobe=128,  # Adjust for accuracy/speed tradeoff
    )

    # Later, load results
    loaded_results = load_results("./msmarco_results/search_results.h5", format="hdf5")

    # Access results for a specific query
    query_id = list(loaded_results.keys())[0]
    print(f"Query: {loaded_results[query_id]['query']}")
    print(f"Top document IDs: {loaded_results[query_id]['doc_ids'][:5]}")
    print(f"Scores: {loaded_results[query_id]['scores'][:5]}")
