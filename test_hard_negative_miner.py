from sentence_transformers.util import mine_hard_negatives
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
# Load a Sentence Transformer model

model = SentenceTransformer("Qwen")

# Load a dataset to mine hard negatives from
dataset = load_dataset("sentence-transformers/natural-questions", split="train")
dataset
Dataset({
    features: ['query', 'answer'],
    num_rows: 100231
})
dataset = mine_hard_negatives(
    dataset=dataset,
    model=model,
    range_min=10,
    range_max=50,
    max_score=0.8,
    relative_margin=0.05,
    num_negatives=5,
    sampling_strategy="random",
    batch_size=128,
   