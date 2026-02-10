# Classification and Clustering Loaders - Implementation Summary

## Overview

Successfully implemented classification and clustering loaders according to the specified requirements from Lee et al. (2025a) and related papers.

## What Was Implemented

### 1. Binary Classification Tasks (New Folder)

**Location:** `tasks/binary_classification_tasks/`

**Datasets Implemented:**
- Amazon Counterfactual (O'Neill et al., 2021)
- Amazon Polarity (McAuley & Leskovec, 2013)
- IMDb sentiment classification (Maas et al., 2011)
- Toxic Conversations (cjadams et al., 2019)
- CoLA (Warstadt et al., 2019)

**Two Loading Strategies:**

1. **Label-Based Approach (Default):**
   - Loader: `load_binary_classification_label_based()`
   - Each input text is a query
   - Label text (e.g., "toxic") is the positive passage
   - Other class label text (e.g., "not toxic") is a hard negative
   - Follows Lee et al. (2025a) specification

2. **Hard Negative Mining Approach (Optional):**
   - Loader: `load_binary_classification_hard_negatives()`
   - Creates corpus of all texts
   - Texts with same label are positives
   - Allows mining hard negatives from opposite class
   - Enabled via `use_hard_negative_mining = True`

### 2. Multi-way Classification Tasks (Updated)

**Location:** `tasks/classification_tasks/`

**Datasets:**
- Banking77 (77 intent classes)
- DBPedia (14 topic classes)

**Loading Strategy:**
- Loader: `load_multiway_classification_sampling()`
- Random sample from same class as positive
- All texts in corpus for hard negative mining
- Follows Lee et al. (2025a): 24 samples from other classes as hard negatives

### 3. Clustering Tasks (Updated)

**Location:** `tasks/clustering_tasks/`

**All 18 Clustering Datasets Updated:**
- Amazon Reviews, Emotion, MTOP Intent/Domain
- Massive Intent/Scenario, Tweet Sentiment
- ArXiv/BioRxiv/MedRxiv Clustering (P2P and S2S)
- Reddit/StackExchange Clustering (P2P and S2S)
- Twenty Newsgroups

**Loading Strategy:**
- Loader: `load_clustering_sampling()`
- Random sample from same cluster as positive
- All texts in corpus for hard negative mining
- Follows Lee et al. (2025a): 24 samples from other clusters as hard negatives

## Key Features

### Flexible Loading
All tasks support switching between strategies via `use_hard_negative_mining` flag:

```python
# Default: label-based for binary, sampling for multi-way/clustering
task = ToxicConversations50k()
data = task.loader(task)

# Enable hard negative mining
task.use_hard_negative_mining = True
data = task.loader(task)
```

### Consistent Interface
All loaders return `RetrievalRawData` objects for compatibility with retrieval training pipelines:
- `query_texts` / `query_ids`: Queries (may have duplicates)
- `positive_texts` / `positive_ids`: Positive passages for each query
- `document_texts` / `document_ids`: Full corpus for mining
- `unique_query_texts` / `unique_query_ids`: Deduplicated queries
- `unique_positive_texts` / `unique_positive_ids`: Deduplicated positives
- `corpus_dict`: Dictionary format corpus

### Proper Deduplication
- Unique query IDs maintained across duplicate texts
- Unique positive texts tracked separately from corpus
- Efficient corpus building with deduplication

## File Structure

```
tasks/
├── binary_classification_tasks/          # NEW
│   ├── __init__.py
│   ├── binary_classification_loaders.py  # NEW
│   └── data_loaders/
│       ├── amazoncounterfactualclassification.py
│       ├── amazonpolarityclassification.py
│       ├── colaclassification.py
│       ├── imdbclassification.py
│       └── toxicconversations50k.py
│
├── classification_tasks/                  # UPDATED
│   ├── __init__.py                       # Updated
│   ├── classification_loaders.py         # Updated with new loaders
│   └── data_loaders/
│       ├── banking77classification.py    # Updated
│       └── dbpediaclassification.py      # Updated
│
├── clustering_tasks/                      # UPDATED
│   ├── __init__.py
│   ├── clustering_loaders.py             # Updated with new loaders
│   └── data_loaders/                     # All 18 files updated
│       ├── amazonreviewsclustering.py
│       ├── emotionclustering.py
│       └── ... (16 more files)
│
└── __init__.py                           # Updated with new task categories
```

## Task Registry Updates

**In `tasks/__init__.py`:**

```python
# New category
BINARY_CLASSIFICATION_TASKS = [
    "toxic_conversations",
    "amazon_counterfactual",
    "amazon_polarity",
    "imdb",
    "cola",
]

# Updated category (removed binary tasks)
CLASSIFICATION_TASKS = [
    "dbpedia",
    "banking77",
]

# Existing category (all tasks updated)
CLUSTERING_TASKS = [
    # ... 18 tasks
]
```

## Usage Examples

### Binary Classification - Label-Based

```python
from tasks.binary_classification_tasks import ToxicConversations50k

task = ToxicConversations50k()
# task.use_hard_negative_mining = False (default)
data = task.loader(task, rank=0)

# Output:
# - query_texts: ["This is a toxic comment", ...]
# - positive_texts: ["toxic", "not toxic", ...]  # Label texts
# - document_texts: ["toxic", "not toxic"]  # Corpus of label texts
```

### Binary Classification - Hard Negative Mining

```python
from tasks.binary_classification_tasks import ImdbClassification

task = ImdbClassification()
task.use_hard_negative_mining = True
data = task.loader(task, rank=0)

# Output:
# - query_texts: ["Great movie!", "Terrible film", ...]
# - positive_texts: ["Excellent!", "Awful movie", ...]  # Same label texts
# - document_texts: [all movie reviews]  # Full corpus for mining
```

### Multi-way Classification

```python
from tasks.classification_tasks import Banking77Classification

task = Banking77Classification()
data = task.loader(task, rank=0)

# Output:
# - query_texts: ["How do I transfer money?", ...]
# - positive_texts: ["Can I send funds?", ...]  # Same class
# - document_texts: [all banking queries]  # Full corpus for mining
```

### Clustering

```python
from tasks.clustering_tasks import EmotionClustering

task = EmotionClustering()
data = task.loader(task, rank=0)

# Output:
# - query_texts: ["I am so happy!", ...]
# - positive_texts: ["This is joyful", ...]  # Same cluster
# - document_texts: [all emotion texts]  # Full corpus for mining
```

## Testing

All implementations tested and verified:
- ✓ All tasks can be instantiated
- ✓ Loaders return correct format
- ✓ Switching between strategies works
- ✓ No linter errors
- ✓ Imports work correctly
- ✓ Task registry updated properly

## Documentation

Created comprehensive documentation:
- `CLASSIFICATION_CLUSTERING_README.md`: Detailed usage guide
- `IMPLEMENTATION_SUMMARY.md`: This file

## Compliance with Requirements

✓ **Binary Classification:** Label-based approach as specified  
✓ **Multi-way Classification:** Random sampling from same class  
✓ **Clustering:** Random sampling from same cluster  
✓ **Hard Negative Mining:** Optional for all task types  
✓ **Separate Folder:** Binary tasks in dedicated folder  
✓ **Consistent Structure:** Same structure as other task folders  
✓ **Flexible Options:** Can switch between label-based and mining approaches

## References

- Lee et al. (2025a): Multi-way classification and clustering sampling strategy
- O'Neill et al. (2021): Amazon Counterfactual dataset
- McAuley & Leskovec (2013): Amazon Polarity dataset
- Maas et al. (2011): IMDb sentiment classification
- cjadams et al. (2019): Toxic Conversations dataset
- Warstadt et al. (2019): CoLA dataset

## Next Steps

The implementation is complete and ready to use. To use these loaders in training:

1. Import the desired task class
2. Optionally set `use_hard_negative_mining = True` for mining approach
3. Call `task.loader(task, rank=0)` to get the data
4. Use the returned `RetrievalRawData` object in your training pipeline

The loaders are compatible with existing retrieval training code and follow the same data format conventions.
