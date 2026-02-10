# Classification and Clustering Loaders Implementation

This document describes the implementation of classification and clustering loaders according to the specified requirements.

## Overview

The implementation divides classification tasks into two categories:
1. **Binary Classification Tasks** - in `binary_classification_tasks/` folder
2. **Multi-way Classification Tasks** - in `classification_tasks/` folder
3. **Clustering Tasks** - in `clustering_tasks/` folder (updated)

## Binary Classification Tasks

### Location
`tasks/binary_classification_tasks/`

### Datasets
- Amazon Counterfactual (O'Neill et al., 2021)
- Amazon Polarity (McAuley & Leskovec, 2013)
- IMDb sentiment classification (Maas et al., 2011)
- Toxic Conversations (cjadams et al., 2019)
- CoLA (Warstadt et al., 2019)

### Loading Strategies

#### 1. Label-Based Approach (Default)
**Loader:** `load_binary_classification_label_based()`

Following Lee et al. (2025a), this approach:
- Treats each input text as a query
- Uses the label text (e.g., "toxic") as the positive passage
- Uses the other class's label text (e.g., "not toxic") as a hard negative

**Example:**
```python
class ToxicConversations50k(AbsTask):
    label_texts = {
        0: "not toxic",
        1: "toxic"
    }
    use_hard_negative_mining = False  # Use label-based approach
```

#### 2. Hard Negative Mining Approach (Optional)
**Loader:** `load_binary_classification_hard_negatives()`

This approach:
- Creates a corpus of all texts in the dataset
- For each text, selects a random text with the same label as positive
- Allows hard negative mining from texts with different labels during training

**Usage:**
```python
task = ToxicConversations50k()
task.use_hard_negative_mining = True  # Enable hard negative mining
```

### Task Configuration

Each binary classification task has:
- `label_texts`: Dictionary mapping label values to text descriptions
- `use_hard_negative_mining`: Boolean flag to switch between approaches
- `loader`: Property that returns the appropriate loader based on the flag

## Multi-way Classification Tasks

### Location
`tasks/classification_tasks/`

### Datasets
- Banking77 (77 intent classes)
- DBPedia (14 topic classes)

### Loading Strategy

**Loader:** `load_multiway_classification_sampling()`

Following Lee et al. (2025a), for each query:
- A random sample from the same class is used as its positive passage
- 24 samples from other classes can be selected as hard negatives during training

The loader:
- Groups texts by class/label
- For each text, randomly selects another text from the same class as positive
- Includes all texts in the corpus for hard negative mining

**Example:**
```python
class Banking77Classification(AbsTask):
    use_hard_negative_mining = False  # Default: use sampling strategy
    
    @property
    def loader(self):
        if self.use_hard_negative_mining:
            return load_multiway_classification_hard_negatives
        else:
            return load_multiway_classification_sampling
```

## Clustering Tasks

### Location
`tasks/clustering_tasks/`

### Datasets
All clustering datasets including:
- Amazon Reviews
- Emotion
- MTOP Intent/Domain
- Massive Intent/Scenario
- Tweet Sentiment
- ArXiv/BioRxiv/MedRxiv Clustering (P2P and S2S)
- Reddit/StackExchange Clustering (P2P and S2S)
- Twenty Newsgroups

### Loading Strategy

**Loader:** `load_clustering_sampling()`

Following Lee et al. (2025a), for each query:
- A random sample from the same cluster is used as its positive passage
- 24 samples from other clusters can be selected as hard negatives during training

The loader:
- Groups texts by cluster/label
- For each text, randomly selects another text from the same cluster as positive
- Includes all texts in the corpus for hard negative mining

**Example:**
```python
class EmotionClustering(AbsTask):
    use_hard_negative_mining = False  # Default: use sampling strategy
    
    @property
    def loader(self):
        if self.use_hard_negative_mining:
            return load_clustering_hard_negatives
        else:
            return load_clustering_sampling
```

## Data Format

All loaders return `RetrievalRawData` objects with the following structure:

```python
@dataclass
class RetrievalRawData:
    query_texts: List[str]           # All query texts (may have duplicates)
    query_ids: List[str]             # Query IDs (unique per text)
    positive_texts: List[str]        # Positive passages for each query
    positive_ids: List[str]          # IDs of positive passages
    document_texts: List[str]        # Corpus of all documents
    document_ids: List[str]          # IDs of all documents
    unique_query_texts: List[str]    # Unique query texts
    unique_query_ids: List[str]      # Unique query IDs
    unique_positive_texts: List[str] # Unique positive texts
    unique_positive_ids: List[str]   # Unique positive IDs
    corpus_dict: Dict[str, Dict]     # Corpus as dictionary
    has_title: bool                  # Whether documents have titles
```

## Usage Examples

### Binary Classification with Label-Based Approach

```python
from tasks.binary_classification_tasks import ToxicConversations50k

task = ToxicConversations50k()
data = task.loader(task, rank=0)

# data.query_texts contains the input texts
# data.positive_texts contains label texts ("toxic" or "not toxic")
# data.document_texts contains all label texts (corpus for mining)
```

### Binary Classification with Hard Negative Mining

```python
from tasks.binary_classification_tasks import ToxicConversations50k

task = ToxicConversations50k()
task.use_hard_negative_mining = True
data = task.loader(task, rank=0)

# data.query_texts contains the input texts
# data.positive_texts contains texts with same label
# data.document_texts contains all texts (corpus for mining)
```

### Multi-way Classification

```python
from tasks.classification_tasks import Banking77Classification

task = Banking77Classification()
data = task.loader(task, rank=0)

# data.query_texts contains the input texts
# data.positive_texts contains texts from same class
# data.document_texts contains all texts (corpus for mining)
```

### Clustering

```python
from tasks.clustering_tasks import EmotionClustering

task = EmotionClustering()
data = task.loader(task, rank=0)

# data.query_texts contains the input texts
# data.positive_texts contains texts from same cluster
# data.document_texts contains all texts (corpus for mining)
```

## Task Registry

All tasks are registered in `tasks/__init__.py`:

```python
# Binary Classification tasks
BINARY_CLASSIFICATION_TASKS = [
    "toxic_conversations",
    "amazon_counterfactual",
    "amazon_polarity",
    "imdb",
    "cola",
]

# Multi-way Classification tasks
CLASSIFICATION_TASKS = [
    "dbpedia",
    "banking77",
]

# Clustering tasks
CLUSTERING_TASKS = [
    "amazon_reviews",
    "emotion",
    # ... (all clustering tasks)
]
```

## Key Features

1. **Flexible Loading**: All tasks support both label-based and hard negative mining approaches
2. **Consistent Interface**: All loaders return `RetrievalRawData` for compatibility with retrieval training pipelines
3. **Proper Deduplication**: Unique queries and positives are tracked separately
4. **Corpus Building**: All texts are included in corpus for hard negative mining
5. **Random Sampling**: Positives are randomly sampled from same class/cluster for each query

## References

- Lee et al. (2025a): Multi-way classification and clustering sampling strategy
- O'Neill et al. (2021): Amazon Counterfactual dataset
- McAuley & Leskovec (2013): Amazon Polarity dataset
- Maas et al. (2011): IMDb sentiment classification
- cjadams et al. (2019): Toxic Conversations dataset
- Warstadt et al. (2019): CoLA dataset
