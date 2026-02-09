# Final Task Module Structure

## Overview

The task module is now fully modularized with **4 separate task categories**, each in its own folder:

```
tasks/
├── loaders.py                  # ⭐ Central loaders with common interface
├── retrieval_tasks/            # 🔍 35 retrieval tasks
├── sts_tasks/                  # 📊 3 STS tasks
├── classification_tasks/       # 🏷️  7 classification tasks
└── clustering_tasks/           # 🔗 18 clustering tasks
```

## Complete Structure

```
tasks/
├── loaders.py                          # CENTRAL LOADERS MODULE
│   ├── Retrieval loaders:
│   │   ├── from_one_hf_dataset
│   │   ├── from_multiple_hf_datasets
│   │   ├── from_multiple_hf_datasets_with_dedup
│   │   ├── load_nli_retrieval
│   │   └── load_sts_retrieval
│   └── Classification/Clustering loaders:
│       └── load_classification_standard
│
├── abs_task.py                         # Base task class
├── prompts.py                          # Task prompts
├── __init__.py                         # Main entry point (imports all)
│
├── retrieval_tasks/                    # 🔍 RETRIEVAL TASKS (35)
│   ├── __init__.py
│   ├── msmarco.py
│   ├── nfcorpus.py
│   ├── fever.py
│   ├── hotpotqa.py
│   ├── naturalquestions.py
│   ├── all_nli.py
│   ├── arguana.py
│   ├── snli.py
│   ├── mnli.py
│   ├── anli.py
│   ├── paq.py
│   ├── squad.py
│   ├── stackexchange.py
│   ├── eli5.py
│   ├── fiqa2018.py
│   ├── bioasq.py
│   ├── miracl.py
│   ├── mrtydi.py
│   ├── scifact.py
│   ├── triviaqa.py
│   ├── coliee.py
│   ├── pubmedqa.py
│   ├── s2orctitleabstract.py
│   ├── s2orctitlecitation.py
│   ├── s2orcabstractcitation.py
│   ├── specter.py
│   ├── xsum.py
│   ├── cnndm.py
│   ├── sentencecompression.py
│   ├── stackexchangedupquestionss2s.py
│   ├── stackexchangedupquestionsp2p.py
│   ├── qqp.py
│   ├── stackoverflow_dup.py
│   ├── amazonqa.py
│   └── msmarcov2.py
│
├── sts_tasks/                          # 📊 STS TASKS (3)
│   ├── __init__.py
│   ├── sts12.py
│   ├── sts22.py
│   └── stsbenchmark.py
│
├── classification_tasks/               # 🏷️  CLASSIFICATION TASKS (7)
│   ├── __init__.py
│   ├── dbpediaclassification.py
│   ├── toxicconversations50k.py
│   ├── banking77classification.py
│   ├── amazoncounterfactualclassification.py
│   ├── amazonpolarityclassification.py
│   ├── imdbclassification.py
│   └── colaclassification.py
│
└── clustering_tasks/                   # 🔗 CLUSTERING TASKS (18)
    ├── __init__.py
    ├── amazonreviewsclustering.py
    ├── emotionclustering.py
    ├── mtopintentclustering.py
    ├── mtopdomainclustering.py
    ├── massivescenarioclustering.py
    ├── massiveintentclustering.py
    ├── tweetsentimentextractionclustering.py
    ├── arxivclusteringp2p.py
    ├── arxivclusterings2s.py
    ├── biorxivclusteringp2p.py
    ├── biorxivclusterings2s.py
    ├── medrxivclusteringp2p.py
    ├── medrxivclusterings2s.py
    ├── redditclusteringp2p.py
    ├── redditclusterings2s.py
    ├── stackexchangeclusteringp2p.py
    ├── stackexchangeclusterings2s.py
    └── twentynewsgroupsclustering.py
```

## Task Distribution

| Category | Count | Location |
|----------|-------|----------|
| **Retrieval** | 35 | `tasks/retrieval_tasks/` |
| **STS** | 3 | `tasks/sts_tasks/` |
| **Classification** | 7 | `tasks/classification_tasks/` |
| **Clustering** | 18 | `tasks/clustering_tasks/` |
| **TOTAL** | **63** | - |

## Classification Tasks (7)

1. **DBPediaClassification** - Categorize wiki passages
2. **ToxicConversations50k** - Identify toxic conversations
3. **Banking77Classification** - Banking intent classification
4. **AmazonCounterfactualClassification** - Counterfactual detection
5. **AmazonPolarityClassification** - Sentiment polarity
6. **ImdbClassification** - Movie review sentiment
7. **ColaClassification** - Linguistic acceptability

## Clustering Tasks (18)

### E-commerce & Reviews
1. **AmazonReviewsClustering** - Product category clustering

### Intent & Emotion
2. **EmotionClustering** - Emotion detection
3. **MTOPIntentClustering** - Task-oriented intent
4. **MTOPDomainClustering** - Task-oriented domain
5. **MassiveScenarioClustering** - Scenario clustering
6. **MassiveIntentClustering** - Intent clustering
7. **TweetSentimentExtractionClustering** - Tweet sentiment

### Scientific Papers
8. **ArxivClusteringP2P** - ArXiv paper clustering (passage-to-passage)
9. **ArxivClusteringS2S** - ArXiv paper clustering (sentence-to-sentence)
10. **BiorxivClusteringP2P** - BioRxiv paper clustering (P2P)
11. **BiorxivClusteringS2S** - BioRxiv paper clustering (S2S)
12. **MedrxivClusteringP2P** - MedRxiv paper clustering (P2P)
13. **MedrxivClusteringS2S** - MedRxiv paper clustering (S2S)

### Social Media & Forums
14. **RedditClusteringP2P** - Reddit post clustering (P2P)
15. **RedditClusteringS2S** - Reddit post clustering (S2S)
16. **StackExchangeClusteringP2P** - StackExchange Q&A clustering (P2P)
17. **StackExchangeClusteringS2S** - StackExchange Q&A clustering (S2S)

### News
18. **TwentyNewsgroupsClustering** - Newsgroup topic clustering

## Loader Distribution

### Shared Loaders (in `tasks/loaders.py`)

**Retrieval:**
- `from_one_hf_dataset` → 15+ tasks
- `from_multiple_hf_datasets` → 2 tasks
- `from_multiple_hf_datasets_with_dedup` → 7 tasks
- `load_nli_retrieval` → 3 tasks (SNLI, MNLI, ANLI)
- `load_sts_retrieval` → 3 tasks (STS12, STS22, STSBenchmark)

**Classification/Clustering:**
- `load_classification_standard` → 25 tasks (7 classification + 18 clustering)

### Task-Specific Loaders (co-located with tasks)

1. `load_arguana_dedup_retrieval` - Arguana deduplication
2. `load_squad_retrieval` - SQuAD context deduplication
3. `load_stackexchange_retrieval` - Combine title+body
4. `load_miracl_retrieval` - Multi-language support
5. `load_pubmedqa_retrieval` - Context processing
6. `load_xsum_retrieval` - Summarization dataset
7. `load_cnndm_retrieval` - CNN/DailyMail dataset
8. `load_stackoverflow_dup_retrieval` - Duplicate detection

## Unified Interface

**Every task** (retrieval, STS, classification, clustering) follows the same pattern:

```python
class TaskName(AbsTask):
    hf_name = "dataset/name"
    split = "train"
    # ... task configuration ...
    metadata = TaskMetadata(type="TaskType", prompt={...})
    loader = loader_function  # ⭐ Direct function reference
```

## Data Loading Flow

```python
# 1. Get task
task = get_task("banking77")  # or any of the 63 tasks

# 2. Load data - same interface for all!
data = load_task_data(task, rank=0)

# 3. Dispatches based on task.metadata.type:
#    - "Retrieval" → _load_retrieval_data()
#    - "Classification" → _load_classification_data()
#    - "Clustering" → _load_classification_data()

# 4. Calls task.loader() function
```

## Key Features

### ✅ Clean Separation
- Classification and Clustering are now separate folders
- Clear distinction between different task types
- Each category has its own namespace

### ✅ Uniform Interface
- All tasks have `loader` attribute
- No more `custom_loader` strings
- Direct function references throughout

### ✅ Scalability
- Easy to add new tasks in any category
- Shared loaders for common patterns
- Task-specific loaders when needed

### ✅ Organization
- 4 clear categories: retrieval, STS, classification, clustering
- Central loaders in `tasks/loaders.py`
- Small, focused files (~20-50 lines each)

## Benefits of Separation

### Before (Classification + Clustering together)
```
tasks/classification_tasks/  (25 tasks mixed)
├── Banking77Classification
├── AmazonReviewsClustering
├── EmotionClustering
├── ...
```

### After (Separated)
```
tasks/classification_tasks/  (7 pure classification)
├── Banking77Classification
├── ImdbClassification
└── ...

tasks/clustering_tasks/      (18 clustering)
├── AmazonReviewsClustering
├── EmotionClustering
└── ...
```

**Advantages:**
- ✨ Clearer semantics (classification ≠ clustering)
- 🎯 Easier to find specific task types
- 📁 Better organization by task purpose
- 🔧 Can have clustering-specific loaders in future
- 📊 Easier to analyze clustering vs classification separately

## Usage Example

```python
from tasks import get_task
from inference.load_datasets import load_task_data

# Retrieval task
retrieval_task = get_task("msmarco")
data, corpus_dict, has_title = load_task_data(retrieval_task)

# STS task
sts_task = get_task("sts12")
data, corpus_dict, has_title = load_task_data(sts_task)

# Classification task
class_task = get_task("banking77")
classification_data = load_task_data(class_task)  # ClassificationRawData

# Clustering task
cluster_task = get_task("emotion")
clustering_data = load_task_data(cluster_task)  # ClassificationRawData
```

## Summary

**Final structure:**
- 🔍 35 Retrieval tasks
- 📊 3 STS tasks
- 🏷️ 7 Classification tasks
- 🔗 18 Clustering tasks
- ⭐ 1 Central loaders module

**Total: 63 tasks organized in 4 categories with a unified interface!**
