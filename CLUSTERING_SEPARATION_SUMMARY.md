# Clustering Tasks Separation Summary

## Change Overview

Separated **clustering tasks** from classification tasks into their own dedicated folder for better organization and clarity.

## What Changed

### Before
```
tasks/
├── classification_tasks/       # 25 tasks (mixed classification + clustering)
│   ├── Banking77Classification
│   ├── AmazonReviewsClustering    ← Clustering
│   ├── EmotionClustering          ← Clustering
│   ├── ImdbClassification
│   └── ...
```

### After
```
tasks/
├── classification_tasks/       # 7 pure classification tasks
│   ├── Banking77Classification
│   ├── ImdbClassification
│   └── ...
│
└── clustering_tasks/           # 18 clustering tasks
    ├── AmazonReviewsClustering
    ├── EmotionClustering
    └── ...
```

## Tasks Moved

**18 clustering tasks** moved from `classification_tasks/` to `clustering_tasks/`:

### E-commerce & Reviews (1)
1. AmazonReviewsClustering

### Intent & Emotion (6)
2. EmotionClustering
3. MTOPIntentClustering
4. MTOPDomainClustering
5. MassiveScenarioClustering
6. MassiveIntentClustering
7. TweetSentimentExtractionClustering

### Scientific Papers (6)
8. ArxivClusteringP2P
9. ArxivClusteringS2S
10. BiorxivClusteringP2P
11. BiorxivClusteringS2S
12. MedrxivClusteringP2P
13. MedrxivClusteringS2S

### Social Media & Forums (4)
14. RedditClusteringP2P
15. RedditClusteringS2S
16. StackExchangeClusteringP2P
17. StackExchangeClusteringS2S

### News (1)
18. TwentyNewsgroupsClustering

## Classification Tasks Remaining (7)

1. **DBPediaClassification** - Wiki passage categorization
2. **ToxicConversations50k** - Toxic conversation detection
3. **Banking77Classification** - Banking intent classification
4. **AmazonCounterfactualClassification** - Counterfactual detection
5. **AmazonPolarityClassification** - Sentiment polarity
6. **ImdbClassification** - Movie review sentiment
7. **ColaClassification** - Linguistic acceptability

## Files Modified

### Created
- `tasks/clustering_tasks/` directory
- `tasks/clustering_tasks/__init__.py` (18 task imports)
- Moved 18 task files from `classification_tasks/` to `clustering_tasks/`

### Updated
- `tasks/classification_tasks/__init__.py` (now imports only 7 tasks)
- `tasks/__init__.py` (added `from .clustering_tasks import *`)
- `tasks/clustering_tasks/redditclusteringp2p.py` (fixed type from "Classification" to "Clustering")

### Unchanged
- `tasks/loaders.py` - Still works for both classification and clustering
- `inference/load_datasets.py` - Already handles both types via `task.metadata.type`

## Why This Separation Matters

### 1. **Semantic Clarity**
- Classification: Assign labels to individual items
- Clustering: Group similar items together
- Different ML paradigms deserve separate organization

### 2. **Easier Navigation**
- Developers can quickly find clustering-specific tasks
- No need to search through mixed classification/clustering lists
- Clear intent when browsing task folders

### 3. **Future Extensibility**
- Can add clustering-specific loaders if needed
- Can add clustering-specific evaluation metrics
- Can treat clustering tasks differently in training pipelines

### 4. **Better Documentation**
- Separate documentation for each task type
- Clearer examples for each category
- Easier to explain to new contributors

## Technical Details

### Loader Compatibility
Both classification and clustering tasks use the same loader:
```python
from tasks.loaders import load_classification_standard

class ClusteringTask(AbsTask):
    # ... configuration ...
    metadata = TaskMetadata(type="Clustering", ...)
    loader = load_classification_standard  # ✅ Same loader works!
```

### Data Loading
The `load_task_data()` function handles both:
```python
def load_task_data(task, rank=0):
    task_type = task.metadata.type
    
    if task_type == "Retrieval":
        return _load_retrieval_data(task, rank)
    elif task_type in ["Classification", "Clustering"]:  # ✅ Both handled
        return _load_classification_data(task, rank)
```

### Backward Compatibility
✅ **100% backward compatible**
- All existing code continues to work
- `get_task("emotion")` still works
- `load_task_data()` API unchanged
- Only internal organization changed

## Final Structure

```
tasks/
├── loaders.py                  # Central loaders
├── retrieval_tasks/            # 35 tasks
├── sts_tasks/                  # 3 tasks
├── classification_tasks/       # 7 tasks ← Now pure classification
└── clustering_tasks/           # 18 tasks ← NEW: Pure clustering
```

## Statistics

| Category | Before | After | Change |
|----------|--------|-------|--------|
| Classification | 25 (mixed) | 7 (pure) | -18 |
| Clustering | 0 | 18 | +18 |
| Total Tasks | 63 | 63 | 0 |

## Usage Examples

### Classification Task
```python
from tasks import get_task
from inference.load_datasets import load_task_data

# Get classification task
task = get_task("banking77")
print(task.metadata.type)  # "Classification"

# Load data
data = load_task_data(task)  # ClassificationRawData
```

### Clustering Task
```python
from tasks import get_task
from inference.load_datasets import load_task_data

# Get clustering task
task = get_task("emotion")
print(task.metadata.type)  # "Clustering"

# Load data - same interface!
data = load_task_data(task)  # ClassificationRawData
```

## Summary

✅ **18 clustering tasks** separated into dedicated folder  
✅ **7 classification tasks** remain in classification folder  
✅ **100% backward compatible** - no API changes  
✅ **Clearer organization** - semantic separation  
✅ **Same loaders work** for both types  
✅ **Better maintainability** - easier to find and manage tasks  

The task module now has **4 clear categories**:
1. 🔍 Retrieval (35 tasks)
2. 📊 STS (3 tasks)
3. 🏷️ Classification (7 tasks)
4. 🔗 Clustering (18 tasks)

**Total: 63 tasks with crystal-clear organization!**
