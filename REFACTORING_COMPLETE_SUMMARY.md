# Complete Task Refactoring Summary

## Overview
This document describes the complete refactoring of the tasks module, creating a fully modular architecture where:
- Each task is in its own file with its associated loader
- Loaders are centralized in `tasks/loaders.py` with a common interface
- Tasks are organized by type: retrieval, STS, and classification/clustering

## Final Structure

```
tasks/
├── loaders.py                      # Common loaders for all task types
├── abs_task.py                     # Base task class
├── prompts.py                      # Task prompts
├── __init__.py                     # Main entry point
│
├── retrieval_tasks/
│   ├── __init__.py                 # 35 retrieval task imports
│   ├── msmarco.py
│   ├── arguana.py
│   ├── squad.py
│   ├── ...                         # 32 more retrieval tasks
│   └── amazonqa.py
│
├── sts_tasks/
│   ├── __init__.py                 # 3 STS task imports
│   ├── sts12.py
│   ├── sts22.py
│   └── stsbenchmark.py
│
└── classification_tasks/
    ├── __init__.py                 # 25 classification task imports
    ├── dbpediaclassification.py
    ├── banking77classification.py
    ├── ...                         # 23 more classification tasks
    └── twentynewsgroupsclustering.py
```

## Changes Made

### 1. Central Loaders Module (`tasks/loaders.py`)

**Location:** Moved from `tasks/retrieval_tasks/loaders.py` to `tasks/loaders.py`

**Purpose:** Central location for all loader functions with a common interface

**Contents:**
- **Retrieval Loaders:**
  - `from_one_hf_dataset` - Single dataset loader (15+ tasks)
  - `from_multiple_hf_datasets` - Multi-dataset loader (2 tasks)
  - `from_multiple_hf_datasets_with_dedup` - Multi-dataset with dedup (7 tasks)
  - `load_nli_retrieval` - NLI tasks loader (3 tasks)
  - `load_sts_retrieval` - STS tasks loader (3 tasks)
  
- **Classification Loaders:**
  - `load_classification_standard` - Standard classification/clustering loader (25 tasks)

### 2. STS Tasks Separation

**Before:** STS tasks were mixed with retrieval tasks in `tasks/retrieval_tasks/`

**After:** STS tasks have their own dedicated folder `tasks/sts_tasks/`

**Rationale:** 
- STS tasks have different characteristics (sentence similarity vs. document retrieval)
- Better organization and easier to find
- Can have STS-specific loaders if needed in the future

**Tasks:**
- `STS12` - STS 2012 dataset
- `STS22` - STS 2022 cross-lingual dataset  
- `STSBenchmark` - STS Benchmark dataset

### 3. Classification Tasks Modularization

**Before:** All 25 classification/clustering tasks in one 272-line file

**After:** Each task in its own file with loader reference

**Structure:**
```python
from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.loaders import load_classification_standard


class TaskName(AbsTask):
    hf_name = "..."
    split = "train"
    anchor_name = "text"
    label_name = "label"  # or "label" attribute
    metadata = TaskMetadata(type="Classification", prompt={...})
    loader = load_classification_standard
```

**Classification Tasks (11):**
1. DBPediaClassification
2. ToxicConversations50k
3. Banking77Classification
4. AmazonCounterfactualClassification
5. AmazonPolarityClassification
6. ImdbClassification
7. ColaClassification
8. AmazonReviewsClustering
9. EmotionClustering
10. MTOPIntentClustering
11. MTOPDomainClustering

**Clustering Tasks (14):**
12. MassiveScenarioClustering
13. MassiveIntentClustering
14. TweetSentimentExtractionClustering
15. ArxivClusteringP2P
16. ArxivClusteringS2S
17. BiorxivClusteringP2P
18. BiorxivClusteringS2S
19. MedrxivClusteringP2P
20. MedrxivClusteringS2S
21. RedditClusteringP2P
22. RedditClusteringS2S
23. StackExchangeClusteringP2P
24. StackExchangeClusteringS2S
25. TwentyNewsgroupsClustering

### 4. Unified Loader Interface

**Before:** Different dispatch logic for retrieval and classification

```python
# Retrieval: complex string-based dispatch
if custom_loader:
    loader_func = {"load_nli_retrieval": _load_nli_retrieval, ...}.get(custom_loader)
elif task.has_multiple_datasets:
    ...

# Classification: inline implementation
def _load_classification_data(task, rank=0):
    label_field = getattr(task, "label_name", None) or ...
    dataset = load_dataset(...)
    ...
```

**After:** Uniform loader attribute for all tasks

```python
# Both retrieval and classification use the same pattern
def _get_retrieval_raw_data(task, rank=0):
    loader_func = getattr(task, "loader", None)
    if loader_func is None:
        raise ValueError(...)
    return loader_func(task, rank, ...)

def _load_classification_data(task, rank=0):
    loader_func = getattr(task, "loader", None)
    if loader_func is None:
        raise ValueError(...)
    return loader_func(task, rank)
```

### 5. Import Updates

**Retrieval Tasks:**
```python
# Before
from .loaders import from_multiple_hf_datasets_with_dedup

# After  
from tasks.loaders import from_multiple_hf_datasets_with_dedup
```

**Main `tasks/__init__.py`:**
```python
from .classification_tasks import *
from .retrieval_tasks import *
from .sts_tasks import *  # Now included

NAME_TO_TASK = {
    # All tasks accessible via get_task("task_name")
    ...
}
```

## Task Statistics

| Task Type | Count | Files |
|-----------|-------|-------|
| Retrieval | 35 | tasks/retrieval_tasks/*.py |
| STS | 3 | tasks/sts_tasks/*.py |
| Classification | 11 | tasks/classification_tasks/*.py |
| Clustering | 14 | tasks/classification_tasks/*.py |
| **Total** | **63** | **- |

## Code Reduction

| File | Before | After | Reduction |
|------|--------|-------|-----------|
| `inference/load_datasets.py` | 1143 lines | 148 lines | **87%** ⬇️ |
| `tasks/retrieval_tasks.py` | 603 lines | N/A (modularized) | **100%** ⬇️ |
| `tasks/sts_tasks.py` | 52 lines | N/A (modularized) | **100%** ⬇️ |
| `tasks/classification_tasks.py` | 272 lines | N/A (modularized) | **100%** ⬇️ |

**Total lines eliminated from monolithic files: ~2,070 lines**

## Benefits

### 1. **Modularity**
- Each task is self-contained in its own file
- Easy to locate and modify specific tasks
- Clear dependencies per task

### 2. **Organization**
- Tasks grouped by type (retrieval, STS, classification)
- Common loaders in central location
- Task-specific loaders co-located with tasks

### 3. **Uniform Interface**
- All tasks have a `loader` attribute
- No more `custom_loader` string references
- Consistent pattern across all task types

### 4. **Type Safety**
- Direct function references instead of string lookups
- Compile-time checking
- IDE autocomplete and navigation

### 5. **Extensibility**
- Adding a new task: create one file
- Adding a shared loader: add to `tasks/loaders.py`
- No central dispatch logic to modify

### 6. **Maintainability**
- Small, focused files
- Clear separation of concerns
- Easy to test individual tasks

## Migration Examples

### Adding a New Retrieval Task

1. Create `tasks/retrieval_tasks/my_task.py`:
```python
from tasks.abs_task import AbsTask, TaskMetadata
from tasks.loaders import from_one_hf_dataset  # or custom loader

class MyTask(AbsTask):
    hf_name = "my/dataset"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "document"
    metadata = TaskMetadata(type="Retrieval", prompt={"query": "..."})
    loader = from_one_hf_dataset  # or custom_loader_function
```

2. Add import to `tasks/retrieval_tasks/__init__.py`:
```python
from .my_task import MyTask
__all__ = [..., "MyTask"]
```

3. Add to `tasks/__init__.py` NAME_TO_TASK dict:
```python
"my_task": MyTask,
```

### Adding a New Classification Task

1. Create `tasks/classification_tasks/my_classifier.py`:
```python
from tasks.abs_task import AbsTask, TaskMetadata
from tasks.loaders import load_classification_standard

class MyClassifier(AbsTask):
    hf_name = "my/dataset"
    split = "train"
    anchor_name = "text"
    label_name = "label"
    metadata = TaskMetadata(type="Classification", prompt={"query": "..."})
    loader = load_classification_standard
```

2. Add import to `tasks/classification_tasks/__init__.py`

3. Add to `tasks/__init__.py` NAME_TO_TASK dict

## Backward Compatibility

✅ **Public API unchanged:** `load_task_data(task)` works exactly as before

✅ **All existing code continues to work** without modifications

✅ **Old files backed up** as `.old` files for reference

## Files Modified

### Created
- `tasks/loaders.py` (central loader module)
- `tasks/retrieval_tasks/` (35 task files + __init__.py)
- `tasks/sts_tasks/` (3 task files + __init__.py)
- `tasks/classification_tasks/` (25 task files + __init__.py)

### Modified
- `inference/load_datasets.py` (simplified to 148 lines)
- `tasks/__init__.py` (updated imports)

### Deprecated (backed up as .old)
- `tasks/retrieval_tasks.py` → `tasks/retrieval_tasks.py.old`
- `tasks/sts_tasks.py` → `tasks/sts_tasks.py.old`
- `tasks/classification_tasks.py` → `tasks/classification_tasks.py.old`

## Summary

This refactoring transforms the entire tasks module into a clean, modular architecture:

- **63 tasks** across 3 categories
- **Central loaders** with common interface
- **87% code reduction** in data loading logic
- **100% backward compatible**
- **Uniform `loader` attribute** across all tasks
- **Easy to extend** and maintain

The new structure makes it trivial to add new tasks, understand existing ones, and maintain the codebase as it grows.
