# Task Module Structure

## Directory Tree

```
tasks/
├── loaders.py                          # ⭐ CENTRAL LOADERS (common interface)
│   ├── Retrieval loaders:
│   │   ├── from_one_hf_dataset         (15+ tasks)
│   │   ├── from_multiple_hf_datasets   (2 tasks)
│   │   ├── from_multiple_hf_datasets_with_dedup  (7 tasks)
│   │   ├── load_nli_retrieval          (3 tasks)
│   │   └── load_sts_retrieval          (3 tasks)
│   └── Classification loaders:
│       └── load_classification_standard (25 tasks)
│
├── abs_task.py                         # Base task class
├── prompts.py                          # Task prompts
├── __init__.py                         # Main entry (imports all tasks)
│
├── retrieval_tasks/                    # 🔍 RETRIEVAL (35 tasks)
│   ├── __init__.py
│   ├── msmarco.py                      # Uses: from_multiple_hf_datasets_with_dedup
│   ├── arguana.py                      # Uses: custom loader (in file)
│   ├── squad.py                        # Uses: custom loader (in file)
│   ├── naturalquestions.py            # Uses: from_one_hf_dataset
│   ├── all_nli.py                      # Uses: from_one_hf_dataset
│   ├── snli.py                         # Uses: load_nli_retrieval
│   ├── mnli.py                         # Uses: load_nli_retrieval
│   ├── anli.py                         # Uses: load_nli_retrieval
│   ├── miracl.py                       # Uses: custom loader (in file)
│   ├── pubmedqa.py                     # Uses: custom loader (in file)
│   ├── xsum.py                         # Uses: custom loader (in file)
│   ├── cnndm.py                        # Uses: custom loader (in file)
│   ├── stackexchange.py                # Uses: custom loader (in file)
│   ├── stackoverflow_dup.py            # Uses: custom loader (in file)
│   └── ... (21 more tasks)
│
├── sts_tasks/                          # 📊 STS (3 tasks)
│   ├── __init__.py
│   ├── sts12.py                        # Uses: load_sts_retrieval
│   ├── sts22.py                        # Uses: load_sts_retrieval
│   └── stsbenchmark.py                 # Uses: load_sts_retrieval
│
└── classification_tasks/               # 🏷️  CLASSIFICATION (25 tasks)
    ├── __init__.py
    ├── dbpediaclassification.py        # Uses: load_classification_standard
    ├── banking77classification.py      # Uses: load_classification_standard
    ├── toxicconversations50k.py        # Uses: load_classification_standard
    ├── amazonreviewsclustering.py      # Uses: load_classification_standard
    └── ... (21 more tasks)
```

## Task File Pattern

### Retrieval Task with Shared Loader
```python
# tasks/retrieval_tasks/msmarco.py
from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.loaders import from_multiple_hf_datasets_with_dedup

class MSMARCO(AbsTask):
    hf_name = "mteb/msmarco"
    # ... task configuration ...
    loader = from_multiple_hf_datasets_with_dedup  # ⭐ Direct reference
```

### Retrieval Task with Custom Loader
```python
# tasks/retrieval_tasks/arguana.py
from tasks.abs_task import AbsTask, TaskMetadata
from datasets import load_dataset

def load_arguana_dedup_retrieval(task):
    """Custom loader specific to this task"""
    # ... custom loading logic ...
    return RetrievalRawData(...)

class Arguana(AbsTask):
    hf_name = "BeIR/arguana-generated-queries"
    # ... task configuration ...
    loader = load_arguana_dedup_retrieval  # ⭐ Co-located custom loader
```

### STS Task
```python
# tasks/sts_tasks/sts12.py
from tasks.abs_task import AbsTask, TaskMetadata
from tasks.loaders import load_sts_retrieval

class STS12(AbsTask):
    hf_name = "mteb/sts12-sts"
    # ... task configuration ...
    loader = load_sts_retrieval  # ⭐ Shared STS loader
```

### Classification Task
```python
# tasks/classification_tasks/banking77classification.py
from tasks.abs_task import AbsTask, TaskMetadata
from tasks.loaders import load_classification_standard

class Banking77Classification(AbsTask):
    hf_name = "mteb/banking77"
    # ... task configuration ...
    loader = load_classification_standard  # ⭐ Shared classification loader
```

## Data Flow

```
User Code
    ↓
tasks.get_task("msmarco")
    ↓
MSMARCO task object (with loader attribute)
    ↓
inference.load_task_data(task)
    ↓
Dispatch based on task.metadata.type
    ↓
┌─────────────────────────────────────┐
│ if type == "Retrieval":             │
│   loader = task.loader              │ ⭐ Get loader from task
│   return loader(task, rank, ...)    │
│                                     │
│ elif type == "Classification":      │
│   loader = task.loader              │ ⭐ Get loader from task
│   return loader(task, rank)         │
└─────────────────────────────────────┘
    ↓
Loader function (from tasks.loaders.py or task file)
    ↓
RetrievalRawData or ClassificationRawData
    ↓
Convert to HuggingFace Datasets
    ↓
Return to user
```

## Key Improvements

### Before
```
❌ String-based dispatch: custom_loader = "load_nli_retrieval"
❌ Monolithic files: 603 lines for retrieval, 272 for classification
❌ Mixed organization: STS tasks with retrieval tasks
❌ Loaders scattered: some in load_datasets.py, some in task files
```

### After
```
✅ Direct references: loader = load_nli_retrieval
✅ Modular files: ~20-50 lines per task file
✅ Clear organization: retrieval, STS, classification separated
✅ Central loaders: all in tasks/loaders.py with common interface
```

## Statistics

| Metric | Value |
|--------|-------|
| Total tasks | 63 |
| Retrieval tasks | 35 |
| STS tasks | 3 |
| Classification tasks | 11 |
| Clustering tasks | 14 |
| Shared loaders | 6 |
| Task-specific loaders | 8 |
| Lines in tasks/loaders.py | ~460 |
| Average lines per task file | ~30 |

## Usage Example

```python
from tasks import get_task
from inference.load_datasets import load_task_data

# Get any task (retrieval, STS, or classification)
task = get_task("msmarco")  # or "sts12" or "banking77"

# Load data - same interface for all task types!
data = load_task_data(task, rank=0)

# For retrieval/STS:
# data = (hf_dataset, corpus_dict, has_title)

# For classification:
# data = ClassificationRawData(texts, labels, ids)
```
