# NLI Tasks Structure Diagram

## Before Refactoring

```
tasks/
├── retrieval_tasks/
│   ├── __init__.py (imports SNLI, MNLI, ANLI, ALL_NLI)
│   ├── snli.py ──────────┐
│   ├── mnli.py ──────────┤
│   ├── anli.py ──────────┼──> All used load_nli_retrieval()
│   ├── all_nli.py ───────┘     (old implementation)
│   └── ... (other retrieval tasks)
│
└── loaders.py
    └── load_nli_retrieval() (simple filtering, no premise grouping)
```

## After Refactoring

```
tasks/
├── nli_tasks/                    [NEW]
│   ├── __init__.py               [NEW - exports NLI tasks]
│   ├── snli.py ──────────┐
│   ├── mnli.py ──────────┼──> Use load_nli_retrieval()
│   ├── anli.py ──────────┤     (new implementation)
│   └── all_nli.py ───────┘──> Uses load_all_nli_retrieval()
│
├── retrieval_tasks/
│   ├── __init__.py (NLI imports removed)
│   └── ... (only retrieval tasks)
│
├── __init__.py
│   └── from .nli_tasks import *  [NEW]
│
└── loaders.py
    ├── load_nli_retrieval()      [UPDATED]
    │   ├─ Groups by premise
    │   ├─ Filters for entailment
    │   ├─ Samples positive
    │   └─ Adds hard negatives to corpus
    │
    └── load_all_nli_retrieval()  [NEW]
        └─ Handles triplet format with negatives
```

## Data Flow

### SNLI/MNLI/ANLI Loader Flow

```
Raw Dataset
    │
    ├─ premise: "A person is walking"
    ├─ hypothesis: "Someone is moving"
    └─ label: 0 (entailment)
    
    ├─ premise: "A person is walking"
    ├─ hypothesis: "A person is running"
    └─ label: 2 (contradiction)
    
    ↓
    
Group by Premise
    │
    └─ "A person is walking":
        ├─ entailment: ["Someone is moving"]
        └─ non_entailment: ["A person is running"]
    
    ↓
    
Filter (keep premises with ≥1 entailment)
    │
    └─ "A person is walking": ✓ (has entailment)
    
    ↓
    
Create Query-Positive Pairs
    │
    ├─ Query: "A person is walking"
    └─ Positive: "Someone is moving" (sampled from entailments)
    
    ↓
    
Build Corpus (includes hard negatives)
    │
    ├─ doc_0: "Someone is moving" (positive)
    └─ doc_1: "A person is running" (hard negative)
    
    ↓
    
Hard Negative Mining
    │
    └─ Will naturally find "A person is running" as hard negative
       (semantically similar but not entailed)
```

### ALL_NLI Loader Flow

```
Raw Dataset (Triplet Format)
    │
    ├─ anchor: "A person is walking"
    ├─ positive: "Someone is moving"
    └─ negative: "A person is sitting"
    
    ↓
    
Deduplicate and Build Corpus
    │
    ├─ doc_0: "Someone is moving" (positive)
    └─ doc_1: "A person is sitting" (negative)
    
    ↓
    
Create Query-Positive Pairs
    │
    ├─ Query: "A person is walking"
    └─ Positive: doc_0
    
    ↓
    
Hard Negative Mining
    │
    └─ Will find doc_1 as hard negative
```

## Key Improvements

1. **Premise Grouping**: Multiple hypotheses per premise are now properly handled
2. **Sampling**: One entailed hypothesis is randomly selected as positive
3. **Hard Negatives**: Neutral/contradictory hypotheses are included in corpus
4. **Deduplication**: Hypotheses are deduplicated across the corpus
5. **Separation**: NLI tasks are logically separated from retrieval tasks
