# NLI Tasks Refactoring Summary

## Overview
Split NLI (Natural Language Inference) tasks from retrieval tasks into a separate folder and updated their loader to properly handle hard negatives according to the specified requirements.

## Changes Made

### 1. File Structure
- **Created**: `tasks/nli_tasks/` folder
- **Moved files**:
  - `snli.py` (from `retrieval_tasks/`)
  - `mnli.py` (from `retrieval_tasks/`)
  - `anli.py` (from `retrieval_tasks/`)
  - `all_nli.py` (from `retrieval_tasks/`)

### 2. Module Organization
- **Created**: `tasks/nli_tasks/__init__.py` - exports all NLI task classes
- **Updated**: `tasks/retrieval_tasks/__init__.py` - removed NLI task imports
- **Updated**: `tasks/__init__.py` - added import from `nli_tasks` module

### 3. Loader Implementation

#### `load_nli_retrieval()` (for SNLI, MNLI, ANLI)
Implements the following logic as specified:
- **Groups by premises**: Collects all hypotheses for each premise
- **Filters premises**: Retains only premises with at least one entailed hypothesis
- **Samples positives**: Randomly selects one entailed hypothesis as the positive
- **Handles hard negatives**: Adds neutral and contradictory hypotheses to the corpus so they can be mined as hard negatives

Key features:
- Deduplicates hypotheses across the corpus
- Properly handles invalid labels (e.g., -1 in SNLI)
- Creates unique query and document IDs
- Builds corpus containing both entailment and non-entailment hypotheses

#### `load_all_nli_retrieval()` (for ALL_NLI)
Specialized loader for the sentence-transformers/all-nli dataset:
- Handles pre-existing triplet structure (anchor, positive, negative)
- Includes negatives in the corpus for hard negative mining
- Deduplicates documents across positives and negatives

### 4. Task Configuration Updates
All NLI task files now use the appropriate loader:
- `SNLI`, `MNLI`, `ANLI` → `load_nli_retrieval`
- `ALL_NLI` → `load_all_nli_retrieval`

## Implementation Details

### NLI Loader Logic
```python
# 1. Group by premise
premise_to_hypotheses = {
    premise: {
        "entailment": [...],      # Label 0
        "non_entailment": [...]   # Labels 1, 2 (neutral, contradiction)
    }
}

# 2. Filter: keep only premises with ≥1 entailed hypothesis
valid_premises = {p: h for p, h in ... if len(h["entailment"]) > 0}

# 3. For each valid premise:
#    - Sample one entailed hypothesis as positive
#    - Add all non-entailment hypotheses to corpus
#    - These will be found as hard negatives during mining

# 4. Build corpus from all unique hypotheses
```

### Corpus Structure
The corpus now contains:
- All entailed hypotheses (positives)
- All neutral hypotheses (potential hard negatives)
- All contradictory hypotheses (potential hard negatives)

This ensures that when hard negatives are mined, the neutral and contradictory hypotheses will be naturally selected as they are semantically similar but not entailed.

## Files Modified
1. `tasks/loaders.py` - Added `load_nli_retrieval()` and `load_all_nli_retrieval()`
2. `tasks/__init__.py` - Added `from .nli_tasks import *`
3. `tasks/retrieval_tasks/__init__.py` - Removed NLI imports
4. `tasks/nli_tasks/__init__.py` - Created with NLI exports
5. `tasks/nli_tasks/*.py` - Updated loader references

## Testing
The refactoring maintains backward compatibility:
- All NLI tasks remain accessible through `tasks.SNLI`, `tasks.MNLI`, etc.
- The `NAME_TO_TASK` dictionary in `tasks/__init__.py` still works
- Loader functions return the same `RetrievalRawData` structure

## Next Steps
When using these tasks:
1. Load task data using the standard `load_task_data()` function
2. Mine hard negatives - neutral/contradictory hypotheses will be found naturally
3. Train with the resulting triplets (query, positive, hard_negatives)
