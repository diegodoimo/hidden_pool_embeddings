#!/usr/bin/env python3
"""Script to add language attribute to all dataset files."""

import os
import re
from pathlib import Path
from datasets import load_dataset


dataset = load_dataset(
    "mteb/xnli", "en", split="train"
)  # Load a multilingual dataset to ensure it's cached


# Define language mappings based on dataset names
MULTILINGUAL_DATASETS = {
    "ayadataset",
    "miracl",
    "mrtydi",
    "pawsxmultilingual",
    "multilingualsentimentclustering",
    "nllb",
}


def add_language_attribute(file_path, language):
    """Add language attribute to a dataset class file."""
    with open(file_path, "r") as f:
        content = f.read()

    # Check if language attribute already exists
    if re.search(r"^\s*language\s*=", content, re.MULTILINE):
        print(f"Language attribute already exists in {file_path}")
        return False

    # Find the class definition
    class_match = re.search(
        r'(class\s+\w+\(AbsTask\):.*?""".*?""")', content, re.DOTALL
    )
    if not class_match:
        print(f"Could not find class definition in {file_path}")
        return False

    # Insert language attribute after the docstring
    insertion_point = class_match.end()
    new_content = (
        content[:insertion_point]
        + f'\n\n    language = "{language}"'
        + content[insertion_point:]
    )

    with open(file_path, "w") as f:
        f.write(new_content)

    print(f"Added language='{language}' to {file_path}")
    return True


def process_directory(base_dir, language):
    """Process all Python files in a directory."""
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                file_path = os.path.join(root, file)

                # Check if it's a multilingual dataset
                file_stem = Path(file).stem.lower()
                if file_stem in MULTILINGUAL_DATASETS:
                    add_language_attribute(file_path, "multilingual")
                else:
                    add_language_attribute(file_path, language)


# Main tasks (English)
english_dirs = [
    "tasks/retrieval_tasks/data_loaders",
    "tasks/nli_tasks/data_loaders",
    "tasks/sts_tasks/data_loaders",
    "tasks/clustering_tasks/data_loaders",
    "tasks/binary_classification_tasks/data_loaders",
    "tasks/classification_tasks/data_loaders",
]

# Chinese tasks
chinese_dirs = [
    "tasks/chinese/retrieval_tasks/data_loaders",
    "tasks/chinese/nli_tasks/data_loaders",
    "tasks/chinese/sts_tasks/data_loaders",
    "tasks/chinese/clustering_tasks/data_loaders",
]

print("Processing English datasets...")
for dir_path in english_dirs:
    if os.path.exists(dir_path):
        process_directory(dir_path, "en")

print("\nProcessing Chinese datasets...")
for dir_path in chinese_dirs:
    if os.path.exists(dir_path):
        process_directory(dir_path, "zh")

print("\nDone!")
