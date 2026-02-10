#!/usr/bin/env python3
"""Fix indentation issues in binary classification files."""

import re
from pathlib import Path

files_to_fix = [
    "tasks/binary_classification_tasks/data_loaders/amazonpolarityclassification.py",
    "tasks/binary_classification_tasks/data_loaders/colaclassification.py",
    "tasks/binary_classification_tasks/data_loaders/imdbclassification.py",
    "tasks/binary_classification_tasks/data_loaders/toxicconversations50k.py",
]

for file_path in files_to_fix:
    print(f"Fixing {file_path}...")
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove misplaced language attribute and pass statement
    content = re.sub(r'(\s+"""Validate task configuration\.""")\n\n\s+language = "en"\n\s+pass', 
                     r'\1\n        pass', content)
    
    # Add language attribute at the class level if not present
    if 'language = "en"' not in content.split('class ')[1].split('hf_name')[0]:
        content = re.sub(r'(class \w+\(AbsTask\):)\n(\s+hf_name)', 
                        r'\1\n    language = "en"\n\2', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"  ✓ Fixed {file_path}")

print("\nAll files fixed!")
