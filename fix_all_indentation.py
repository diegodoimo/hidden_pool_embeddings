#!/usr/bin/env python3
"""Fix all indentation issues caused by misplaced language attribute."""

import re
from pathlib import Path

# Find all Python files with the issue
base_path = Path("tasks")
files_with_issues = []

for py_file in base_path.rglob("*.py"):
    if "__pycache__" in str(py_file) or py_file.name == "__init__.py":
        continue
    
    try:
        with open(py_file, 'r') as f:
            content = f.read()
        
        # Check if file has the misplaced language attribute
        if re.search(r'^\s{4}language = "(?:en|zh|multilingual)"\n\s{8}pass', content, re.MULTILINE):
            files_with_issues.append(py_file)
    except:
        pass

print(f"Found {len(files_with_issues)} files with indentation issues")

for file_path in files_with_issues:
    print(f"Fixing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove the misplaced language attribute and fix the pass statement
    content = re.sub(
        r'(def validate_config\(cls\) -> None:\n\s+"""[^"]+""")\n\n\s{4}language = "(?:en|zh|multilingual)"\n\s{8}pass',
        r'\1\n        pass',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"  ✓ Fixed {file_path}")

print(f"\n✓ All {len(files_with_issues)} files fixed!")
