#!/usr/bin/env python3
"""
Fix weatherflow notebook import issues.

This script modifies all Jupyter notebooks in the weatherflow repository to include
the proper system path adjustments, ensuring they can import the weatherflow package
when run directly from the repository.
"""

import os
import json
import glob
import sys
from pathlib import Path

def add_path_code(notebook_path):
    """Add code to notebook to fix import path."""
    print(f"Processing: {notebook_path}")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Check if the notebook has cells
    if 'cells' not in notebook:
        print(f"Warning: {notebook_path} doesn't have cells structure.")
        return False
    
    # Find the first code cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            # Check if imports are in this cell
            source = ''.join(cell['source'])
            if 'import' in source and 'weatherflow' in source:
                # Add path modification code
                path_code = [
                    "# Add repository root to Python path\n",
                    "import sys\n",
                    "import os\n",
                    "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('__file__'), '..')))\n",
                    "\n"
                ]
                
                # Insert at the beginning of the cell
                notebook['cells'][i]['source'] = path_code + notebook['cells'][i]['source']
                print(f"  - Added path code to cell {i}")
                
                # Write back the notebook
                with open(notebook_path, 'w', encoding='utf-8') as f:
                    json.dump(notebook, f, indent=2)
                
                return True
    
    print(f"  - No suitable import cell found in {notebook_path}")
    return False

def fix_all_notebooks(repo_dir):
    """Fix all notebooks in the repository."""
    # Find all notebooks
    notebooks = glob.glob(os.path.join(repo_dir, "**/*.ipynb"), recursive=True)
    
    success_count = 0
    for notebook_path in notebooks:
        if add_path_code(notebook_path):
            success_count += 1
    
    print(f"\nFixed {success_count} out of {len(notebooks)} notebooks.")

if __name__ == "__main__":
    # Get repository root directory
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Repository directory: {repo_dir}")
    
    fix_all_notebooks(repo_dir)
    print("\nDone! Run notebooks with the current Python environment.")