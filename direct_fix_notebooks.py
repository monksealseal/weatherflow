#!/usr/bin/env python3
"""
Direct fix for notebook imports - works without additional dependencies.
This script rewrites the notebooks to ensure they can directly import weatherflow
from the repository.
"""

import os
import json
import glob
from pathlib import Path

def fix_notebook(notebook_path):
    """Fix a single notebook to properly import weatherflow."""
    print(f"Processing: {notebook_path}")
    
    # Read the notebook file
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    try:
        notebook = json.loads(content)
    except json.JSONDecodeError:
        print(f"  - Error: Could not parse {notebook_path} as JSON.")
        return False
    
    # Check for cells
    if 'cells' not in notebook:
        print(f"  - Warning: No cells found in {notebook_path}")
        return False
    
    # Look for import cell
    import_cell_idx = None
    for idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'import weatherflow' in source:
                import_cell_idx = idx
                break
    
    if import_cell_idx is None:
        print(f"  - Warning: No import weatherflow found in {notebook_path}")
        return False
    
    # Create new code for import
    path_code = [
        "# Add repository root to Python path to find weatherflow package\n",
        "import sys\n",
        "import os\n",
        "# Get absolute path to repo root\n",
        "notebook_dir = os.path.dirname(os.path.abspath('__file__'))\n",
        "repo_root = os.path.abspath(os.path.join(notebook_dir, '..'))\n",
        "# Add to path if not already there\n",
        "if repo_root not in sys.path:\n",
        "    sys.path.insert(0, repo_root)\n",
        "print(f\"Added {repo_root} to Python path\")\n",
        "\n"
    ]
    
    # Add before the import
    cell = notebook['cells'][import_cell_idx]
    cell['source'] = path_code + cell['source']
    notebook['cells'][import_cell_idx] = cell
    
    # Write back to file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"  - Fixed import for {notebook_path}")
    return True

def fix_all_notebooks():
    """Fix all notebooks in the repository."""
    repo_root = Path(__file__).parent.absolute()
    print(f"Repository root: {repo_root}")
    
    # Find all notebooks
    notebook_patterns = [
        os.path.join(repo_root, "notebooks", "*.ipynb"),
        os.path.join(repo_root, "examples", "*.ipynb")
    ]
    
    notebook_files = []
    for pattern in notebook_patterns:
        notebook_files.extend(glob.glob(pattern))
    
    # Fix each notebook
    fixed_count = 0
    for path in notebook_files:
        if fix_notebook(path):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} out of {len(notebook_files)} notebooks.")

if __name__ == "__main__":
    fix_all_notebooks()
    print("\nDone! You should now be able to run the notebooks.")
    print("To run a notebook, use:")
    print("  jupyter notebook <notebook_path> # If Jupyter is installed")
    print("  - OR -")
    print("  Open in Google Colab by uploading both the notebook and the repository.")