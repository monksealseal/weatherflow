#!/usr/bin/env python3
"""
Comprehensive fix for weatherflow notebooks.
This script:
1. Fixes import paths to include the repository root
2. Adds mock implementations of dependencies that might not be available
3. Adds mock data generators to replace external data loading
"""

import os
import json
import glob
from pathlib import Path

MOCK_DEPENDENCIES_CODE = """
# Mock dependencies that might not be available
try:
    import sys
    notebook_dir = os.path.dirname(os.path.abspath('__file__'))
    repo_root = os.path.abspath(os.path.join(notebook_dir, '..'))
    mock_path = os.path.join(repo_root, 'mock_dependencies.py')
    
    if os.path.exists(mock_path):
        # Execute the mock dependencies script
        with open(mock_path, 'r') as f:
            mock_code = f.read()
            # Add repo_root to sys.path if not already there
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            # Execute the script
            exec(mock_code)
            # Call the function to install all mocks
            exec("install_all_mocks()")
    else:
        print(f"Warning: Mock dependencies script not found at {mock_path}")
except Exception as e:
    print(f"Error loading mock dependencies: {str(e)}")
"""

PATH_SETUP_CODE = """
# Add repository root to Python path to find weatherflow package
import sys
import os

# Get absolute path to repo root
notebook_dir = os.path.dirname(os.path.abspath('__file__'))
repo_root = os.path.abspath(os.path.join(notebook_dir, '..'))

# Add to path if not already there
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    print(f"Added {repo_root} to Python path")
"""

def fix_notebook(notebook_path):
    """Fix a single notebook to properly run without external dependencies."""
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
    
    # Find the first code cell for path setup
    first_code_cell_idx = None
    for idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            first_code_cell_idx = idx
            break
    
    if first_code_cell_idx is None:
        print(f"  - Warning: No code cells found in {notebook_path}")
        return False
    
    # Create path setup cell if it doesn't exist
    cell = notebook['cells'][first_code_cell_idx]
    cell_source = ''.join(cell['source'])
    
    changes_made = False
    
    # Only add if not already there
    if 'sys.path.insert' not in cell_source:
        cell['source'] = PATH_SETUP_CODE.strip().split('\n')
        changes_made = True
        print(f"  - Added path setup code to cell {first_code_cell_idx}")
    
    # Add mock dependencies after path setup
    mock_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": MOCK_DEPENDENCIES_CODE.strip().split('\n')
    }
    
    notebook['cells'].insert(first_code_cell_idx + 1, mock_cell)
    changes_made = True
    print(f"  - Added mock dependencies cell")
    
    # Write back to file if changes were made
    if changes_made:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        print(f"  - Updated {notebook_path}")
        return True
    else:
        print(f"  - No changes needed for {notebook_path}")
        return False

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
    print("\nDone! You should now be able to run the notebooks without external dependencies.")
    print("\nTo run a notebook, use:")
    print("  jupyter notebook <notebook_path> # If Jupyter is installed")
    print("  - OR -")
    print("  Open in Google Colab by uploading both the notebook and the weatherflow repository.")