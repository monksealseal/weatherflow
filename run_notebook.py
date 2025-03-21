#!/usr/bin/env python3
"""
Run a specific notebook with the correct environment and imports.

This script:
1. Ensures paths are set correctly for imports
2. Launches the notebook in Jupyter
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_notebook(notebook_path, no_browser=False):
    """Run a notebook with proper environment setup."""
    notebook_path = Path(notebook_path).absolute()
    
    if not notebook_path.exists():
        print(f"Error: Notebook '{notebook_path}' does not exist!")
        return False
    
    # Get repository root
    repo_root = Path(__file__).parent.absolute()
    
    print(f"\nPreparing to run notebook: {notebook_path}")
    print(f"Repository root: {repo_root}")
    
    # First check if we need to fix imports
    from notebooks.fix_notebook_imports import add_path_code
    fixed = add_path_code(notebook_path)
    if fixed:
        print("Fixed imports in notebook.")
    
    # Launch Jupyter
    notebook_dir = notebook_path.parent
    cmd = ["jupyter", "notebook", str(notebook_path)]
    if no_browser:
        cmd.append("--no-browser")
    
    print(f"\nLaunching notebook with command: {' '.join(cmd)}")
    subprocess.run(cmd)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a WeatherFlow notebook with proper environment setup")
    parser.add_argument("notebook", help="Path to the notebook to run")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    
    args = parser.parse_args()
    
    # Run the notebook
    success = run_notebook(args.notebook, args.no_browser)
    if not success:
        sys.exit(1)