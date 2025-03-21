#!/usr/bin/env python3
"""
Set up a notebook environment for the weatherflow repository.

This script:
1. Creates a new virtual environment for running notebooks
2. Installs all required dependencies
3. Registers the environment as a Jupyter kernel
4. Fixes the notebook imports
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, verbose=True):
    """Run a shell command and print output."""
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Error: {result.stderr}")
        return False
    
    if verbose and result.stdout:
        print(result.stdout)
    
    return True

def setup_environment(env_name="weatherflow-notebooks", use_conda=False, verbose=True):
    """Set up a virtual environment for running notebooks."""
    repo_root = Path(__file__).parent.absolute()
    notebooks_dir = repo_root / "notebooks"
    
    print(f"\n{'='*80}")
    print(f"Setting up notebook environment for WeatherFlow")
    print(f"{'='*80}\n")
    
    print(f"Repository root: {repo_root}")
    print(f"Notebooks directory: {notebooks_dir}")
    
    # Create the environment
    print(f"\n1. Creating environment '{env_name}'...")
    if use_conda:
        # Use conda
        success = run_command(["conda", "create", "-y", "-n", env_name, "python=3.10"], verbose=verbose)
        if not success:
            return False
        
        # Activate conda env
        if sys.platform == "win32":
            python_cmd = ["conda", "run", "-n", env_name, "python"]
            pip_cmd = ["conda", "run", "-n", env_name, "pip"]
        else:
            python_cmd = ["conda", "run", "-n", env_name, "python"]
            pip_cmd = ["conda", "run", "-n", env_name, "pip"]
    else:
        # Use venv
        env_path = repo_root / env_name
        success = run_command([sys.executable, "-m", "venv", str(env_path)], verbose=verbose)
        if not success:
            return False
        
        # Set up python and pip commands
        if sys.platform == "win32":
            python_cmd = [str(env_path / "Scripts" / "python.exe")]
            pip_cmd = [str(env_path / "Scripts" / "pip.exe")]
        else:
            python_cmd = [str(env_path / "bin" / "python")]
            pip_cmd = [str(env_path / "bin" / "pip")]
    
    # Install dependencies
    print(f"\n2. Installing dependencies...")
    
    # Upgrade pip
    run_command([*pip_cmd, "install", "--upgrade", "pip"], verbose=verbose)
    
    # Install notebook requirements
    run_command([*pip_cmd, "install", "-r", str(notebooks_dir / "notebook_requirements.txt")], verbose=verbose)
    
    # Install the weatherflow package
    print(f"\n3. Installing weatherflow package...")
    run_command([*pip_cmd, "install", "-e", str(repo_root)], verbose=verbose)
    
    # Install and register ipykernel
    print(f"\n4. Registering Jupyter kernel...")
    run_command([*pip_cmd, "install", "ipykernel"], verbose=verbose)
    run_command([*python_cmd, "-m", "ipykernel", "install", "--user", "--name", env_name, "--display-name", f"Python (WeatherFlow)"], verbose=verbose)
    
    # Fix notebook imports
    print(f"\n5. Fixing notebook imports...")
    run_command([*python_cmd, str(notebooks_dir / "fix_notebook_imports.py")], verbose=verbose)
    
    print(f"\n{'='*80}")
    print(f"Environment setup complete!")
    print(f"{'='*80}\n")
    print(f"To run the notebooks:")
    print(f"1. Start Jupyter Lab or Notebook: jupyter lab")
    print(f"2. Select the 'Python (WeatherFlow)' kernel when opening a notebook")
    print(f"3. Run cells normally")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up notebook environment for WeatherFlow")
    parser.add_argument("--conda", action="store_true", help="Use conda instead of venv")
    parser.add_argument("--env-name", default="weatherflow-notebooks", help="Name of the environment to create")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity of output")
    
    args = parser.parse_args()
    
    setup_environment(env_name=args.env_name, use_conda=args.conda, verbose=not args.quiet)