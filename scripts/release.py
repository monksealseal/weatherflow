#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path

def run_checks():
    """Run pre-release checks"""
    checks = [
        ("Running tests...", ["pytest", "tests/"]),
        ("Checking code format...", ["black", "--check", "weatherflow"]),
        ("Checking imports...", ["isort", "--check", "weatherflow"]),
        ("Running type checker...", ["mypy", "weatherflow"]),
        ("Building documentation...", ["mkdocs", "build"]),
        ("Building package...", ["python", "-m", "build"]),
    ]
    
    for message, command in checks:
        print(message)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed: {' '.join(command)}")
            print(result.stdout)
            print(result.stderr)
            return False
        print("✅ Passed")
    return True

def update_version():
    """Update version number"""
    version = input("Enter new version number (e.g., 0.1.0): ")
    init_file = Path("weatherflow") / "__init__.py"
    with open(init_file, "r") as f:
        content = f.read()
    
    # Update version
    with open(init_file, "w") as f:
        f.write(f'__version__ = "{version}"
')
    
    return version

def main():
    print("WeatherFlow Release Preparation")
    print("==============================")
    
    # Run checks
    if not run_checks():
        sys.exit(1)
    
    # Update version
    version = update_version()
    
    print(f"\nRelease {version} is ready!")
    print("\nNext steps:")
    print("1. Create and push a new git tag:")
    print(f"   git tag v{version}")
    print(f"   git push origin v{version}")
    print("\n2. The GitHub Action will automatically:")
    print("   - Run tests")
    print("   - Build package")
    print("   - Upload to PyPI")
    print("   - Deploy documentation")

if __name__ == "__main__":
    main()
