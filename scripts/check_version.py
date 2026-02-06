#!/usr/bin/env python
"""Check that version numbers are consistent across all project files."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VERSION_SOURCES = {
    "weatherflow/version.py": re.compile(r'__version__\s*=\s*"([^"]+)"'),
    "pyproject.toml": re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE),
    "setup.py": re.compile(r'version\s*=\s*"([^"]+)"'),
}


def main() -> int:
    versions = {}
    errors = []

    for filepath, pattern in VERSION_SOURCES.items():
        full_path = ROOT / filepath
        if not full_path.exists():
            errors.append(f"  {filepath}: FILE NOT FOUND")
            continue
        content = full_path.read_text()
        match = pattern.search(content)
        if not match:
            errors.append(f"  {filepath}: version string not found")
            continue
        versions[filepath] = match.group(1)

    if errors:
        print("Version check errors:")
        for e in errors:
            print(e)
        return 1

    unique = set(versions.values())
    if len(unique) == 1:
        version = unique.pop()
        print(f"Version {version} is consistent across all files:")
        for f, v in versions.items():
            print(f"  {f}: {v}")
        return 0

    print("Version MISMATCH detected:")
    for f, v in versions.items():
        print(f"  {f}: {v}")
    print("\nAll files must declare the same version.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
