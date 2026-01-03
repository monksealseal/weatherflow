"""
Download script for large Model Zoo models hosted externally.

For models too large to include in the repository, this script downloads
them from cloud storage.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import hashlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# External model registry
# In production, these would be hosted on cloud storage (S3, GCS, etc.)
EXTERNAL_MODELS = {
    'wf_global_multivariable_v2': {
        'url': 'https://example.com/models/wf_global_multivariable_v2.pt',
        'size_mb': 256,
        'sha256': 'dummy_hash_would_go_here',
        'description': 'Large multi-variable global forecast model'
    },
    'wf_high_resolution_regional_v1': {
        'url': 'https://example.com/models/wf_high_resolution_regional_v1.pt',
        'size_mb': 512,
        'sha256': 'dummy_hash_would_go_here',
        'description': 'High-resolution regional model for North America'
    }
}


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Verify file integrity using SHA256 checksum."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_sha256


def download_model(model_id: str, output_dir: Optional[Path] = None, verify: bool = True) -> Path:
    """
    Download a model from external storage.

    Args:
        model_id: Unique model identifier
        output_dir: Directory to save the model (default: auto-detect from model zoo)
        verify: Verify checksum after download

    Returns:
        Path to the downloaded model file
    """
    if model_id not in EXTERNAL_MODELS:
        print(f"Error: Model '{model_id}' is not available for download.")
        print("\nAvailable external models:")
        for mid, info in EXTERNAL_MODELS.items():
            print(f"  {mid}: {info['description']} ({info['size_mb']} MB)")
        sys.exit(1)

    model_info = EXTERNAL_MODELS[model_id]

    # Determine output directory
    if output_dir is None:
        # Find the model's directory in the model zoo
        model_zoo_dir = Path(__file__).parent
        # Search for the model card to determine the correct directory
        for model_card_path in model_zoo_dir.rglob("model_card.json"):
            with open(model_card_path) as f:
                data = json.load(f)
                if data['model_id'] == model_id:
                    output_dir = model_card_path.parent
                    break

        if output_dir is None:
            print(f"Error: Could not find model card for '{model_id}'")
            print("Please create the model card first or specify --output-dir")
            sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{model_id}.pt"

    # Check if already downloaded
    if output_file.exists() and not verify:
        print(f"Model already exists at {output_file}")
        return output_file

    print(f"Downloading {model_id}...")
    print(f"  URL: {model_info['url']}")
    print(f"  Size: {model_info['size_mb']} MB")
    print(f"  Destination: {output_file}")

    # In production, this would use urllib, requests, or a cloud SDK
    # For now, provide instructions
    print("\n" + "="*60)
    print("DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\nThis model is hosted externally. To download it:")
    print(f"\n1. Download from: {model_info['url']}")
    print(f"2. Save to: {output_file}")
    print(f"3. Verify SHA256: {model_info['sha256']}")
    print("\nAlternatively, you can use wget or curl:")
    print(f"\n  wget -O {output_file} {model_info['url']}")
    print(f"\n  curl -o {output_file} {model_info['url']}")
    print("\n" + "="*60)

    # Simulated download (would be actual download in production)
    print("\nNote: Automatic download not implemented yet.")
    print("Please download manually using the instructions above.")

    return output_file


def list_external_models():
    """List all models available for external download."""
    print("\nModels Available for Download:")
    print("="*60)
    for model_id, info in EXTERNAL_MODELS.items():
        print(f"\n{model_id}")
        print(f"  Description: {info['description']}")
        print(f"  Size: {info['size_mb']} MB")
        print(f"  Download: python model_zoo/download_model.py {model_id}")


def main():
    parser = argparse.ArgumentParser(
        description='Download large Model Zoo models from external storage'
    )
    parser.add_argument(
        'model_id',
        nargs='?',
        help='Model ID to download (omit to list available models)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory (default: auto-detect from model card)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip checksum verification'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available external models'
    )

    args = parser.parse_args()

    if args.list or args.model_id is None:
        list_external_models()
        return

    download_model(
        model_id=args.model_id,
        output_dir=args.output_dir,
        verify=not args.no_verify
    )


if __name__ == '__main__':
    main()
