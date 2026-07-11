"""
Build server deployment package

This script creates RDCAROTS_SERVER_PACKAGE.zip for offline Linux server deployment.
"""

import os
import shutil
import zipfile
import hashlib
from pathlib import Path
import json


def calculate_sha256(filepath):
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def should_exclude(path, exclude_patterns):
    """Check if path should be excluded."""
    path_str = str(path)
    for pattern in exclude_patterns:
        if pattern in path_str:
            return True
    return False


def build_server_package(source_root, output_dir):
    """
    Build server deployment package.

    Args:
        source_root: Root of CAROTS project
        output_dir: Where to create RDCAROTS_SERVER_PACKAGE
    """
    source_root = Path(source_root)
    output_dir = Path(output_dir)

    package_dir = output_dir / 'RDCAROTS_SERVER_PACKAGE'
    project_dir = package_dir / 'project'

    # Clean existing package
    if package_dir.exists():
        shutil.rmtree(package_dir)

    package_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    # Exclude patterns
    exclude_patterns = [
        '__pycache__',
        '.pytest_cache',
        '.idea',
        '.vscode',
        '.git',
        'data/SWaT',
        'data/WADI',
        'data/synthetic_regime_delay',
        'results/',
        '.pyc',
        '.pyo',
        '.pyd',
        '.so',
        '.dll',
        '.egg-info',
        'checkpoint_',
        '.DS_Store'
    ]

    # Files to include
    include_dirs = [
        'models',
        'datasets',
        'layers',
        'utils',
        'tests',
        'scripts',
        'configs',
        'docs'
    ]

    include_files = [
        'main.py',
        'run_rd_carots.py',
        'config.py',
        'trainer.py',
        'threshold.py',
        'requirements.txt',
        'requirements-server.txt',
        'requirements-server-lock.txt',
        'environment-server.yml',
        'RUN_ON_OFFLINE_SERVER.md',
        'FILE_MANIFEST.txt'
    ]

    # Copy directories
    for dir_name in include_dirs:
        src_dir = source_root / dir_name
        if src_dir.exists():
            dst_dir = project_dir / dir_name
            shutil.copytree(src_dir, dst_dir, ignore=lambda d, files: [
                f for f in files if should_exclude(Path(d) / f, exclude_patterns)
            ])
            print(f"Copied directory: {dir_name}")

    # Copy files
    for file_name in include_files:
        src_file = source_root / file_name
        if src_file.exists():
            dst_file = project_dir / file_name
            shutil.copy2(src_file, dst_file)
            print(f"Copied file: {file_name}")

    # Copy root-level docs to package root
    root_docs = [
        'RUN_ON_OFFLINE_SERVER.md',
        'SERVER_TEST_CHECKLIST.md',
        'DATA_MANIFEST.md'
    ]
    for doc in root_docs:
        src = source_root / doc
        if src.exists():
            dst = package_dir / doc
            shutil.copy2(src, dst)
            print(f"Copied root doc: {doc}")

    # Create offline directory
    offline_dir = package_dir / 'offline' / 'wheels'
    offline_dir.mkdir(parents=True, exist_ok=True)
    (offline_dir / '.gitkeep').touch()

    # Generate file manifest with SHA256
    manifest = []
    sha256sums = []

    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if not should_exclude(Path(root) / d, exclude_patterns)]

        for file in files:
            filepath = Path(root) / file
            rel_path = filepath.relative_to(package_dir)

            manifest.append(str(rel_path))

            sha256 = calculate_sha256(filepath)
            sha256sums.append(f"{sha256}  {rel_path}")

    # Save file manifest
    manifest_path = package_dir / 'FILE_MANIFEST.txt'
    with open(manifest_path, 'w') as f:
        f.write('\n'.join(sorted(manifest)))
    print(f"Generated FILE_MANIFEST.txt ({len(manifest)} files)")

    # Save SHA256SUMS
    sha256sums_path = package_dir / 'PACKAGE_SHA256SUMS.txt'
    with open(sha256sums_path, 'w') as f:
        f.write('\n'.join(sorted(sha256sums)))
    print(f"Generated PACKAGE_SHA256SUMS.txt")

    # Generate README_FIRST.txt
    readme_path = package_dir / 'README_FIRST.txt'
    with open(readme_path, 'w') as f:
        f.write("""RDCAROTS Server Deployment Package

QUICK START:
1. Extract this package on your Linux server
2. Read RUN_ON_OFFLINE_SERVER.md for complete instructions
3. Set environment variables (see below)
4. Run scripts/server/RUN_ALL_SERVER.sh

ENVIRONMENT VARIABLES:
export PROJECT_ROOT=$(pwd)/project
export DATA_ROOT=/path/to/your/data
export OUTPUT_ROOT=$PROJECT_ROOT/results/rd_carots
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0

FIRST COMMAND:
cd project
bash scripts/server/00_check_environment.sh

For detailed instructions, see RUN_ON_OFFLINE_SERVER.md

NO INTERNET REQUIRED - This package contains everything needed for offline deployment.
""")

    # Create ZIP
    zip_path = output_dir / 'RDCAROTS_SERVER_PACKAGE.zip'
    print(f"Creating ZIP archive: {zip_path}")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                filepath = Path(root) / file
                arcname = filepath.relative_to(output_dir)
                zipf.write(filepath, arcname)

    # Calculate ZIP SHA256
    zip_sha256 = calculate_sha256(zip_path)
    sha256_file = output_dir / 'RDCAROTS_SERVER_PACKAGE.zip.sha256'
    with open(sha256_file, 'w') as f:
        f.write(f"{zip_sha256}  RDCAROTS_SERVER_PACKAGE.zip\n")

    print(f"\n{'='*80}")
    print("Package build complete!")
    print(f"{'='*80}")
    print(f"Package directory: {package_dir}")
    print(f"ZIP file: {zip_path}")
    print(f"ZIP size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"ZIP SHA256: {zip_sha256}")
    print(f"SHA256 file: {sha256_file}")
    print(f"Total files: {len(manifest)}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        source_root = sys.argv[1]
    else:
        source_root = Path(__file__).parent.parent

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = Path(__file__).parent.parent / 'dist'

    build_server_package(source_root, output_dir)
