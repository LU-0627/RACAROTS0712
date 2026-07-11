"""
Verify server deployment package integrity
"""

import zipfile
import hashlib
from pathlib import Path
import re


def verify_package(package_path):
    """
    Verify RDCAROTS_SERVER_PACKAGE.zip integrity.

    Args:
        package_path: Path to RDCAROTS_SERVER_PACKAGE.zip

    Returns:
        List of issues found (empty if all checks pass)
    """
    issues = []
    package_path = Path(package_path)

    if not package_path.exists():
        issues.append(f"Package not found: {package_path}")
        return issues

    # Check if ZIP is valid
    try:
        with zipfile.ZipFile(package_path, 'r') as zipf:
            file_list = zipf.namelist()
            print(f"✓ ZIP is valid ({len(file_list)} files)")
    except Exception as e:
        issues.append(f"Invalid ZIP: {e}")
        return issues

    # Required files
    required_files = [
        'RDCAROTS_SERVER_PACKAGE/project/run_rd_carots.py',
        'RDCAROTS_SERVER_PACKAGE/project/models/rd_carots/__init__.py',
        'RDCAROTS_SERVER_PACKAGE/project/models/rd_carots/trainer_rd_carots.py',
        'RDCAROTS_SERVER_PACKAGE/project/configs/rd_carots/synthetic.yaml',
        'RDCAROTS_SERVER_PACKAGE/project/scripts/server/RUN_ALL_SERVER.sh',
        'RDCAROTS_SERVER_PACKAGE/RUN_ON_OFFLINE_SERVER.md',
        'RDCAROTS_SERVER_PACKAGE/FILE_MANIFEST.txt',
        'RDCAROTS_SERVER_PACKAGE/PACKAGE_SHA256SUMS.txt'
    ]

    for req_file in required_files:
        if req_file not in file_list:
            issues.append(f"Missing required file: {req_file}")

    # Check for excluded patterns
    exclude_patterns = [
        '__pycache__',
        '.pyc',
        'data/SWaT',
        'data/WADI'
    ]

    for file in file_list:
        for pattern in exclude_patterns:
            if pattern in file:
                issues.append(f"Should be excluded: {file}")
                break

    # Check for Windows paths
    with zipfile.ZipFile(package_path, 'r') as zipf:
        for file in file_list:
            if file.endswith('.py') or file.endswith('.sh') or file.endswith('.yaml'):
                try:
                    content = zipf.read(file).decode('utf-8', errors='ignore')

                    # Check for Windows absolute paths
                    if re.search(r'[A-Z]:\\', content):
                        issues.append(f"Windows path in {file}")

                    # Check for hardcoded .cuda()
                    if '.cuda()' in content:
                        issues.append(f"Hardcoded .cuda() in {file}")

                except Exception as e:
                    pass

    # Check shell scripts
    with zipfile.ZipFile(package_path, 'r') as zipf:
        for file in file_list:
            if file.endswith('.sh'):
                try:
                    content = zipf.read(file).decode('utf-8')

                    # Check shebang
                    if not content.startswith('#!/'):
                        issues.append(f"Missing shebang in {file}")

                    # Check set -euo pipefail
                    if 'set -euo pipefail' not in content:
                        issues.append(f"Missing 'set -euo pipefail' in {file}")

                except Exception as e:
                    pass

    if len(issues) == 0:
        print("✓ All checks passed")
    else:
        print(f"✗ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")

    return issues


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        package_path = sys.argv[1]
    else:
        package_path = Path(__file__).parent.parent / 'dist' / 'RDCAROTS_SERVER_PACKAGE.zip'

    issues = verify_package(package_path)
    sys.exit(0 if len(issues) == 0 else 1)
