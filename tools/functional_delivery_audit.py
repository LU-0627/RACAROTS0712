"""Static functional delivery audit for RDCAROTS"""
import re
from pathlib import Path

def audit_project(project_root):
    project_root = Path(project_root)
    issues = []
    
    # Check 1: No n_vars // 2
    for py_file in project_root.glob('models/rd_carots/**/*.py'):
        content = py_file.read_text()
        if re.search(r'n_vars\s*//\s*2|shape\[-1\]\s*//\s*2', content):
            issues.append(f"{py_file}: Found n_vars // 2 hardcoding")
    
    # Check 2: No A_delay全零占位
    scorer_file = project_root / 'models/rd_carots/scorer_rd_carots.py'
    if scorer_file.exists():
        content = scorer_file.read_text()
        if 'torch.zeros_like' in content and 'delay' in content:
            if 'compute_delay_score' not in content:
                issues.append(f"{scorer_file}: A_delay appears to be placeholder")
    
    # Check 3: No pass in model_bank load_state_dict
    model_bank_file = project_root / 'models/rd_carots/delaymix/model_bank.py'
    if model_bank_file.exists():
        content = model_bank_file.read_text()
        if re.search(r'def load_state_dict.*?pass', content, re.DOTALL):
            issues.append(f"{model_bank_file}: load_state_dict contains pass")
    
    # Check 4: synthetic_regime_delay registered
    build_file = project_root / 'datasets/build.py'
    if build_file.exists():
        content = build_file.read_text()
        if 'synthetic_regime_delay' not in content:
            issues.append(f"{build_file}: synthetic_regime_delay not registered")
    
    print(f"=== Functional Delivery Audit ===")
    print(f"Scanned {project_root}")
    print(f"Found {len(issues)} issues:")
    for issue in issues:
        print(f"  - {issue}")
    
    return len(issues) == 0

if __name__ == '__main__':
    import sys
    project_root = sys.argv[1] if len(sys.argv) > 1 else '.'
    success = audit_project(project_root)
    sys.exit(0 if success else 1)
