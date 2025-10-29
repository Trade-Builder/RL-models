"""Select best trained model by scanning output logs and copy its model files to models/best/.

This script looks for `output/train_*` folders containing `params.json` and a training log.
It parses `Max PV:` from the train log and picks the model with the highest Max PV.

Usage:
    python scripts/select_and_keep_best_model.py

After running, the chosen model files (policy & value) will be copied into `models/best/`.
"""

import os
import json
import glob
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'output'
MODELS_DIR = ROOT / 'models'
BEST_DIR = MODELS_DIR / 'best'

LOG_MAX_PV_RE = re.compile(r'Max PV:\s*([0-9,]+(?:\.[0-9]+)?)')


def parse_max_pv_from_log(log_path: Path):
    try:
        text = log_path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return None
    m = LOG_MAX_PV_RE.search(text)
    if not m:
        return None
    num = m.group(1).replace(',', '')
    try:
        return float(num)
    except Exception:
        return None


def find_train_folders():
    if not OUTPUT_DIR.exists():
        return []
    candidates = []
    for p in OUTPUT_DIR.iterdir():
        if p.is_dir() and p.name.startswith('train_'):
            candidates.append(p)
    return candidates


def load_params(folder: Path):
    params_path = folder / 'params.json'
    if not params_path.exists():
        return None
    try:
        return json.loads(params_path.read_text(encoding='utf-8'))
    except Exception:
        return None


def find_log_file(folder: Path):
    # prefer any .log in folder
    logs = list(folder.glob('*.log'))
    if logs:
        return logs[0]
    # otherwise search recursively
    logs = list(folder.rglob('*.log'))
    return logs[0] if logs else None


def main():
    train_folders = find_train_folders()
    if not train_folders:
        print('No train_*/ folders under output/ found.')
        return

    best = None
    best_info = None

    for folder in train_folders:
        params = load_params(folder) or {}
        name = params.get('name') or folder.name.replace('train_', '')
        log_file = find_log_file(folder)
        if not log_file:
            print(f'  skip {folder}: no log file')
            continue
        max_pv = parse_max_pv_from_log(log_file)
        print(f'  {folder.name}: name={name} log={log_file.name} max_pv={max_pv}')
        if max_pv is None:
            continue
        if best is None or max_pv > best:
            best = max_pv
            best_info = {'name': name, 'folder': folder, 'log': log_file, 'max_pv': max_pv}

    if best_info is None:
        print('No valid Max PV entries found in logs.')
        return

    print('\nSelected best model:')
    print(f"  name={best_info['name']} max_pv={best_info['max_pv']:,}")

    # find model files for that name
    pattern_policy = str(MODELS_DIR / f"{best_info['name']}*policy.mdl")
    pattern_value = str(MODELS_DIR / f"{best_info['name']}*value.mdl")
    policy_candidates = glob.glob(pattern_policy)
    value_candidates = glob.glob(pattern_value)

    if not policy_candidates or not value_candidates:
        print('Model files for the selected name were not found in models/.')
        print('  policy pattern ->', pattern_policy)
        print('  value pattern  ->', pattern_value)
        return

    BEST_DIR.mkdir(parents=True, exist_ok=True)

    # Copy the first match of each
    policy_src = Path(policy_candidates[0])
    value_src = Path(value_candidates[0])
    policy_dst = BEST_DIR / policy_src.name
    value_dst = BEST_DIR / value_src.name

    shutil.copy2(policy_src, policy_dst)
    shutil.copy2(value_src, value_dst)

    print('Copied:')
    print('  ', policy_src, '->', policy_dst)
    print('  ', value_src, '->', value_dst)
    print('\nDone. You can now commit only models/best/ to repository.')


if __name__ == '__main__':
    main()
