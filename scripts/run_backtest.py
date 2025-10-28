"""Find the best or latest model and run backtest via main.py (test mode).

Usage:
    python scripts/run_backtest.py [--model-name NAME] [--start_date YYYYMMDD] [--end_date YYYYMMDD] [--stock_code SYMBOL]

If --model-name is omitted, the script will try models/best/*_policy.mdl, else fallback to latest *_policy.mdl in models/.
"""
import argparse
import glob
import os
import re
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', help='Base model name (prefix before _<rl>_<net>_policy.mdl)')
parser.add_argument('--start_date', default='20240101')
parser.add_argument('--end_date', default='20251027')
parser.add_argument('--stock_code', default='KRW-BTC')
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models'

def pick_model():
    if args.model_name:
        # try to find matching policy file
        pattern = str(MODELS / f"{args.model_name}_*_policy.mdl")
        files = glob.glob(pattern)
        if files:
            return files[0]
        else:
            raise SystemExit(f'No model file found for name {args.model_name}')

    # prefer models/best
    best_policy = list((MODELS / 'best').glob('*_policy.mdl')) if (MODELS / 'best').exists() else []
    if best_policy:
        # pick first
        return str(best_policy[0])

    # fallback: latest policy in models/
    candidates = list(MODELS.glob('*_policy.mdl'))
    if not candidates:
        raise SystemExit('No policy model files found under models/')
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])

policy_path = pick_model()
print('Using policy model:', policy_path)
# extract base name
m = re.match(r'.*\\([^\\]+)_policy\.mdl$', policy_path)
if not m:
    m = re.match(r'.*/([^/]+)_policy\.mdl$', policy_path)
if not m:
    raise SystemExit('Unable to parse model filename')
base = m.group(1)
# parse rl_method and net from base: expected pattern <name>_<rl>_<net>
parts = base.split('_')
if len(parts) >= 3:
    rl_method = parts[-2]
    net = parts[-1]
    name = '_'.join(parts[:-2])
else:
    # fallback
    rl_method = 'a2c'
    net = 'dnn'
    name = base

print('Detected name:', name, 'rl_method:', rl_method, 'net:', net)

# build command
cmd = [
    'python', 'main.py', '--mode', 'test', '--name', name,
    '--stock_code', args.stock_code, '--rl_method', rl_method, '--net', net,
    '--start_date', args.start_date, '--end_date', args.end_date
]
# set PYTHONPATH
env = os.environ.copy()
env['PYTHONPATH'] = str(ROOT)
print('Running backtest with command:', ' '.join(cmd))
subprocess.run(cmd, env=env)
print('Backtest finished.')
