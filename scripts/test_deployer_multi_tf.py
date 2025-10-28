import os
import glob
import pandas as pd
import numpy as np
from quantylab.rltrader.deployer import ModelDeployer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(BASE, 'data', 'KRW-BTC_multi_tf_sample.csv')
MODEL_DIR = os.path.join(BASE, 'models', 'best')

intervals = ['minute1', 'minute5', 'minute15', 'hour1', 'hour4', 'day']
base_interval = 'minute1'

# find model prefix in models/best
policy_files = glob.glob(os.path.join(MODEL_DIR, '*_policy.mdl'))
if not policy_files:
    print('No policy file found in', MODEL_DIR)
    print('Listing models folder:')
    for p in glob.glob(os.path.join(BASE, 'models', '*')):
        print(' ', p)
    raise SystemExit(1)

policy_path = policy_files[0]
model_name = os.path.basename(policy_path).split('_')[0]
print('Using model_name:', model_name)

if not os.path.exists(DATA_CSV):
    print('Sample CSV not found:', DATA_CSV)
    raise SystemExit(1)

print('Loading sample CSV:', DATA_CSV)
df = pd.read_csv(DATA_CSV, index_col=0, parse_dates=True)

# build tf_data by matching column naming conventions used in deployer
# base interval uses 'close' and 'volume' without suffix; others use suffix like _m5, _m15, _h1, _h4, _d1
suffix_map = {}
for iv in intervals:
    if iv == base_interval:
        suffix_map[iv] = ''
    else:
        s = iv.replace('minute', 'm').replace('hour', 'h')
        if iv == 'day':
            s = 'd1'
        suffix_map[iv] = s

# assemble tf_data
tf_data = {}
for iv in intervals:
    suf = suffix_map[iv]
    if suf == '':
        close_col = 'close'
        vol_col = 'volume'
    else:
        close_col = f'close_{suf}'
        vol_col = f'volume_{suf}'

    if close_col in df.columns and vol_col in df.columns:
        closes = df[close_col].dropna().values.tolist()
        vols = df[vol_col].dropna().values.tolist()
        # ensure oldest->newest order
        tf_data[iv] = {'close': closes[-200:] if len(closes) >= 200 else closes, 'volume': vols[-200:] if len(vols) >= 200 else vols}
        print(f'Found {iv}: close_col={close_col} len={len(tf_data[iv]["close"])}, vol len={len(tf_data[iv]["volume"]) }')
    else:
        print(f'WARNING: columns for {iv} not found ({close_col}, {vol_col})')

if not tf_data:
    print('No TF data assembled, aborting')
    raise SystemExit(1)

# instantiate deployer and run test
deployer = ModelDeployer(model_name=model_name, model_dir=MODEL_DIR)
print('Initializing deployer with multi-TF data...')
deployer.load_initial_multi_tf(tf_data, base_interval=base_interval)

# build latest tick mapping from last elements
tf_latest = {}
for iv, d in tf_data.items():
    if len(d['close']) == 0:
        continue
    tf_latest[iv] = {'close': float(d['close'][-1]), 'volume': float(d['volume'][-1]) if len(d['volume'])>0 else 0.0}

print('Calling on_new_multi_tf with latest sample...')
res = deployer.on_new_multi_tf(tf_latest, base_interval=base_interval, execute=False)
print('Result: action, confidence, trade_unit, portfolio_value ->', res)
print('Done')
