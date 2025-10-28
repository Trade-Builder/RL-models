import os
from pathlib import Path
import pandas as pd

# ensure package import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantylab.rltrader import data_manager
from quantylab.rltrader.learners import PPOLearner

print('Loading sample multi-tf data...')
df = pd.read_csv('data/KRW-BTC_multi_tf_sample.csv')
print('Raw shape:', df.shape)

df = data_manager.preprocess_crypto_data(df)
print('After preprocess shape:', df.shape)

# chart_data must have 'close' column for Environment
if 'close' not in df.columns:
    raise RuntimeError('chart_data missing close column')

# training features: drop date and keep numeric columns
training_df = df.select_dtypes(include=['number']).copy()
print('Training feature shape:', training_df.shape)

# prepare output path
output_path = os.path.join('output', 'ppo_smoke_test')
os.makedirs(output_path, exist_ok=True)

# set model paths
policy_path = os.path.join('models', 'ppo_smoke_policy.mdl')
value_path = os.path.join('models', 'ppo_smoke_value.mdl')

print('Creating PPOLearner...')
learner = PPOLearner(
    rl_method='ppo', stock_code='KRW-BTC', chart_data=df,
    training_data=training_df, min_trading_price=10000, max_trading_price=1000000,
    net='dnn', num_steps=1, lr=1e-4, discount_factor=0.99,
    num_epoches=2, balance=10000000, start_epsilon=1,
    output_path=output_path, reuse_models=False,
    policy_network_path=policy_path, value_network_path=value_path
)

print('Running PPOLearner for 2 epochs (smoke test)...')
learner.run(learning=True)
print('Smoke run finished. PV:', learner.agent.portfolio_value)

# save models (optional)
os.makedirs('models', exist_ok=True)
learner.policy_network_path = policy_path
learner.value_network_path = value_path
learner.save_models()
print('Saved policy/value to models/ (if created)')
