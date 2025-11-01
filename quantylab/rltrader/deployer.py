import numpy as np
import pandas as pd
import os
from typing import List, Optional, Dict

from quantylab.rltrader.data_manager import COLUMNS_CRYPTO_DATA
from quantylab.rltrader.networks import DNN
from quantylab.rltrader.environment import Environment
from quantylab.rltrader.agent import Agent


def compute_features_from_closes(closes: List[float]) -> pd.DataFrame:
    """
    주어진 종가 시간열로부터 학습에 사용 가능한 기술적 지표들을 계산합니다.

    closes: 과거->최신 순서의 종가 리스트 또는 배열. 길이는 200 권장.
    반환: feature DataFrame (index 순서 동일, 26개 features for model input)
    """
    s = pd.Series(closes).astype(float)
    df = pd.DataFrame({
        'close': s,
        'open': s,  # close로 대체
        'high': s,  # close로 대체
        'low': s,   # close로 대체
        'volume': 0  # 더미 값
    })

    # OHLC 비율들
    df['open_lastclose_ratio'] = df['open'] / df['close'].shift(1)
    df['high_close_ratio'] = df['high'] / df['close']
    df['low_close_ratio'] = df['low'] / df['close']
    
    # 변화율
    df['diffratio'] = df['close'].pct_change()
    
    # 거래량 비율 (더미 값)
    df['volume_lastvolume_ratio'] = 0

    # 이동평균 비율들 (close & volume)
    for ma in (5, 10, 20, 60, 120):
        df[f'close_ma{ma}_ratio'] = df['close'] / df['close'].rolling(ma).mean()
        df[f'volume_ma{ma}_ratio'] = 0  # 더미 값

    # Bollinger Band ratio (20)
    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    df['bb_ratio_20'] = (df['close'] - lower) / (upper - lower)

    # RSI(14)
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD diff (12/26, signal 9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_diff'] = macd - macd_signal

    # 결측 처리: 앞뒤로 보간 후 0으로 채움
    df = df.bfill().ffill().fillna(0)

    # COLUMNS_CRYPTO_DATA 순서대로 반환 (date 제외한 23개)
    result_cols = [col for col in COLUMNS_CRYPTO_DATA if col != 'date']
    return df[result_cols]


def _compute_indicators_for_tf(closes: List[float], volumes: List[float]) -> pd.DataFrame:
    """Compute indicators for a single timeframe given closes and volumes.

    This mirrors `UpbitDataCollector.add_technical_indicators` but accepts raw arrays.
    """
    s = pd.Series(closes).astype(float)
    v = pd.Series(volumes).astype(float) if volumes is not None else pd.Series([0]*len(s))
    df = pd.DataFrame({'close': s, 'volume': v})

    # ratios
    df['open_lastclose_ratio'] = df['close'] / df['close'].shift(1) - 1
    df['high_close_ratio'] = df['close'] / df['close'] - 1
    df['low_close_ratio'] = df['close'] / df['close'] - 1
    df['diffratio'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['volume_lastvolume_ratio'] = df['volume'] / df['volume'].shift(1) - 1

    for period in [5, 10, 20, 60, 120]:
        df[f'close_ma{period}_ratio'] = df['close'] / df['close'].rolling(period).mean() - 1
        df[f'volume_ma{period}_ratio'] = df['volume'] / df['volume'].rolling(period).mean() - 1

    # Bollinger Bands
    period = 20
    ma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    df[f'bb_ratio_{period}'] = (df['close'] - lower) / (upper - lower)

    # RSI(14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD diff
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_diff'] = macd - macd_signal

    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    return df


class ModelDeployer:
    """배포용 예측기. 실시간으로 들어오는 종가를 받아 내부에서 전처리/상태를 관리하고 액션을 반환합니다.

    사용법 요약:
      deployer = ModelDeployer(model_name='20251027124731')
      deployer.load_initial_closes(list_of_200_closes)
      action, confidence, trade_unit = deployer.on_new_close(new_close, execute=False)
    """

    def __init__(self, model_name: str, initial_balance: int = 100000000,
                 min_trading_price: int = 100, max_trading_price: int = 10000000,
                 model_dir: str = 'models'):
        self.model_name = model_name
        self.model_dir = model_dir
        # 내부 시계열 저장
        self.closes: List[float] = []
        self.chart_data: Optional[pd.DataFrame] = None

        # 환경 및 에이전트(상태 유지)
        self.environment = None
        self.agent = None

        # 모델 로드 준비
        self.policy = None
        self.value = None

        self.initial_balance = initial_balance
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

    def load_models(self):
        # model paths are left generic (we will try typical suffixes)
        cand_policy = os.path.join(self.model_dir, f'{self.model_name}_*_policy.mdl')
        # try a2c/ppo names explicitly
        policy_path = os.path.join(self.model_dir, f'{self.model_name}_a2c_dnn_policy.mdl')
        if not os.path.exists(policy_path):
            # fallback: search any matching policy file starting with model_name
            matches = glob = __import__('glob').glob(os.path.join(self.model_dir, f'{self.model_name}_*_policy.mdl'))
            policy_path = matches[0] if matches else None
        value_path = os.path.join(self.model_dir, f'{self.model_name}_a2c_dnn_value.mdl')
        if not os.path.exists(value_path):
            matches = __import__('glob').glob(os.path.join(self.model_dir, f'{self.model_name}_*_value.mdl'))
            value_path = matches[0] if matches else None

        self.policy_path = policy_path if policy_path and os.path.exists(policy_path) else None
        self.value_path = value_path if value_path and os.path.exists(value_path) else None

    def load_initial_closes(self, closes: List[float]):
        """초기 과거 데이터(최소 120~200 권장)를 로드하여 내부 상태를 초기화합니다."""
        assert len(closes) > 0
        self.closes = list(map(float, closes))
        # 전처리해서 chart_data 생성
        features = compute_features_from_closes(self.closes)
        self.chart_data = features

        # 환경/에이전트 초기화
        self.environment = Environment(self.chart_data)
        self.environment.reset()
        self.environment.initial_balance = self.initial_balance
        # Call observe to set the current observation (needed for get_price())
        self.environment.observe()
        self.agent = Agent(self.environment, self.initial_balance, self.min_trading_price, self.max_trading_price)
        # 초기 에이전트 상태 초기화 (포트폴리오 가치 등)
        self.agent.reset()

        # 모델 로드
        self.load_models()
        num_features = (features.shape[1] if features is not None else 0) + self.agent.STATE_DIM
        # 정책 네트워크 래퍼 생성 및 로드
        if self.policy_path is not None:
            print(f"deployer: loading policy model from {self.policy_path}")
            import torch
            self.policy = torch.load(self.policy_path, weights_only=False)
            # Set model to evaluation mode
            if hasattr(self.policy, 'eval'):
                self.policy.eval()
            if hasattr(self.policy, 'model') and hasattr(self.policy.model, 'eval'):
                self.policy.model.eval()
            print(f"deployer: policy model loaded successfully")
        # 가치 네트워크는 선택적
        if self.value_path is not None:
            print(f"deployer: loading value model from {self.value_path}")
            import torch
            self.value = torch.load(self.value_path, weights_only=False)
            # Set model to evaluation mode
            if hasattr(self.value, 'eval'):
                self.value.eval()
            if hasattr(self.value, 'model') and hasattr(self.value.model, 'eval'):
                self.value.model.eval()
            print(f"deployer: value model loaded successfully")

    def _append_close_and_update(self, close: float):
        self.closes.append(float(close))
        # keep memory manageable: keep last 1000
        if len(self.closes) > 2000:
            self.closes = self.closes[-2000:]
        features = compute_features_from_closes(self.closes)
        # append only last row to chart_data
        last_row = features.iloc[[-1]].reset_index(drop=True)
        if self.chart_data is None:
            self.chart_data = last_row
        else:
            self.chart_data = pd.concat([self.chart_data, last_row], ignore_index=True)

    def on_new_close(self, close: float, execute: bool = False):
        """새로운 종가가 들어올 때 호출.

        Args:
            close: 최신 종가
            execute: True면 Agent.act()를 실행해 내부 상태(잔고/보유)를 갱신

        Returns:
            action(int), confidence(float), trade_unit(float), portfolio_value(float)
        """
        assert self.agent is not None, "call load_initial_closes() first"
        # append and update features/chart_data
        self._append_close_and_update(close)

        # 환경에서 다음 관측으로 이동
        obs = self.environment.observe()
        # build sample: training feature row + agent states
        feature_cols = [c for c in self.chart_data.columns if c != 'date']
        row = self.chart_data.iloc[self.environment.idx][feature_cols].values.astype(np.float32)
        states = np.array(self.agent.get_states(), dtype=np.float32)
        sample = np.concatenate([row, states]).reshape((1, len(row) + len(states)))

        # predict
        pred_policy = None
        pred_value = None
        if self.policy is not None:
            # Use raw torch model directly
            import torch
            x = torch.from_numpy(sample).float()
            out = self.policy(x).detach().cpu().numpy().flatten()
            pred_policy = out
        if self.value is not None and pred_policy is None:
            import torch
            x = torch.from_numpy(sample).float()
            out = self.value(x).detach().cpu().numpy().flatten()
            pred_value = out

        # 결정
        if pred_policy is not None:
            action = int(np.argmax(pred_policy))
            confidence = float(pred_policy[action])
        elif pred_value is not None:
            action = int(np.argmax(pred_value))
            # confidence는 sigmoid(value)
            confidence = 1.0 / (1.0 + np.exp(-float(pred_value[action])))
        else:
            # 모델이 없을 경우 HOLD
            action = Agent.ACTION_HOLD
            confidence = 0.0

        # 매매 수량 결정
        trade_unit = float(self.agent.decide_trading_unit(confidence))

        # 실행 옵션: 실제 에이전트 실행(잔고/포지션 변경)
        if execute and action in (Agent.ACTION_BUY, Agent.ACTION_SELL):
            self.agent.act(action, confidence)

        return int(action), float(confidence), float(trade_unit), float(self.agent.portfolio_value)

    # --- New multi-timeframe methods ---
    def load_initial_multi_tf(self, tf_data: Dict[str, Dict[str, List[float]]], base_interval: str = 'minute1'):
        """Initialize deployer with multi-timeframe historical arrays.

        tf_data: mapping from interval string (e.g. 'minute1', 'minute5') to dict with keys:
            'close': list[float], 'volume': list[float]
        All lists should be ordered from oldest->newest and length >= 120 (권장 200).
        
        NOTE: This model was trained on single timeframe data, so only base_interval is used for prediction.
        Other timeframes are ignored.
        """
        # Extract base interval data only (model was trained on single TF)
        if base_interval not in tf_data:
            raise ValueError(f"base_interval '{base_interval}' not found in tf_data keys: {list(tf_data.keys())}")
        
        base_data = tf_data[base_interval]
        closes = base_data.get('close', [])
        volumes = base_data.get('volume', [0] * len(closes))
        
        # Compute indicators for base interval only
        df_ind = _compute_indicators_for_tf(closes, volumes)
        df_ind = df_ind.reset_index().rename(columns={'index': 'date'})
        
        # Take last row as initial state
        self.chart_data = df_ind.iloc[-1:].reset_index(drop=True)
        
        # final cleanup
        self.chart_data = self.chart_data.replace([np.inf, -np.inf], np.nan).fillna(0)

        # initialize environment and agent (environment expects 'close' for price)
        # if base_interval produced 'close' column (no suffix), good; otherwise try to map
        if 'close' not in self.chart_data.columns:
            # try base interval mapping
            base_suffix = base_interval.replace('minute', 'm').replace('hour', 'h')
            if base_interval == 'day':
                base_suffix = 'd1'
            mapped = f'close_{base_suffix}'
            if mapped in self.chart_data.columns:
                self.chart_data['close'] = self.chart_data[mapped]
            else:
                # as last resort, pick any close_* column
                close_cols = [c for c in self.chart_data.columns if c.startswith('close')]
                if close_cols:
                    self.chart_data['close'] = self.chart_data[close_cols[-1]]

        # Validate close value
        close_val = float(self.chart_data['close'].iloc[0])
        assert close_val > 0, f"deployer: invalid close value: {close_val}"

        self.environment = Environment(self.chart_data)
        self.environment.reset()
        self.environment.initial_balance = self.initial_balance
        # Call observe to set the current observation (needed for get_price())
        self.environment.observe()
        self.agent = Agent(self.environment, self.initial_balance, self.min_trading_price, self.max_trading_price)
        # 초기 에이전트 상태 초기화
        try:
            self.agent.reset()
        except Exception:
            pass

        # load models now that we can compute feature dimension
        # chart_data may contain a 'date' column which is not used as a feature
        feature_cols = [c for c in self.chart_data.columns if c != 'date']
        num_features = (len(feature_cols) if self.chart_data is not None else 0) + self.agent.STATE_DIM
        self.load_models()
        # Try loading a full torch model first (preserves original architecture)
        self.expected_input_dim = num_features
        if self.policy_path is not None:
            try:
                import torch
                # attempt to load full model object
                loaded = torch.load(self.policy_path, weights_only=False)
                # if loaded is a Network wrapper with predict, keep it; otherwise it's likely a raw torch.nn.Module
                self.policy = loaded
                # Set model to evaluation mode
                if hasattr(self.policy, 'eval'):
                    self.policy.eval()
                if hasattr(self.policy, 'model') and hasattr(self.policy.model, 'eval'):
                    self.policy.model.eval()
                # try to infer expected input dim from first BatchNorm1d (if present)
                try:
                    bn_dim = None
                    for m in self.policy.modules():
                        import torch as _torch
                        if isinstance(m, _torch.nn.BatchNorm1d):
                            bn_dim = m.num_features
                            break
                    if bn_dim is not None:
                        # bn_dim should represent the full input dimension the model expects
                        self.expected_input_dim = int(bn_dim)
                except Exception:
                    # leave expected_input_dim as computed
                    pass
            except Exception:
                # fallback: create DNN wrapper and try to load state_dict or model
                self.policy = DNN(input_dim=num_features, output_dim=self.agent.NUM_ACTIONS, lr=0.0001)
                try:
                    self.policy.load_model(self.policy_path)
                except Exception:
                    # if loading failed, leave wrapper as-is
                    pass
                # Set model to evaluation mode
                if hasattr(self.policy, 'eval'):
                    self.policy.eval()
                if hasattr(self.policy, 'model') and hasattr(self.policy.model, 'eval'):
                    self.policy.model.eval()
                # try to infer actual input dim from the (possibly replaced) model's BatchNorm
                try:
                    bn_dim = None
                    model_obj = getattr(self.policy, 'model', self.policy)
                    for m in model_obj.modules():
                        import torch as _torch
                        if isinstance(m, _torch.nn.BatchNorm1d):
                            bn_dim = m.num_features
                            break
                    if bn_dim is not None:
                        self.expected_input_dim = int(bn_dim)
                        # if we created a DNN wrapper then update its input_dim to match the loaded params
                        try:
                            setattr(self.policy, 'input_dim', int(bn_dim))
                        except Exception:
                            pass
                    else:
                        self.expected_input_dim = getattr(self.policy, 'input_dim', num_features)
                except Exception:
                    self.expected_input_dim = getattr(self.policy, 'input_dim', num_features)

    def on_new_multi_tf(self, tf_latest: Dict[str, Dict[str, float]], base_interval: str = 'minute1', execute: bool = False):
        """Accepts a dict mapping interval -> {'close': float, 'volume': float} for latest tick and returns action.
        This updates internal chart_data by recomputing last-row indicators and predicting.
        
        NOTE: This model was trained on single timeframe data, so only base_interval is used for prediction.
        Other timeframes are ignored.
        """
        assert self.chart_data is not None, 'call load_initial_multi_tf first'
        
        # Use only base_interval data (model was trained on single TF)
        if base_interval not in tf_latest:
            raise ValueError(f"base_interval '{base_interval}' not found in tf_latest keys: {list(tf_latest.keys())}")
        
        base_data = tf_latest[base_interval]
        close = float(base_data.get('close', 0.0))
        vol = float(base_data.get('volume', 0.0))
        
        # Compute indicator row for base interval only
        df_ind = _compute_indicators_for_tf([close], [vol])
        new_row = df_ind.iloc[-1:].reset_index(drop=True)
        
        # Update chart_data with new values
        for col in new_row.columns:
            if col in self.chart_data.columns:
                self.chart_data.at[0, col] = new_row[col].iloc[0]

        # build sample and predict using same logic as on_new_close
        feature_cols = [c for c in self.chart_data.columns if c != 'date']
        row = self.chart_data.iloc[0][feature_cols].values.astype(np.float32)
        states = np.array(self.agent.get_states(), dtype=np.float32)
        sample = np.concatenate([row, states]).reshape((1, len(row) + len(states)))

        pred_policy = None
        pred_value = None
        # ensure sample length matches expected input dim (if known)
        expected = getattr(self, 'expected_input_dim', sample.shape[1])
        if sample.shape[1] != expected:
            # pad or truncate features (best-effort). Warn the user.
            if sample.shape[1] < expected:
                diff = expected - sample.shape[1]
                pad = np.zeros((1, diff), dtype=sample.dtype)
                sample = np.concatenate([sample, pad], axis=1)
            else:
                sample = sample[:, :expected]

        if self.policy is not None:
            # prefer wrapper.predict if available
            try:
                if hasattr(self.policy, 'predict'):
                    pred_policy = self.policy.predict(sample)
                else:
                    import torch
                    x = torch.from_numpy(sample).float()
                    if x.ndim == 1:
                        x = x.unsqueeze(0)
                    model_obj = getattr(self.policy, 'model', self.policy)
                    out = model_obj(x).detach().cpu().numpy().flatten()
                    pred_policy = out
            except Exception as e:
                # fallback: try direct model call if wrapper failed
                try:
                    import torch
                    x = torch.from_numpy(sample).float()
                    if x.ndim == 1:
                        x = x.unsqueeze(0)
                    model_obj = getattr(self.policy, 'model', self.policy)
                    out = model_obj(x).detach().cpu().numpy().flatten()
                    pred_policy = out
                except Exception:
                    raise

        if pred_policy is not None:
            action = int(np.argmax(pred_policy))
            confidence = float(pred_policy[action])
        else:
            action = Agent.ACTION_HOLD
            confidence = 0.0

        trade_unit = float(self.agent.decide_trading_unit(confidence))
        if execute and action in (Agent.ACTION_BUY, Agent.ACTION_SELL):
            self.agent.act(action, confidence)

        return int(action), float(confidence), float(trade_unit), float(self.agent.portfolio_value)
