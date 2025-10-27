import numpy as np
import pandas as pd
import os
from typing import List, Optional

from quantylab.rltrader.data_manager import COLUMNS_CRYPTO_DATA
from quantylab.rltrader.networks import DNN
from quantylab.rltrader.environment import Environment
from quantylab.rltrader.agent import Agent


def compute_features_from_closes(closes: List[float]) -> pd.DataFrame:
    """
    주어진 종가 시간열로부터 학습에 사용 가능한 기술적 지표들을 계산합니다.

    closes: 과거->최신 순서의 종가 리스트 또는 배열. 길이는 200 권장.
    반환: feature DataFrame (index 순서 동일)
    """
    s = pd.Series(closes).astype(float)
    df = pd.DataFrame({'close': s})

    # 변화율
    df['diffratio'] = df['close'].pct_change().fillna(0)

    # 이동평균 비율들
    for ma in (5, 10, 20, 60, 120):
        col_ma = f'close_ma{ma}'
        col_ratio = f'close_ma{ma}_ratio'
        df[col_ma] = df['close'].rolling(ma).mean()
        # ratio: close / ma
        df[col_ratio] = df['close'] / df[col_ma]

    # Bollinger Band ratio (20)
    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    # 클로즈가 lower~upper 사이에서 어느 위치인지 0..1로 정규화
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
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

    # 선택된 feature만 반환 (훈련 시 사용한 컬럼의 교차집합)
    features = {}
    for col in COLUMNS_CRYPTO_DATA:
        if col == 'date':
            continue
        if col in df.columns:
            # 컬럼이 존재하면 사용
            features[col] = df[col]
    features_df = pd.DataFrame(features)
    return features_df


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
        policy_path = os.path.join(self.model_dir, f'{self.model_name}_a2c_dnn_policy.mdl')
        value_path = os.path.join(self.model_dir, f'{self.model_name}_a2c_dnn_value.mdl')
        # 모델이 저장된 형태는 torch.save(model), wrapper DNN을 사용해 load_model을 호출
        # 입력 차원은 나중에 feature 계산 후 설정
        self.policy_path = policy_path if os.path.exists(policy_path) else None
        self.value_path = value_path if os.path.exists(value_path) else None

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
        self.agent = Agent(self.environment, self.initial_balance, self.min_trading_price, self.max_trading_price)

        # 모델 로드
        self.load_models()
        num_features = (features.shape[1] if features is not None else 0) + self.agent.STATE_DIM
        # 정책 네트워크 래퍼 생성 및 로드
        if self.policy_path is not None:
            self.policy = DNN(input_dim=num_features, output_dim=self.agent.NUM_ACTIONS, lr=0.0001)
            try:
                self.policy.load_model(self.policy_path)
            except Exception:
                # fallback: try to load raw torch module
                import torch
                self.policy = torch.load(self.policy_path)
        # 가치 네트워크는 선택적
        if self.value_path is not None:
            self.value = DNN(input_dim=num_features, output_dim=self.agent.NUM_ACTIONS, lr=0.0001)
            try:
                self.value.load_model(self.value_path)
            except Exception:
                import torch
                self.value = torch.load(self.value_path)

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
            try:
                pred_policy = self.policy.predict(sample)
            except Exception:
                # if policy is a raw torch model
                import torch
                x = torch.from_numpy(sample).float()
                out = self.policy(x).detach().cpu().numpy().flatten()
                pred_policy = out
        if self.value is not None and pred_policy is None:
            try:
                pred_value = self.value.predict(sample)
            except Exception:
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
