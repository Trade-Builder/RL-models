import pandas as pd
import numpy as np
from pathlib import Path
from quantylab.rltrader.data_upbit import UpbitDataCollector

COLUMNS_CRYPTO_DATA = [
    'date', 'open', 'high', 'low', 'close', 'volume',
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'diffratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
    'bb_ratio_20', 'rsi_14', 'macd_diff'
]

def load_crypto_data(fpath, date_from=None, date_to=None):
    """
    암호화폐 데이터 로드 함수
    
    Args:
        fpath: CSV 파일 경로
        date_from: 시작 날짜 (YYYYMMDD 또는 YYYYMMDDHHMM)
        date_to: 종료 날짜 (YYYYMMDD 또는 YYYYMMDDHHMM)
    """
    # 먼저 일반적으로 'date' 컬럼이 있는 CSV를 시도
    df = pd.read_csv(fpath, thousands=',')

    # 만약 'date' 컬럼이 없으면, CSV가 index로 날짜를 저장했을 수 있으니
    # 첫 번째 컬럼을 인덱스로 읽어와서 'date' 컬럼으로 변환
    if 'date' not in df.columns:
        try:
            df = pd.read_csv(fpath, thousands=',', index_col=0)
            df.index.name = 'date'
            df = df.reset_index()
        except Exception:
            # 실패하면 기존 df 그대로 두고 이후에 KeyError가 발생하도록 함
            pass

    # 날짜 컬럼을 datetime으로 변환하여 비교 (입력은 'YYYYMMDD' 또는 'YYYYMMDDHHMM' 등 자유 형식 허용)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
        df = df.dropna(subset=['date'])
        if date_from is not None:
            try:
                df = df[df['date'] >= pd.to_datetime(str(date_from))]
            except Exception:
                pass
        if date_to is not None:
            try:
                df = df[df['date'] <= pd.to_datetime(str(date_to))]
            except Exception:
                pass
    
    # 결측치 제거
    df = df.dropna()
    
    return df

def preprocess_crypto_data(df):
    """
    데이터 전처리
    """
    # 인덱스 초기화
    df = df.reset_index(drop=True)
    
    # 무한대 값 제거
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df

def split_data(df, train_ratio=0.8):
    """
    학습/검증 데이터 분리
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    return train_df, val_df


def load_multi_tf_crypto_data(ticker="KRW-BTC", days=365,
                              intervals=None, base_interval='minute1'):
    """Collect multi-timeframe data from Upbit and merge into a single DataFrame.

    intervals: list of upbit interval strings, e.g. ['minute1','minute5','minute15','minute60','minute240','day']
    base_interval: the smallest timeframe to align rows to (default 'minute1')
    """
    if intervals is None:
        intervals = ['minute1', 'minute5', 'minute15', 'minute60', 'minute240', 'day']

    collector = UpbitDataCollector(ticker=ticker)
    dfs = {}
    # collect and compute indicators per timeframe
    for interval in intervals:
        print(f"Collecting {interval} for {ticker} ...")
        df = collector.collect_large_dataset(interval=interval, days=days)
        if df is None or df.empty:
            print(f"Warning: no data for {interval}")
            continue
        df_ind = collector.add_technical_indicators(df)
        # keep base interval columns unchanged so Environment.get_price() finds 'close'
        suffix = interval.replace('minute', 'm').replace('hour', 'h')
        if interval == 'day':
            suffix = 'd1'
        df_ind = df_ind.reset_index().rename(columns={'index': 'date'})
        if interval != base_interval:
            # normalize column names to include timeframe suffix for non-base intervals
            df_ind = df_ind.rename(columns={c: f"{c}_{suffix}" for c in df_ind.columns if c not in ['date']})
        dfs[interval] = df_ind

    # align everything to base interval timestamps using left join + ffill
    if base_interval not in dfs:
        # pick smallest available interval
        base_interval = sorted(dfs.keys(), key=lambda x: len(x))[0]

    base_df = dfs[base_interval].copy()
    base_df['date'] = pd.to_datetime(base_df['date'])
    base_df = base_df.sort_values('date').reset_index(drop=True)

    # for each other timeframe, merge on nearest previous timestamp (ffill after merge)
    merged = base_df
    for interval, df_tf in dfs.items():
        if interval == base_interval:
            continue
        df_tf['date'] = pd.to_datetime(df_tf['date'])
        # merge_asof to align previous timeframe values to base timestamps
        merged = pd.merge_asof(merged.sort_values('date'), df_tf.sort_values('date'), on='date')

    # final cleanup
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
    # ensure date column exists
    if 'date' in merged.columns:
        merged['date'] = pd.to_datetime(merged['date'])
    return merged


def compute_multi_tf_features(merged_df):
    """Placeholder for extra feature engineering combining multi-tf columns.
    Currently this function passes through the merged dataframe, but can be extended
    to add cross-timeframe ratios, aggregated volume signals, etc.
    """
    df = merged_df.copy()
    # Example: compute volume ratio across timeframes if present
    # If minute and hourly volume columns exist, create normalized ratios
    vol_cols = [c for c in df.columns if c.startswith('volume')]
    # create simple normalized volume features
    for c in vol_cols:
        try:
            df[f'{c}_z'] = (df[c] - df[c].rolling(100).mean()) / (df[c].rolling(100).std() + 1e-9)
        except Exception:
            df[f'{c}_z'] = df[c]
    return df