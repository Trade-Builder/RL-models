import pandas as pd
import numpy as np

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