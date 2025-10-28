import pyupbit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class UpbitDataCollector:
    """Upbit 비트코인 데이터 수집 클래스"""
    
    def __init__(self, ticker="KRW-BTC"):
        self.ticker = ticker
        
    def get_ohlcv_data(self, interval='minute60', count=200, to=None):
        """
        OHLCV 데이터 수집
        
        Args:
            interval: 'minute1', 'minute3', 'minute5', 'minute10', 'minute15',
                     'minute30', 'minute60', 'minute240', 'day', 'week', 'month'
            count: 조회할 데이터 개수
            to: 마지막 캔들 시각 (YYYYMMDD or YYYYMMDDHHmmss)
        """
        df = pyupbit.get_ohlcv(self.ticker, interval=interval, count=count, to=to)
        return df
    
    def collect_large_dataset(self, interval='minute60', days=365):
        """
        대량의 과거 데이터 수집
        
        Args:
            interval: 캔들 간격
            days: 수집할 일수
        """
        all_data = []
        to_date = datetime.now()
        
        # interval alias 처리 및 한번에 가져올 캔들 수
        # 사용자 입력으로 'hour1', 'hour4', 'minute1', 'minute60', 'day' 등이 들어올 수 있으므로
        # pyupbit에서 사용하는 interval 문자열로 정규화합니다.
        norm_interval = interval
        if interval.startswith('hour'):
            try:
                hrs = int(interval.replace('hour', ''))
                norm_interval = f'minute{hrs * 60}'
            except Exception:
                norm_interval = 'minute60'

        count_per_request = 200

        # 필요한 총 요청 횟수 계산
        total_candles = None
        if 'minute' in norm_interval:
            minutes = int(norm_interval.replace('minute', ''))
            total_candles = (days * 24 * 60) // minutes
        elif norm_interval == 'day':
            total_candles = days

        if total_candles is None:
            # 알 수 없는 interval일 경우 기본값으로 하루 단위로 계산
            total_candles = days

        requests_needed = (total_candles // count_per_request) + 1
        
        print(f"총 {requests_needed}번의 요청이 필요합니다...")
        
        for i in range(requests_needed):
            try:
                df = pyupbit.get_ohlcv(
                    self.ticker, 
                    interval=norm_interval, 
                    count=count_per_request,
                    to=to_date.strftime("%Y%m%d%H%M%S")
                )
                
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    to_date = df.index[0] - timedelta(seconds=1)
                    print(f"진행: {i+1}/{requests_needed}, 마지막 날짜: {to_date}")
                    
                time.sleep(0.1)  # API 요청 제한 고려
                
            except Exception as e:
                print(f"데이터 수집 오류: {e}")
                break
        
        # 모든 데이터 결합
        if all_data:
            result_df = pd.concat(all_data)
            result_df = result_df.sort_index()
            result_df = result_df[~result_df.index.duplicated(keep='first')]
            return result_df
        
        return None
    
    def add_technical_indicators(self, df):
        """
        기술적 지표 추가 (주식 데이터와 유사한 형태로)
        """
        df = df.copy()
        
        # 컬럼명 변경 (소문자로)
        df.columns = ['open', 'high', 'low', 'close', 'volume', 'value']
        
        # 비율 계산
        df['open_lastclose_ratio'] = df['open'] / df['close'].shift(1) - 1
        df['high_close_ratio'] = df['high'] / df['close'] - 1
        df['low_close_ratio'] = df['low'] / df['close'] - 1
        df['diffratio'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        df['volume_lastvolume_ratio'] = df['volume'] / df['volume'].shift(1) - 1
        
        # 이동평균 비율
        for period in [5, 10, 20, 60, 120]:
            df[f'close_ma{period}_ratio'] = df['close'] / df['close'].rolling(period).mean() - 1
            df[f'volume_ma{period}_ratio'] = df['volume'] / df['volume'].rolling(period).mean() - 1
        
        # Bollinger Bands
        for period in [20]:
            df[f'bb_upper_{period}'] = df['close'].rolling(period).mean() + 2 * df['close'].rolling(period).std()
            df[f'bb_lower_{period}'] = df['close'].rolling(period).mean() - 2 * df['close'].rolling(period).std()
            df[f'bb_ratio_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # RSI 계산
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # NaN 제거
        df = df.dropna()
        
        return df
    
    def save_to_csv(self, df, filename):
        """데이터를 CSV로 저장"""
        df.to_csv(filename)
        print(f"데이터가 {filename}에 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    collector = UpbitDataCollector(ticker="KRW-BTC")
    
    # 1년치 1시간봉 데이터 수집
    df = collector.collect_large_dataset(interval='minute60', days=365)
    
    if df is not None:
        # 기술적 지표 추가
        df = collector.add_technical_indicators(df)
        
        # CSV로 저장
        collector.save_to_csv(df, 'data/KRW-BTC_hourly.csv')
        
        print(f"데이터 크기: {df.shape}")
        print(f"기간: {df.index[0]} ~ {df.index[-1]}")
        print(df.head())
