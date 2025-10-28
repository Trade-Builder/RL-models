# Best 모델 방식: 단일 시간봉 데이터 수집
# Multi-TF 사용하지 않고 단순 hourly 데이터만 사용

$env:PYTHONPATH = 'C:\Users\user\Desktop\RL\EC 해커톤\rltrader'

Write-Host "단일 시간봉(hourly) 데이터 수집 중..." -ForegroundColor Cyan

python -c @"
from quantylab.rltrader.data_upbit import UpbitDataCollector
import os

collector = UpbitDataCollector(ticker='KRW-BTC')
print('365일치 1시간봉 데이터 수집 중...')
df = collector.collect_large_dataset(interval='minute60', days=365)

if df is not None:
    df = collector.add_technical_indicators(df)
    os.makedirs('data', exist_ok=True)
    collector.save_to_csv(df, 'data/KRW-BTC_hourly.csv')
    print(f'데이터 크기: {df.shape}')
    print(f'기간: {df.index[0]} ~ {df.index[-1]}')
else:
    print('데이터 수집 실패')
"@

Write-Host "`n데이터 수집 완료! data/KRW-BTC_hourly.csv 생성됨" -ForegroundColor Green
