from quantylab.rltrader.deployer import ModelDeployer
from quantylab.rltrader.data_manager import load_crypto_data, preprocess_crypto_data, COLUMNS_CRYPTO_DATA

# 예제: CSV로부터 과거 200봉을 읽어 초기화한 뒤 스트리밍처럼 한 봉씩 예측

def run_example():
    # 데이터 준비: 기존에 저장된 hourly CSV 사용
    df = load_crypto_data('data/KRW-BTC_hourly.csv')
    df = preprocess_crypto_data(df)
    # 종가만 추출 (가장 오래된 -> 최신 순서)
    closes = df['close'].tolist()
    if len(closes) < 201:
        raise RuntimeError('데이터 길이가 부족합니다. 최소 201개 이상의 봉 필요')

    # 초기 200봉으로 초기화
    init_closes = closes[:200]
    rest = closes[200:]

    deployer = ModelDeployer(model_name='20251027124731')
    deployer.load_initial_closes(init_closes)

    # 스트리밍 시뮬레이션: 나머지 봉을 한 번씩 넣어 예측
    for i, close in enumerate(rest):
        action, conf, unit, pv = deployer.on_new_close(close, execute=False)
        action_str = {0:'BUY',1:'SELL',2:'HOLD'}.get(action, 'UNK')
        print(f'tick {i}: close={close:,.0f} action={action_str} conf={conf:.3f} unit={unit:.6f} pv={pv:,.0f}')

if __name__ == '__main__':
    run_example()
