"""
RL 모델 추론 스크립트 (Electron에서 직접 실행용)

사용법:
    python predict.py --market KRW-BTC --timeframe 1h --count 200

출력 형식 (JSON):
    {
        "action": 0,          # 0: HOLD, 1: BUY, 2: SELL
        "signal": "HOLD",     # "HOLD", "BUY", "SELL"
        "confidence": 0.85,
        "trade_unit": 0.1,
        "portfolio_value": 100000000
    }
"""

import sys
import os
import json
import argparse

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantylab.rltrader.deployer import ModelDeployer
from quantylab.rltrader.data_upbit import UpbitDataCollector


def main():
    parser = argparse.ArgumentParser(description='RL 모델 추론')
    parser.add_argument('--market', type=str, required=True, help='마켓 코드 (예: KRW-BTC)')
    parser.add_argument('--timeframe', type=str, default='1h', help='타임프레임 (1m, 5m, 15m, 1h, 4h, 1d)')
    parser.add_argument('--count', type=int, default=200, help='캔들 개수')
    parser.add_argument('--model_dir', type=str, default='models/best', help='모델 디렉토리')
    parser.add_argument('--model_name', type=str, default='20251027124731_a2c_dnn', help='모델 이름')

    args = parser.parse_args()

    try:
        # 1. 데이터 수집
        collector = UpbitDataCollector(ticker=args.market)

        # timeframe 매핑
        interval_map = {
            '1m': 'minute1',
            '5m': 'minute5',
            '15m': 'minute15',
            '1h': 'minute60',
            '4h': 'minute240',
            '1d': 'day'
        }

        interval = interval_map.get(args.timeframe, 'minute60')

        # 데이터 가져오기
        df = collector.get_ohlcv_data(interval=interval, count=args.count)

        if df is None or len(df) == 0:
            result = {
                "success": False,
                "error": f"No data available for {args.market}"
            }
            print(json.dumps(result))
            sys.exit(1)

        # 기술 지표 추가
        df = collector.add_technical_indicators(df)

        # 2. 모델 로드 및 추론
        model_dir = os.path.join(os.path.dirname(__file__), args.model_dir)

        deployer = ModelDeployer(
            model_dir=model_dir,
            model_name=args.model_name,
            balance=100000000
        )

        # 추론 실행
        prediction = deployer.predict(df)

        # 3. 결과 출력 (JSON)
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}

        result = {
            "success": True,
            "action": int(prediction['action']),
            "signal": action_map[int(prediction['action'])],
            "confidence": float(prediction['confidence']),
            "trade_unit": float(prediction['trade_unit']),
            "portfolio_value": float(prediction['portfolio_value']),
            "market": args.market,
            "timeframe": args.timeframe,
            "candles_used": len(df)
        }

        # JSON 출력 (Electron이 stdout에서 읽음)
        print(json.dumps(result))
        sys.exit(0)

    except Exception as e:
        # 에러 발생 시 JSON 형식으로 에러 출력
        result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(result))
        sys.exit(1)


if __name__ == '__main__':
    main()
