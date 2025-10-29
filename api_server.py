"""
FastAPI 서버 - RL 모델 추론 API

Electron 앱에서 호출하여 매매 신호를 받을 수 있습니다.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os
import sys
import logging

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantylab.rltrader.deployer import ModelDeployer
from quantylab.rltrader.data_upbit import UpbitDataCollector

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="RLTrader API",
    description="강화학습 기반 암호화폐 트레이딩 모델 추론 API",
    version="1.0.0"
)

# CORS 설정 (Electron에서 호출 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 origin만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 글로벌 모델 deployer (서버 시작 시 한 번만 로드)
deployer: Optional[ModelDeployer] = None


class PredictionRequest(BaseModel):
    """추론 요청 모델"""
    market: str  # 예: "KRW-BTC"
    timeframe: str = "1h"  # 예: "1m", "5m", "15m", "1h", "4h", "1d"
    count: int = 200  # 필요한 캔들 개수


class PredictionResponse(BaseModel):
    """추론 응답 모델"""
    action: int  # 0: HOLD, 1: BUY, 2: SELL
    confidence: float  # 신뢰도
    trade_unit: float  # 거래 비율
    portfolio_value: float  # 예상 포트폴리오 가치
    signal: str  # "HOLD", "BUY", "SELL"


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    model_loaded: bool
    model_name: str


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    global deployer

    try:
        logger.info("🚀 Loading RL model...")

        # Best 모델 경로
        model_dir = os.path.join(os.path.dirname(__file__), "models", "best")
        model_name = "20251027124731_a2c_dnn"

        # ModelDeployer 초기화
        deployer = ModelDeployer(
            model_dir=model_dir,
            model_name=model_name,
            balance=100000000  # 1억원 시작 자본
        )

        logger.info(f"✅ Model loaded successfully: {model_name}")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        deployer = None


@app.get("/", response_model=HealthResponse)
async def root():
    """루트 엔드포인트 - 서버 상태 확인"""
    return {
        "status": "running",
        "model_loaded": deployer is not None,
        "model_name": "20251027124731_a2c_dnn" if deployer else "none"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy" if deployer else "model_not_loaded",
        "model_loaded": deployer is not None,
        "model_name": "20251027124731_a2c_dnn" if deployer else "none"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    RL 모델 추론 엔드포인트

    Args:
        request: 추론 요청 (market, timeframe, count)

    Returns:
        PredictionResponse: 매매 신호 및 신뢰도

    Example:
        POST /predict
        {
            "market": "KRW-BTC",
            "timeframe": "1h",
            "count": 200
        }
    """
    if deployer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please restart the server."
        )

    try:
        logger.info(f"📊 Prediction request: {request.market} {request.timeframe}")

        # 1. Upbit에서 데이터 수집
        collector = UpbitDataCollector(ticker=request.market)

        # timeframe 매핑 (API 요청용)
        interval_map = {
            '1m': 'minute1',
            '5m': 'minute5',
            '15m': 'minute15',
            '1h': 'minute60',
            '4h': 'minute240',
            '1d': 'day'
        }

        interval = interval_map.get(request.timeframe, 'minute60')

        # 데이터 가져오기
        df = collector.get_ohlcv_data(interval=interval, count=request.count)

        if df is None or len(df) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {request.market}"
            )

        # 기술 지표 추가
        df = collector.add_technical_indicators(df)

        logger.info(f"✅ Data collected: {len(df)} candles")

        # 2. 모델 추론
        result = deployer.predict(df)

        # 3. 응답 생성
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}

        response = PredictionResponse(
            action=result['action'],
            confidence=result['confidence'],
            trade_unit=result['trade_unit'],
            portfolio_value=result['portfolio_value'],
            signal=action_map[result['action']]
        )

        logger.info(f"🎯 Prediction: {response.signal} (confidence: {response.confidence:.2%})")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/markets")
async def get_markets():
    """지원되는 마켓 목록 반환"""
    return {
        "markets": [
            "KRW-BTC",
            "KRW-ETH",
            "KRW-XRP",
            "KRW-ADA",
            "KRW-DOGE"
        ]
    }


if __name__ == "__main__":
    # 서버 실행
    # 기본 포트: 8000
    # Electron에서 http://localhost:8000 으로 호출

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 개발 모드
        log_level="info"
    )
