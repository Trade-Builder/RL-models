# RLTrader (crypto-focused) — Quickstart, 상태 및 배포 가이드

암호화폐 시계열 데이터를 사용해 강화학습(RL) 에이전트를 학습하고, 학습된 모델을 실시간(또는 시뮬레이션) 환경에서 배포하는 예제 프로젝트입니다.

> 이 README는 2025-10-28 기준으로 작성되었습니다. 최근 수정 내용과 현재 동작 상태, 테스트 방법을 자세히 포함합니다.

## 요약 — 핵심 기능
- Multi-timeframe 입력 지원: 여러 TF(예: 1m,5m,15m,1h,4h,1d)에서 N개의 최근 종가/거래량을 받아 동일한 전처리(지표)를 수행합니다.
- 학습: A2C/ PPO 등 강화학습 루틴이 포함되어 있으며, PyTorch 기반 네트워크 래퍼를 사용합니다.
- 배포: `quantylab/rltrader/deployer.py`의 `ModelDeployer`로 실시간/시뮬레이션 추론이 가능합니다.

## 최근 수정 및 알려진 이슈(해결/완료된 항목)
- `data_upbit.collect_large_dataset`의 interval alias 처리 문제(예: `hour1`)로 인한 UnboundLocalError 수정.
- 배포 시 `close`가 0으로 들어와 발생하던 ZeroDivisionError를 방지하도록 `ModelDeployer`와 `Agent`에 방어 코드 추가.
- 모델 입력(feature) 차원 불일치 문제를 완화하기 위해 배포자에서 모델의 BatchNorm 채널 정보를 시도해 읽고 패딩/잘라내기 보정 적용(임시책). 근본 해결은 아래 권장 작업에서 설명.

이 변경들로 `scripts/test_deployer_multi_tf.py` 같은 E2E 테스트가 예외 없이 작동하도록 개선되었습니다. (예시 출력: `Result: action, confidence, trade_unit, portfolio_value -> (0, 1.0, 1e-06, 100000000.0)`)

## 요구사항(간단)
- Python 3.8+ (3.10 권장)
- PyTorch (학습/추론), pandas, numpy, pyupbit(데이터 수집 시)

예: Windows PowerShell에서 빠른 설치 (가상환경 사용 권장):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements_common.txt
```

## 빠른 시작(핵심 커맨드)

1) 데이터 준비

 - 프로젝트의 `data/` 폴더에 심볼별 CSV(예: `data/KRW-BTC_hourly.csv`)을 둡니다. CSV에는 최소한 `date,open,high,low,close,volume` 컬럼이 있어야 합니다.

2) 수집 (pyupbit 사용 시)

```powershell
python main.py --mode collect --stock_code KRW-BTC --days 90 --tf_intervals minute1,minute5,minute15,hour1,hour4,day
```

3) 학습 (smoke / 장기 예시)

단기 smoke (테스트용):
```powershell
$env:PYTHONPATH = 'C:\Users\user\Desktop\RL\EC 해커톤\rltrader'; python scripts\run_ppo_smoke.py
```

장기 학습 예시 (PPO, multi-TF):
```powershell
$env:PYTHONPATH = 'C:\Users\user\Desktop\RL\EC 해커톤\rltrader'; python main.py --mode train --rl_method ppo --net dnn --stock_code KRW-BTC --multi_tf --days 90 --tf_intervals minute1,minute5,minute15,hour1,hour4,day --num_epoches 2000 --lr 0.0003 --name ppo_long_20251028
```

4) 테스트/백테스트
```powershell
$env:PYTHONPATH = 'C:\Users\user\Desktop\RL\EC 해커톤\rltrader'; python scripts\run_backtest.py --name <model_name>
```

5) 배포 / E2E 로컬 시뮬레이션

```powershell
$env:PYTHONPATH = 'C:\Users\user\Desktop\RL\EC 해커톤\rltrader'; python scripts\test_deployer_multi_tf.py
```

이 스크립트는 `data/KRW-BTC_multi_tf_sample.csv`를 사용해 multi-TF 입력을 구성하고 `ModelDeployer`를 통해 action/confidence/trade_unit을 출력합니다.

## 주요 파일 설명

- `main.py` : 학습/평가/수집을 위한 CLI 진입점
- `quantylab/rltrader/data_upbit.py` : pyupbit 기반 대량 수집기
- `quantylab/rltrader/data_manager.py` : multi-TF 데이터 로드 및 전처리
- `quantylab/rltrader/agent.py` : 에이전트 상태/행동/거래 단위 계산
- `quantylab/rltrader/environment.py` : 환경 래퍼 (관측/보상)
- `quantylab/rltrader/learners.py` : A2C/PPO 학습자 구현
- `quantylab/rltrader/networks/networks_pytorch.py` : PyTorch 네트워크 래퍼
- `quantylab/rltrader/deployer.py` : 배포용 ModelDeployer (multi-TF 전처리 포함)
- `scripts/` : 테스트 및 유틸 스크립트 (smoke, backtest, deployer 테스트 등)

## 최근 버그/오류 및 디버깅 가이드(상세)

1) UnboundLocalError during data collection

 - 증상: `data_upbit.collect_large_dataset` 호출 시 `total_candles`가 정의되지 않아 예외 발생.
 - 원인: interval alias(예: 'hour1') 처리 누락으로 total_candles 미설정.
 - 해결: interval 정규화 로직 추가 및 기본값 보완으로 문제 해결.

2) ZeroDivisionError in deployer/agent

 - 증상: deploy 시 `Agent.decide_trading_unit`에서 가격으로 나눌 때 0으로 나누는 에러 발생.
 - 원인: multi-TF 병합 결과 `close` 컬럼이 없거나 0으로 채워져 있었음.
 - 해결: `ModelDeployer`에 초기화/업데이트 단계에서 유효한 `close` 값을 보장하는 폴백 로직 추가. 또한 `Agent.get_states`와 `decide_trading_unit`에서 가격/포트폴리오 값이 0일 때 안전하게 처리하도록 수정.

3) 모델 입력 차원 불일치(BatchNorm 오류)

 - 증상: 배포 시 모델에서 `running_mean should contain X elements not Y` 또는 `Expected more than 1 value per channel when training` 같은 BatchNorm 관련 runtime 오류.
 - 원인: 학습 시 사용한 feature 컬럼 수/순서와 배포 시 동적으로 생성한 feature 벡터가 불일치.
 - 임시 완화책: 배포자에서 모델을 로드할 때 BatchNorm의 expected 채널 수를 시도 추론해 입력 벡터를 패딩/자르는 로직을 추가했습니다. 이는 임시방편이며 근본 해결은 아래 권장 작업에서 설명합니다.

## 권장 개선(우선순위)

1) 모델 메타데이터 저장 및 배포에서 재사용 (권장 최우선)

 - 학습 스크립트가 모델을 저장할 때, 모델 파라미터(.mdl) 옆에 `{model_name}.meta.json` 형태로 아래 정보를 함께 저장하도록 하세요:
   - `feature_columns` (정확한 순서)
   - `input_dim`
   - `tf_intervals` (사용된 TF 목록)
 - 배포자는 메타파일을 읽어 `chart_data`를 정확히 같은 순서로 재정렬 -> 모델 입력 오류 근본 해결.

2) 배포자 리팩터

 - 현재의 패딩/자르기 보정은 임시방편입니다. 모델 메타데이터가 있으면 해당 경로를 우선 사용하세요.
 - FastAPI 기반 추론 서버 래퍼를 만들어 모델 로드 및 추론 엔드포인트를 노출하면 운영에 편리합니다.

## 모델 메타데이터 예시

```json
{
  "model_name": "20251027124731",
  "feature_columns": ["close","diffratio","close_ma5_ratio",...,"close_m5","diffratio_m5",...],
  "input_dim": 84,
  "tf_intervals": ["minute1","minute5","minute15","hour1","hour4","day"]
}
```

## 문제 재현/디버깅 체크리스트

1) 모델 로딩 실패 / BatchNorm 오류 발생 시
 - `models/best/`에 모델 파일(.mdl) 존재 여부 확인
 - 모델과 함께 저장된 `.meta.json` 존재 여부 확인(추후 추가 권장)
 - `scripts/test_deployer_multi_tf.py`로 E2E 재현

2) deployer가 0 또는 비정상적 trade_unit을 반환할 때
 - 입력 CSV의 multi-TF close/volume 열이 올바른지 확인
 - `ModelDeployer`가 초기화 시 사용한 `close` 컬럼을 로그(또는 print)로 확인

## 다음에 제가 도와드릴 수 있는 작업

- 학습 코드에 모델 메타데이터 자동 저장 추가(바로 구현 가능)
- `deployer`에서 메타데이터를 읽어 정확한 reindex를 적용하도록 수정(바로 구현 가능)
- FastAPI 추론 서버 및 Dockerfile 템플릿 작성

원하시면 제가 우선순위에 따라 위 작업 중 하나를 바로 적용해 드리겠습니다. 예: "훈련 시 feature 메타데이터 자동 추가"를 먼저 구현할까요?

---
감사합니다. 이 README를 더 줄이거나, 한글/영문 버전 분리, 또는 상세 API 문서화(예: Sphinx)를 추가하길 원하면 알려주세요.
