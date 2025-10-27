# RLTrader (crypto-focused) — quickstart & deployment

이 저장소는 암호화폐(또는 시계열 자산) 가격 데이터를 이용해 강화학습 에이전트를 학습하고 배포하는 예제 프로젝트입니다.

최근 변경 사항 요약:
- 데이터 파이프라인: `quantylab/rltrader/data_manager.py` 의 `load_crypto_data`/`preprocess_crypto_data`를 사용해 CSV를 읽고 전처리합니다.
- 에이전트: `quantylab/rltrader/agent.py`는 소수점 매매를 지원하고, 단계별(틱별) 포트폴리오 가치 변화량을 보상으로 반환합니다.
- 네트워크: PyTorch 기반 wrapper(`quantylab/rltrader/networks/networks_pytorch.py`)를 사용합니다.
- 배포: `quantylab/rltrader/deployer.py`와 `deploy/run_deploy_example.py`를 추가하여, 실시간 들어오는 봉(종가)을 자동 전처리하고 모델에 입력해 행동을 생성합니다.

최소 요구사항
- Python 3.8+ 권장
- PyTorch (GPU 권장은 선택사항)
- pandas, numpy, tqdm

빠른 시작
1. 데이터 준비
   - `data/{SYMBOL}_hourly.csv` 형식의 CSV (date,open,high,low,close,volume 등)를 `data/`에 둡니다.

2. 학습 예시
   - 로컬 학습 예시:
     ```bash
     python main.py --mode train --rl_method a2c --net dnn --stock_code KRW-BTC --start_date 20240101 --end_date 20251026 --lr 0.0005 --discount_factor 0.95 --num_epoches 2000
     ```
   - 학습 결과(모델)는 `models/{name}_a2c_dnn_policy.mdl` 및 `models/{name}_a2c_dnn_value.mdl`로 저장됩니다.

3. 테스트/평가
   - 학습이 끝난 모델로 평가:
     ```bash
     python main.py --mode test --name 20251027124731 --stock_code KRW-BTC --rl_method a2c --net dnn --start_date 20240101 --end_date 20251026
     ```

배포(예: 실시간 예측)
- `quantylab/rltrader/deployer.py`의 `ModelDeployer`를 사용하면, 초기 200봉으로 초기화한 뒤 `on_new_close(close)`로 틱 단위 예측을 받을 수 있습니다.
- 예제: `python deploy/run_deploy_example.py` (CSV에서 초깃값 읽어 시뮬레이션 실행)

모델 관리를 위한 권장 워크플로
- 훈련/검증을 여러 번 돌리면 모델 파일이 많이 쌓입니다. 기본적으로 `models/` 내부의 모든 `.mdl`은 Git에서 무시하도록 설정되어 있습니다.
- 단, 최종적으로 배포하거나 버전으로 보존할 '최고 성능 모델'은 `models/best/` 폴더에 복사해서 추적하도록 권장합니다. 레포지토리에는 helper 스크립트 `scripts/select_and_keep_best_model.py`를 추가하여 `output/`의 로그를 스캔해 최고 `Max PV`를 보인 학습의 모델을 `models/best/`로 복사할 수 있습니다.

파일/폴더 설명
- `main.py` : 학습/테스트/예측 진입점 (CLI)
- `quantylab/rltrader/` : 라이브러리 코드 (agent, learners, networks, data_manager, deployer 등)
- `data/` : (로컬) 원시 CSV 데이터 (커밋에 포함하지 마세요)
- `output/` : 학습/테스트 로그와 가시화 결과 (커밋에 포함하지 마세요)
- `models/best/` : Git에 올릴 최종 선택된 모델을 보관하는 폴더

다음 작업(옵션)
- REST API로 모델을 서비스하는 FastAPI 래퍼 추가
- 실거래 주문 라우터/슬리피지 처리를 추가한 배포 예제

문의 및 기여
- 사용 중 문제가 있거나 개선하고 싶은 점이 있으면 이슈를 열어 주세요.
