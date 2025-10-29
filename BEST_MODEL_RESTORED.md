# Best 모델 학습 설정으로 복원 완료

## 현재 상태
프로젝트가 **가장 좋은 성능을 보인 모델 설정**으로 되돌려졌습니다.

## Best 모델 성능
- **모델명**: `20251027124731_a2c_dnn`
- **Max PV**: 137,989,972원 (**38% 수익**)
- **위치**: `models/best/20251027124731_a2c_dnn_policy.mdl` (policy), `*_value.mdl` (value)

## Best 모델의 핵심 설정

### 데이터
- **형식**: 단일 시간봉 (hourly) — Multi-TF 사용 안 함
- **파일**: `data/KRW-BTC_hourly.csv`
- **기간**: 2024-01-01 ~ 2025-10-26 (약 1년치)
- **Timesteps**: 1,741개

### 학습 파라미터
- **RL Method**: `a2c` (Actor-Critic)
- **Network**: `dnn` (Deep Neural Network)
- **Learning Rate**: `0.0005`
- **Discount Factor**: `0.95`
- **Epochs**: `2000`
- **Initial Balance**: `100,000,000`

### 학습 특징
- 매 epoch마다 활발한 매매 활동 (Buy: ~500, Sell: ~500 per epoch)
- 초기 탐험율(Epsilon) 1.0에서 점진적으로 0.0까지 감소
- 약 3.9시간의 학습 시간

## 빠른 시작 (Best 방식)

### 1. 데이터 수집 (필요시)
```powershell
.\scripts\collect_best_style_data.ps1
```

### 2. Best 방식으로 학습
```powershell
.\scripts\train_best_model_style.ps1
```

또는 직접:
```powershell
$env:PYTHONPATH = 'C:\Users\user\Desktop\RL\EC 해커톤\rltrader'
python main.py --mode train --rl_method a2c --net dnn --stock_code KRW-BTC --start_date 20240101 --end_date 20251026 --lr 0.0005 --discount_factor 0.95 --num_epoches 2000 --name a2c_retry
```

### 3. 결과 확인
```powershell
# 로그 파일에서 Max PV 확인
Select-String -Path "output\train_*\*.log" -Pattern "Max PV" | Select-Object -Last 5

# 137,989,972 이상이면 성공!
```

### 4. 백테스트
```powershell
$env:PYTHONPATH = 'C:\Users\user\Desktop\RL\EC 해커톤\rltrader'
python scripts\run_backtest.py --name <새_모델_이름>
```

## PPO vs A2C 성능 비교

| 모델 | RL Method | Max PV | 수익률 | Timesteps | 특징 |
|------|-----------|--------|--------|-----------|------|
| **20251027124731** | A2C + DNN | **137,989,972** | **+38%** | 1,741 | ✅ Best |
| KRW-BTC | A2C + DNN | 108,109,644 | +8% | 1,741 | 초기 학습 |
| ppo_long_20251028 | PPO + DNN | 100,184,482 | +0.2% | 86 | ❌ 학습 실패 |

### PPO 학습이 실패한 이유
1. **데이터 부족**: Multi-TF 데이터가 86개 timesteps만 생성됨 (A2C는 1,741개)
2. **HOLD 전략 수렴**: 거의 모든 epoch에서 매수/매도 없이 HOLD만 선택
3. **보상 신호 부족**: 짧은 데이터로 인해 충분한 학습이 이루어지지 않음

## 권장 사항

### ✅ 지금 바로 사용
- Best 모델(`20251027124731_a2c_dnn`)이 이미 `models/best/`에 있습니다
- 배포/추론 테스트: `python scripts\test_deployer_multi_tf.py`

### 🎯 추가 개선 시도
1. **Best 방식 재학습**: 더 긴 데이터 기간으로 재시도
   ```powershell
   # 2년치 데이터로 학습
   python main.py --mode train --rl_method a2c --net dnn --stock_code KRW-BTC --days 730 --lr 0.0005 --discount_factor 0.95 --num_epoches 3000
   ```

2. **하이퍼파라미터 튜닝**: LR/DF 조정하여 더 나은 결과 탐색

3. **앙상블 전략**: 여러 모델의 예측을 결합

## 파일 위치

### 새로 생성된 스크립트
- `scripts/train_best_model_style.ps1` - Best 방식 학습 스크립트
- `scripts/collect_best_style_data.ps1` - 단일 시간봉 데이터 수집

### Best 모델
- `models/best/20251027124731_a2c_dnn_policy.mdl`
- `models/best/20251027124731_a2c_dnn_value.mdl`

### 학습 결과
- `output/train_20251027124731_a2c_dnn/` - 로그 및 epoch 결과

---
**다음 단계**: `.\scripts\train_best_model_style.ps1` 실행으로 Best 방식 재현 또는 개선 시도
