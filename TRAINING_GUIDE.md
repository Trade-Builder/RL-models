# PPO 모델 재학습 가이드 (23 Features)

## ✅ 준비 완료
- **Feature 계산**: 23개 기술적 지표 생성 확인됨
- **데이터**: 17,503 rows (2023-01-01 ~ 2025-11-01)
- **학습 준비**: 완료

---

## 🚀 학습 실행 방법

### 옵션 1: 간단한 명령어 (추천)
```powershell
$timestamp = Get-Date -Format "yyyyMMddHHmmss"
python main.py --mode train --rl_method ppo --net dnn --stock_code KRW-BTC --start_date 20230101 --end_date 20251101 --lr 0.0003 --discount_factor 0.99 --num_epoches 3000 --name "${timestamp}_ppo_23feat"
```

### 옵션 2: 스크립트 사용
```powershell
# 스크립트 실행 (확인 필요)
.\scripts\retrain_ppo_correct.ps1

# y 입력하여 시작
```

### 옵션 3: 백그라운드 실행
```powershell
$timestamp = Get-Date -Format "yyyyMMddHHmmss"
Start-Process powershell -ArgumentList "-NoProfile", "-Command", "cd 'C:\Users\user\Desktop\RL\EC 해커톤\RL-models'; python main.py --mode train --rl_method ppo --net dnn --stock_code KRW-BTC --start_date 20230101 --end_date 20251101 --lr 0.0003 --discount_factor 0.99 --num_epoches 3000 --name ${timestamp}_ppo_23feat" -WindowStyle Minimized
```

---

## ⏱️ 예상 소요 시간
- **약 9-10시간** (3000 epochs)
- 중간에 중단하려면: Ctrl+C

---

## 📊 학습 진행 확인

### 로그 확인
```powershell
# 최신 학습 로그 찾기
Get-ChildItem "output\train_*ppo_23feat*" -Recurse -Filter "*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 20
```

### 진행률 확인
```powershell
.\scripts\check_training.ps1
```

---

## 📦 학습 완료 후

### 1. 모델 파일 확인
```powershell
Get-ChildItem "models\" -Filter "*ppo_23feat*"
```

### 2. Best 모델로 교체
```powershell
# 백업
$backup_dir = "models\backup\$(Get-Date -Format 'yyyyMMddHHmmss')"
New-Item -ItemType Directory -Path $backup_dir -Force | Out-Null
Copy-Item "models\best\*.mdl" $backup_dir

# 새 모델 복사
$new_model = Get-ChildItem "models\" -Filter "*ppo_23feat_ppo_dnn_policy.mdl" | Select-Object -First 1
$model_name = $new_model.BaseName -replace '_ppo_dnn_policy$', ''
Copy-Item "models\${model_name}_ppo_dnn_policy.mdl" "models\best\" -Force
Copy-Item "models\${model_name}_ppo_dnn_value.mdl" "models\best\" -Force

Write-Host "✅ Best 모델 교체 완료: $model_name"
```

### 3. socket_server.py 업데이트
```python
# socket_server.py 8번째 줄
deployer = ModelDeployer(model_name='[새 모델 이름]', model_dir='models/best')
```

### 4. BUY 신호 테스트
```powershell
python find_buy_signals.py
```

---

## 🎯 성공 기준
- ✅ 학습 완료 (3000 epochs)
- ✅ BUY 신호 > 20%
- ✅ 포트폴리오 가치 증가

---

## ⚠️ 주의사항
1. 학습 중 컴퓨터를 끄지 마세요
2. GPU 사용 시 과열 주의
3. 로그 파일로 진행상황 확인 가능
4. 중단 시 Ctrl+C (데이터는 보존됨)

---

## 📝 문제 해결

### 학습이 시작되지 않으면
```powershell
# Python 환경 확인
python --version
pip list | Select-String "torch"

# 데이터 파일 확인
Test-Path ".\data\KRW-BTC_hourly.csv"
```

### 메모리 부족 시
- Epochs 줄이기: `--num_epoches 1000`
- 또는 데이터 기간 줄이기: `--start_date 20240101`

---

## 🎉 완료!
학습이 완료되면 새 PPO 모델이 BUY 신호를 제대로 생성할 거예요!
