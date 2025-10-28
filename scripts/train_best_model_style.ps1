# Best 모델(20251027124731_a2c_dnn) 방식으로 학습하는 스크립트
# Max PV: 137,989,972 (38% 수익) 달성한 설정

$env:PYTHONPATH = 'C:\Users\user\Desktop\RL\EC 해커톤\rltrader'

# Best 모델과 동일한 설정으로 학습
python main.py `
    --mode train `
    --rl_method a2c `
    --net dnn `
    --stock_code KRW-BTC `
    --start_date 20240101 `
    --end_date 20251026 `
    --lr 0.0005 `
    --discount_factor 0.95 `
    --num_epoches 2000 `
    --balance 100000000 `
    --name "a2c_best_style_$(Get-Date -Format 'yyyyMMddHHmmss')"

Write-Host "`n학습 완료! output/ 폴더에서 결과를 확인하세요." -ForegroundColor Green
Write-Host "Max PV가 137,989,972 이상이면 models/best/에 복사하세요." -ForegroundColor Yellow
