#!/usr/bin/env pwsh
# PPO 모델 재학습 (올바른 feature 계산 포함)

Write-Host ("=" * 80)
Write-Host "🚀 PPO 모델 재학습 (23 features)"
Write-Host ("=" * 80)

Write-Host "`n📋 학습 설정:"
Write-Host "  - 알고리즘: PPO"
Write-Host "  - 데이터: 2023-01-01 ~ 2025-11-01 (2년)"
Write-Host "  - Epochs: 3000"
Write-Host "  - Learning Rate: 0.0003"
Write-Host "  - Discount Factor: 0.99"
Write-Host "  - Feature 계산: 23개 기술적 지표 포함"

Write-Host "`n⏱️  예상 시간: 9-10시간`n"

# 확인
$response = Read-Host "계속 진행하시겠습니까? (y/n)"
if ($response -ne 'y') {
    Write-Host "`n❌ 취소됨"
    exit 0
}

# 데이터 파일 확인
if (Test-Path ".\data\KRW-BTC_hourly.csv") {
    Write-Host "`n1️⃣  데이터 파일 확인 완료 (기존 파일 사용)"
} else {
    Write-Host "`n❌ 데이터 파일이 없습니다: data\KRW-BTC_hourly.csv"
    exit 1
}

# 학습 시작
Write-Host "`n2️⃣  모델 학습 시작... (Ctrl+C로 중단 가능)`n"

$timestamp = Get-Date -Format "yyyyMMddHHmmss"
$model_name = "${timestamp}_ppo_23features"

python main.py `
    --mode train `
    --rl_method ppo `
    --net dnn `
    --stock_code KRW-BTC `
    --start_date 20230101 `
    --end_date 20251101 `
    --lr 0.0003 `
    --discount_factor 0.99 `
    --num_epoches 3000 `
    --name $model_name

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ 학습 실패!"
    exit 1
}

Write-Host "`n3️⃣  학습 완료! 결과 확인 중...`n"

# 학습 결과 확인
$log_files = Get-ChildItem -Path "output\train_${model_name}*" -Filter "*.log" -Recurse | Sort-Object LastWriteTime -Descending
if ($log_files) {
    $log_file = $log_files[0].FullName
    Write-Host "📊 학습 로그: $log_file"
    $last_lines = Get-Content $log_file -Tail 5
    Write-Host "`n마지막 로그:"
    $last_lines | ForEach-Object { Write-Host "  $_" }
}

# 모델 파일 확인
$policy_file = Get-ChildItem "models\" -Filter "${model_name}_*_policy.mdl" | Select-Object -First 1
$value_file = Get-ChildItem "models\" -Filter "${model_name}_*_value.mdl" | Select-Object -First 1

if ($policy_file -and $value_file) {
    Write-Host "`n4️⃣  모델 파일 생성됨:"
    Write-Host "  ✅ $($policy_file.Name)"
    Write-Host "  ✅ $($value_file.Name)"
    
    # 백업 및 교체 여부 확인
    Write-Host "`n5️⃣  Best 모델로 교체하시겠습니까?"
    $replace = Read-Host "   (y/n)"
    
    if ($replace -eq 'y') {
        # 백업
        $backup_dir = "models\backup\$(Get-Date -Format 'yyyyMMddHHmmss')"
        New-Item -ItemType Directory -Path $backup_dir -Force | Out-Null
        Copy-Item "models\best\*.mdl" $backup_dir -ErrorAction SilentlyContinue
        Write-Host "  📦 기존 모델 백업: $backup_dir"
        
        # 교체
        Copy-Item $policy_file.FullName "models\best\$($policy_file.Name)" -Force
        Copy-Item $value_file.FullName "models\best\$($value_file.Name)" -Force
        Write-Host "  ✅ Best 모델 교체 완료"
        
        # socket_server.py 업데이트
        $model_basename = $policy_file.BaseName -replace '_ppo_dnn_policy$', ''
        Write-Host "`n  ⚠️  socket_server.py를 수동으로 업데이트하세요:"
        Write-Host "     model_name='$model_basename'"
    }
} else {
    Write-Host "`n❌ 모델 파일을 찾을 수 없습니다."
    exit 1
}

Write-Host "`n" + ("=" * 80)
Write-Host "✅ 완료!"
Write-Host ("=" * 80)
