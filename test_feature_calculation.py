#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature 계산 테스트"""

import sys
sys.path.append('.')

from quantylab.rltrader import data_manager
import pandas as pd

print("=" * 80)
print("🧪 Feature 계산 테스트")
print("=" * 80)

try:
    # 1. 데이터 로드
    print("\n1️⃣  데이터 로드...")
    df = data_manager.load_crypto_data('data/KRW-BTC_hourly.csv', '20230101', '20251101')
    print(f"   ✅ 로드 완료: {len(df)} rows")
    print(f"   컬럼: {list(df.columns)}")
    
    # 2. 전처리 (feature 계산 포함)
    print("\n2️⃣  전처리 및 Feature 계산...")
    df = data_manager.preprocess_crypto_data(df)
    print(f"   ✅ 전처리 완료: {len(df)} rows")
    print(f"   전체 컬럼 수: {len(df.columns)}")
    
    # 3. 학습용 컬럼 추출
    print("\n3️⃣  학습용 Feature 추출...")
    cols = [c for c in data_manager.COLUMNS_CRYPTO_DATA if c in df.columns and c != 'date']
    print(f"   ✅ Feature 수: {len(cols)}")
    print(f"   Features: {cols}")
    
    if len(cols) > 0:
        training_data = df[cols]
        print(f"\n   Training data shape: {training_data.shape}")
        print(f"\n   첫 5행:")
        print(training_data.head())
        
        # 결측치 확인
        null_counts = training_data.isnull().sum()
        if null_counts.sum() > 0:
            print(f"\n   ⚠️  결측치 발견:")
            print(null_counts[null_counts > 0])
        else:
            print(f"\n   ✅ 결측치 없음")
        
        # 무한대 확인
        inf_counts = training_data.isin([float('inf'), float('-inf')]).sum()
        if inf_counts.sum() > 0:
            print(f"\n   ⚠️  무한대 값 발견:")
            print(inf_counts[inf_counts > 0])
        else:
            print(f"\n   ✅ 무한대 값 없음")
    else:
        print("\n   ❌ 학습용 Feature를 찾을 수 없습니다!")
    
    print("\n4️⃣  학습 준비 상태:")
    if len(cols) == 23:
        print("   ✅ 23개 Feature 생성 완료 - 학습 가능!")
    else:
        print(f"   ⚠️  {len(cols)}개 Feature (23개 필요)")
    
except Exception as e:
    print(f"\n❌ 에러 발생: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
