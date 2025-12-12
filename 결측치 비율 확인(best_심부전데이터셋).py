import pandas as pd
import numpy as np

# ----------------------------------------------------------------------

# 💡 Python 3.10+ 문법 적용: Optional 대신 pd.Series | None 사용
def check_missing_data_vectorized(df: pd.DataFrame) -> pd.Series | None:
    """
    DataFrame의 모든 컬럼에 대해 결측치 비율을 벡터화하여 계산하고, 
    결측치가 있는 컬럼만 내림차순 Series로 반환합니다.
    """
    # 1. isna()와 mean()을 사용하여 벡터화된 비율 계산
    missing_rates = df.isna().mean() * 100
    
    # 2. 논리형 인덱싱을 사용하여 비율이 0%를 초과하는 컬럼만 선택
    #    (missing_rates > 0)이 True/False 마스크를 생성합니다.
    missing_rates_present = missing_rates[missing_rates > 0].sort_values(ascending=False)
    
    if missing_rates_present.empty:
        return None
    return missing_rates_present.round(2)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # 💡 데이터 로드: heart.csv 파일명을 직접 사용
    try:
        heart = pd.read_csv('heart.csv') 
    except FileNotFoundError:
        print("🚨 heart.csv 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()
        
    missing_info = check_missing_data_vectorized(heart)
    
    if missing_info is not None:
        print("\n--- 🚨 심부전 데이터셋 결측치(Null) 비율 분석 결과 🚨 ---")
        # to_string()으로 Series 형태를 깔끔하게 출력합니다.
        print(missing_info.to_string()) 
        print(f"\n💡 총 {len(missing_info)}개의 컬럼에서 결측치가 발견되었습니다.")
        
    else:
        print("✅ 데이터셋에 결측치가 발견되지 않았습니다. 분석을 진행하세요.")