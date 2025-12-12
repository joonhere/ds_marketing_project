import pandas as pd
import numpy as np

# 💡 특정 값의 비율을 확인하는 재사용 함수 (Transfer Learning)

def check_placeholder_rate(df: pd.DataFrame, column: str, value: int | float) -> float:
    """
    특정 컬럼에서 지정된 플레이스 홀더 값의 비율을 계산하여 반환합니다.
    """
    # 벡터화: 조건(True/False 마스크)을 생성하고, .mean()으로 비율 계산
    rate = (df[column] == value).mean() * 100
    return rate

# ----------------------------------------------------------------------
if __name__ == "__main__":

    df = pd.read_csv('heart.csv')
    
    # 💡 가상의 데이터 (심부전 데이터셋에 -1 플레이스 홀더가 있다고 가정)
    # data = {'Age': [50, 60, -1, 70, 55], 'Cholesterol': [200, 180, 220, -1, 190]}
    # df = pd.DataFrame(data) 

    print("\n--- 🚨 특정 플레이스 홀더 비율 확인 (Transfer Learning) 🚨 ---")
    
    # 1. 'Age' 컬럼에서 -1의 비율 확인
    age_rate = check_placeholder_rate(df, 'Age', -1)
    print(f"Age 컬럼의 '-1' (이상치/플레이스 홀더) 비율: {age_rate:.2f}%")
    # 결과: 20.00% (총 5개 중 1개)

    # 2. 'Cholesterol' 컬럼에서 -1의 비율 확인
    chol_rate = check_placeholder_rate(df, 'Cholesterol', -1)
    print(f"Cholesterol 컬럼의 '-1' 비율: {chol_rate:.2f}%")
    # 결과: 20.00% (총 5개 중 1개)