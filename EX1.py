import pandas as pd
import numpy as np # 가상 데이터 생성을 위해 NumPy 사용

# 가상의 고객 데이터 파일명을 사용합니다.
file_path = 'marketing_campaign_data.csv'

# 파일을 불러올 때 인코딩을 'utf-8'로 지정합니다. (오타 수정: uft-8 -> utf-8)
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except FileNotFoundError:
    # 파일이 실제로 존재하지 않을 경우를 대비하여 mock DataFrame 생성
    print(f"경고: 실제 파일 '{file_path}'이(가) 없어 가상 데이터를 생성합니다.")
    
# 마케팅 데이터 과학자를 위한 가상 데이터 생성
np.random.seed(42)
data = {
    'CustomerID': range(1001, 1006),
    'Name': ['홍길동', '김철수', '이영희', '박지민', '최현우'],
    'Age': np.random.randint(20, 60, 5),
    'TotalSpend': np.round(np.random.uniform(50.0, 500.0, 5), 2),
    'EnrollmentDate': ['2023-01-15', '2023-03-20', '2022-11-01', '2024-05-10', '2021-09-28'],
    'Churn': np.random.choice([0, 1], 5, p=[0.7, 0.3]) # 이탈 여부 (0: 유지, 1: 이탈)
}
df = pd.DataFrame(data)

# 미션 목표: 데이터의 상위 5개 행을 확인합니다.
# df.head()는 데이터의 크기나 구조를 빠르게 파악할 때 유용합니다.
print("--- [M1.2 미션 1 결과: 고객 데이터 상위 5행] ---")
print(df.head())