import pandas as pd
import numpy as np

file_path = 'marketing_campaign_data.csv'

try:
    df = pd.read_csv(file_path, encoding='utf-8')
except FileNotFoundError:
    print(f"만약 {file_path}을 찾지 못한다면 생성된 data로 진행합니다.")
    
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
    print(df)

    columns_change ={
        'CustomerID':'customer_id',
        'Name':'name',
        'Age':'age',
        'TotalSpend':'total_spend',
        'EnrollmentDate':'enrollment_date',
        'Churn':'churn'
    }
    df.rename(columns=columns_change, inplace=True)
    print(df.columns)
    print(df)

    print("="*50)
    df['enrollment_date'].dtype
    print(f"테이터 타입: {df['enrollment_date'].dtype}") 
    df['enrollment_date'] = pd.to_datetime(df['enrollment_date'])
    print(f"pd.datetime()사용후 데이터 타입 변화: {df['enrollment_date'].dtype}") 

    print("="*50)
    print("churn 컬럼 데이터 타입 변환 전: {df['churn'].dtype}")
    df['churn'] = df['churn'].astype('int')
    print(f".astype('int') 사용 후 데이터 타입 변화: {df['churn'].dtype}")
    print('='*50)