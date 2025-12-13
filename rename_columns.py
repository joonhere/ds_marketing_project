import pandas as pd
import numpy as np

file_path = 'marketing_campaign_data.csv'
try:
	df = pd.read_csv(file_path, encoding='utf-8')
except FileNotFoundError:
	print(f"만약 {file_path}이 없다면 가상 data로 변경한다.")
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

name_change = {
	'CustomerID':'customer_id',
	'TotalSpend':'total_spend'
}
df.rename(columns=name_change, inplace=True)
#print(df.columns)
print(df)
