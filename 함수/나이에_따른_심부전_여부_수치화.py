import pandas as pd

heart = pd.read_csv('heart.csv')

heart_age = heart.groupby(['Age'])['HeartDisease'].value_counts()#.unstack(level='HeartDisease')
print(heart_age.head())