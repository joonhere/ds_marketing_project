import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import re 

# --- 0. 상수 정의 (코드의 유연성 및 유지보수성 확보) ---
COUNTRY_COLUMN = 'country'
GENRE_COLUMN = 'listed_in'
TITLE_COLUMN = 'title'
DESCRIPTION_COLUMN = 'description'
STOP_WORDS_CUSTOM = {'series', 'film', 'movie', 'show', 'story', 'life', 'new', 'world', 'us', 'korean', 'korea', 'drama', 'kdrama'} # 사용자 정의 불용어

# --- 1. 데이터 로드 및 필터링 함수 ---
def load_and_filter_data(
    file_path: str, 
    filter_keyword: str, 
    cols_to_check: list[str]
) -> pd.DataFrame | None:     # 🚨 Optional 대신 '타입 | None' 사용
    """
    CSV 파일을 로드하고 지정된 컬럼들에서 키워드를 포함하는 행을 필터링합니다.
    """
    try:
        df = pd.read_csv(file_path)
        
        # 필터링 조건 조합: 여러 컬럼에 대해 OR 조건을 적용
        filter_condition = False
        for col in cols_to_check:
            # 💡 Null 값 처리 및 키워드 포함 여부 확인
            if col in df.columns:
                filter_condition = filter_condition | df[col].fillna('').str.contains(filter_keyword, case=False, na=False)
            else:
               print(f"🚨 경고: 컬럼 '{col}'을 찾을 수 없습니다. 이 컬럼은 필터링에서 제외됩니다.")
        
        filtered_df = df[filter_condition].copy()
        
        if filtered_df.empty:    # 만약 filtered_df 데이터프레임이 비어있다면: True 임
            print(f"🚨 경고: '{filter_keyword}' 관련 콘텐츠를 찾을 수 없습니다.")
            return None
            
        return filtered_df    # 최종 결과 반환
    
    except FileNotFoundError:
        print(f"🚨 오류: {file_path} 파일을 찾을 수 없습니다.")
        return None

#--- 실행 예시 (테스트용) ---
# 주의: 이 코드를 실행하려면 실제 'netflix_titles.csv' 파일 경로가 필요합니다.
file_path = 'netflix_titles.csv'
keyword = 'korea'
cols_to_check = [COUNTRY_COLUMN, GENRE_COLUMN]

filtered_data= load_and_filter_data(
    file_path, 
    keyword, 
    cols_to_check 
)   
print(filtered_data)