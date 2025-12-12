from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

def extract_word_frequency(text: str) -> pd.DataFrame:
    """
    단일 텍스트에서 단어 빈도를 추출하여 DataFrame으로 반환합니다.
    """
    # 1. CountVectorizer 초기화 및 학습
    # stop_words에 list()가 필요하므로, 여기서는 None을 사용합니다.
    vectorizer = CountVectorizer() 
    
    # 2. 텍스트 변환: 희소 행렬 생성
    word_matrix = vectorizer.fit_transform([text])
    
    # 3. 데이터 변환: 희소 행렬을 1차원 배열로 변환 (핵심: toarray().flatten())
    # word_freq는 모든 단어의 빈도수를 순서대로 담은 1차원 배열입니다.
    word_freq = word_matrix.toarray().flatten()
    
    # 4. 결과 DataFrame 생성
    word_df = pd.DataFrame({
        'word': vectorizer.get_feature_names_out(),
        'freq': word_freq
    }).sort_values(by='freq', ascending=False)
    
    return word_df

# 실전 사용 예시
sample_text = "Analysis is fun. Python is great for analysis. Python is easy."
result_df = extract_word_frequency(sample_text)

print(result_df)