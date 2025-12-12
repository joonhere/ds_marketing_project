import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 💡 오류 1: seaborn 모듈 임포트 추가
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import re # 💡 오류 2: 데이터 필터링 정확도를 위한 re 모듈 임포트 추가

# 1. 데이터 로드
file_path = 'netflix_preprocessed.csv'
# 🚨 주의: 파일 경로와 파일 이름이 정확한지 확인하세요.
try:
    netflix = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"🚨 오류: {file_path} 파일을 찾을 수 없습니다. 파일 경로를 확인하세요.")
    exit()

# 2. 필터링 : 'korea'와 관련된 항목 필터링
korea_data = netflix[
    # na=False는 NaN 값을 False로 처리하여 필터링 오류 방지 (이미 na=False가 있더라도 안전한 방식)
    (netflix['description'].fillna('').str.contains('Korea', case=False, na=False)) |
    (netflix['title'].fillna('').str.contains('Korea', case=False, na=False)) |
    (netflix['listed_in'].fillna('').str.contains('Korean', case=False, na=False))
].copy() # 💡 메모리 복사 및 경고 방지

if korea_data.empty:
    print("🚨 경고: 'Korea' 관련 콘텐츠를 찾을 수 없어 분석을 중단합니다.")
    exit()

# 3. 텍스트 데이터 결합 및 전처리 (정규표현식으로 특수 문자 제거 추가)
text = ' '.join(korea_data['description'].dropna().tolist())
# raw_text = ' '.join(korea_data['description'].dropna().tolist())
# 💡 한글/영문/공백만 남기고 모두 제거하여 분석 정확도 향상
#text = re.sub(r'[^가-힣a-zA-Z\s]', '', raw_text) 

# 4. 피처 엔지니어링: CountVectorizer를 사용하여 단어 빈도 추출
# 💡 오류 3: stop_words='englist' -> 'english'로 수정
vectorizer = CountVectorizer(stop_words='english') # , token_pattern=r'(?u)\b\w\w+\b') # 두 글자 이상 단어만 추출
word_matrix = vectorizer.fit_transform([text])
# 💡 오류 4: wors_freq -> word_freq (변수명 오타 수정)
word_freq = word_matrix.toarray().flatten() 

# 단어와 빈도수를 데이터프레임으로 생성
word_df = pd.DataFrame({
    'word': vectorizer.get_feature_names_out(), 
    'freq': word_freq
})

# 💡 오류 5: sort_valuse -> sort_values로 수정
# 💡 오류 6: by='frequency' -> by='freq'로 컬럼명 일치 수정
# 💡 오류 7: top_wird_df -> top_words_df (변수명 오타 수정)
word_df_sorted = word_df.sort_values(by='freq', ascending=False)
top_words_df = word_df_sorted.head(10)
print(top_words_df)

# 5. 시각화: 상위 단어를 seaborn을 사용하여 시각화
plt.figure(figsize=(10,6))
# 💡 오류 8: platte='viridis' -> palette='viridis'로 오타 수정
sns.barplot(data=top_words_df, x='freq', y='word', palette='viridis')
plt.title('Top 10 Korean Words in Descriptions Related to Korea')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()

# 6. 시각화: 워드클라우드 생성
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
    dict(zip(word_df_sorted['word'], word_df_sorted['freq'])) # 💡 수정: 정렬된 word_df_sorted 사용
)

# 워드클라우드 시각화
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()