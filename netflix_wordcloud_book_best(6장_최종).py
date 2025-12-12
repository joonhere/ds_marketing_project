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
CUSTOM_STOP_WORDS = {'series', 'film', 'movie', 'show', 'story', 'life', 'new', 'world', 'us', 'korean', 'korea', 'drama', 'kdrama'} # 사용자 정의 불용어

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

# --- 2. 텍스트 전처리 및 단어 빈도 분석 함수 ---
def analyze_word_frequency(
		df: pd.DataFrame, 
		text_col: str, 
		custom_stopwords: set[str]
) -> tuple[pd.DataFrame, str]:
    """
    텍스트 컬럼을 전처리하고 CountVectorizer를 사용하여 단어 빈도를 추출.
    """
    # 데이터 프레임에서 텍스트 컬럼을 추출하여 하나의 긴 문자열로 결합
    raw_text = ' '.join(df[text_col].dropna().tolist())
    # 💡 긴 문자열로 결합된 텍스트에서 특수문자를 제거하고 소문자로 정제
    text_clean = re.sub(r'[^가-힣a-zA-Z\s]', ' ', raw_text).lower()
    
    # 영문 불용어 리스트에 사용자 정의 불용어 추가  
    # 🚨 수정: frozenset을 set()으로 변환하여 update가 가능하도록 합니다.
    all_stopwords = set(CountVectorizer(stop_words='english').get_stop_words())
    # 이제 all_stopwords는 일반 set이므로 update가 가능합니다.
    all_stopwords.update(custom_stopwords)
    
    # CountVectorizer()는 기본적으로 2단어부터 출력된다 r"(?u)\b\w\w+\b" 내포함.
    # 💡 토큰 패턴(token_pattern=r'(?u)\b\w\w+\b')을 사용하는 이유 명시적으로 인식하기 위해 
    vectorizer = CountVectorizer(
        # 🚨 수정: set 형태인 all_stopwords를 list()로 변환하여 전달합니다.
        stop_words=list(all_stopwords),
        token_pattern=r'(?u)\b\w\w+\b' 
    )
    
    # 합쳐진 불용어를 사용하여 단어 빈도 행렬을 생성
    word_matrix = vectorizer.fit_transform([text_clean])
    word_freq = word_matrix.toarray().flatten()    # 행렬을 1차원 배열로 변환 flatten /ˈflatn/ 단조롭게하다
    
		# 빈도 결과를 dataFrame으로 정리(word_df)
    word_df = pd.DataFrame({
        'word': vectorizer.get_feature_names_out(), 
        'freq': word_freq
    }).sort_values(by='freq', ascending=False)
    
    return word_df, text_clean
    
# --- 3. 시각화 함수 ---
def visualize_results(word_df: pd.DataFrame, top_n: int, title_prefix: str) -> None:
    """
    상위 단어에 대한 Bar plot과 WordCloud를 생성하고 표시합니다.
    """
    top_words_df = word_df.head(top_n)

    # 3.1 Bar Plot 시각화
    plt.figure(figsize=(10,6))
    sns.barplot(
	    data=top_words_df, 
	    x='freq', 
	    y='word', 
	    hue='word',         # ✅ y 변수인 'word'를 hue에 할당
	    palette='viridis',
	    legend=False        # ✅ 불필요한 범례를 숨김
		)
    plt.title(f'{title_prefix} Top {top_n} Words in Descriptions', fontsize=16)
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.show()

    # 3.2 WordCloud 시각화
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
        dict(zip(word_df['word'], word_df['freq']))
    )
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'{title_prefix} WordCloud', fontsize=16)
    plt.axis('off')
    plt.show()

# --- 4. 메인 실행 블록 ---
if __name__ == "__main__":
    
    # 💡 분석할 대상을 명시적으로 정의 (하드코딩 제거)
    CSV_FILE = 'netflix_preprocessed.csv'
    FILTER_KEYWORD = 'Korea'
    COLUMNS_TO_CHECK = [DESCRIPTION_COLUMN, TITLE_COLUMN, GENRE_COLUMN]
    TOP_N_WORDS = 10
    
    print(f"--- 넷플릭스 '{FILTER_KEYWORD}' 콘텐츠 분석 시작 ---")
    
    # 1. 데이터 로드 및 필터링
    korea_df = load_and_filter_data(
        file_path=CSV_FILE,
        filter_keyword=FILTER_KEYWORD,
        cols_to_check=COLUMNS_TO_CHECK
    )
    
    if korea_df is None:
        exit()
        
    print(f"✅ '{FILTER_KEYWORD}' 관련 콘텐츠 총 {len(korea_df)}개 발견.")
    print(korea_df)

    # 2. 텍스트 전처리 및 분석
    word_freq_df, _ = analyze_word_frequency(
        df=korea_df, 
        text_col=DESCRIPTION_COLUMN,
        custom_stopwords=CUSTOM_STOP_WORDS
    )
    print(word_freq_df)

    # 3. 시각화
    visualize_results(
        word_df=word_freq_df, 
        top_n=TOP_N_WORDS, 
        title_prefix=f'KOREAN Netflix Content'
    )
    
    print("--- 분석 완료 ---")