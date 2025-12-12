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