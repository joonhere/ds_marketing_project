from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
# from typing import list # Python 3.10+에서는 list[str] 사용

def apply_vectorization(text_data: list[str]) -> None:
    """
    CountVectorizer에 문서 목록(list of documents)을 안전하게 전달하는 함수.
    """
    if not isinstance(text_data, list):
        # API 제약을 벗어난 잘못된 사용을 방지하는 방어적 코드
        raise TypeError("text_data는 반드시 문서 목록(list) 형태여야 합니다.")
    # CountVectorizer()는 기본적으로 2단어부터 출력된다 r"(?u)\b\w\w+\b" 내포함. 
    vectorizer = CountVectorizer()
    
    # fit_transform은 문서 목록을 받으므로, list 형태가 필수적임
    word_matrix = vectorizer.fit_transform(text_data) 
    
    # 1개의 문서가 입력되면 1xN 행렬, 5개의 문서가 입력되면 5xN 행렬이 생성됨.
    print(f"생성된 행렬 크기: {word_matrix.shape}")

# ✅ 올바른 사용 예시 (단일 문서)
# text_clean은 이미 결합된 하나의 긴 문자열이라고 가정합니다.
text_clean: str = "python is great for analysis and python is fast"
apply_vectorization([text_clean]) 
# 출력: 생성된 행렬 크기: (1, 7)

# ✅ 올바른 사용 예시 (여러 문서)
multiple_documents: list[str] = [
    "영화 리뷰 1번입니다.",
    "이 영화는 정말 재미있습니다.",
    "분석을 위해 세 번째 문서를 추가합니다."
]
apply_vectorization(multiple_documents) 
# 출력: 생성된 행렬 크기: (3, 11)

from sklearn.feature_extraction.text import CountVectorizer

docs = [
    "영화 리뷰 1번입니다.",
    "이 영화는 정말 재미있습니다.",
    "분석을 위해 세 번째 문서를 추가합니다."
]

vectorizer = CountVectorizer()
vectorizer.fit(docs)
print(vectorizer.get_feature_names_out())
