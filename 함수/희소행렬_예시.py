from sklearn.feature_extraction.text import CountVectorizer

# 1. 예시 텍스트 데이터 (작은 어휘 목록)
documents = [
    "저는 사과와 바나나를 좋아합니다", 
    "당신은 오직 바나나만 좋아합니까",
    "사과와 배는 맛있는 과일입니다"
]

# 2. CountVectorizer를 사용하여 문서-단어 행렬 생성
vectorizer = CountVectorizer()
word_matrix = vectorizer.fit_transform(documents) # 이 결과가 희소 행렬입니다.

# 3. 희소 행렬 정보 출력
print(f"--- 희소 행렬 정보 ---")
print(f"변환된 행렬 타입: {type(word_matrix)}")
# (3, 7)은 3개 문서, 7개 고유 단어를 의미합니다.
print(f"행렬 크기 (문서 수, 단어 수): {word_matrix.shape}")
print(f"저장된 0이 아닌 값의 개수: {word_matrix.nnz}")
print("-" * 20)

# 4. toarray()를 사용하여 밀집 행렬(일반 배열)로 변환
dense_matrix = word_matrix.toarray()

print(f"--- 밀집 행렬 (toarray() 변환) ---")
print(f"변환된 행렬 타입: {type(dense_matrix)}")
print(dense_matrix)

# 5. 고유 단어 (컬럼 이름) 확인
print("-" * 20)
print(f"고유 단어 목록: {vectorizer.get_feature_names_out()}")