# 튜플 언패킹(Unpacking)을 통해 두 변수에 동시에 저장
result_df, status_message = analyze_word_frequency(df, 'description', stopwords)

# 상태를 먼저 확인하여 안정적으로 다음 단계로 넘어감
if "Error" in status_message:
    print(f"분석 실패: {status_message}")
else:
    # result_df를 안전하게 사용
    print("분석 완료!")