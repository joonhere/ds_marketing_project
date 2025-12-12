import pandas as pd
# pandas는 이미 import 되었다고 가정

def process_data() -> tuple[pd.DataFrame, str]:
    """
    분석을 수행하고 결과 DataFrame과 정리된 텍스트를 반환한다고 가정합니다.
    """
    # 임시 결과 생성
    result_df = pd.DataFrame({'word': ['python', 'code'], 'freq': [10, 5]})
    clean_text = "clean python code example"
    
    # 두 개의 요소를 가진 튜플 반환
    return result_df, clean_text 

# --- 함수 호출 및 활용 ---
# 튜플의 언패킹(Unpacking)을 통해 각 요소를 개별 변수에 할당할 수 있습니다.
word_freq_table, cleaned_data_string = process_data()

print("--- 반환된 DataFrame 정보 ---")
print(f"변수명: word_freq_table")
print(word_freq_table)

print("\n--- 반환된 문자열 정보 ---")
print(f"변수명: cleaned_data_string")
print(f"길이: {len(cleaned_data_string)}자")