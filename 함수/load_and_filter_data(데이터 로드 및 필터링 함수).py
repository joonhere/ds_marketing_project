def load_and_filter_data(
    file_path: str,
    filter_keyword: str,
    cols_to_check: List[str]
) -> Optional[pd.DataFrame]:

    try:
        df = pd.read_csv(file_path)    #1
        
        filter_condition = False    #2

        for col in cols_to_check:    #3
            if col in df.columns:
                filter_condition = filter_condition | df[col].fillna(' ').str.contains(filter_keyword, case=False, na=False)    #
            else:
                print(f"경고: 컬럼 '{col}'을 찾을 수 없습니다.")    #

        filtered_df = df[filter_condition].copy()    #4

        if filter_df.empty:    #5
            print(f"경고: '{filter_keyword}' 관련 콘텐츠를 찾을 수 없습니다.")
            return None
        
        return filter_df  #6

    except FileNotFoundError:    #7
        print(f"오류: '{file_path}' 경로에 파일을 찾을 수 없습니다.")
        return None

# 1. 데이터 로드

# 2. OR 조건 누적을 위한 False 초기화
 # filter_condition = False
  # 다중 필터링 조건을 안전하게 누적하기 위한 초기화 역할
  # '아무 행도 선택하지 않음'을 의미 그래야 선택한 행을 받아 쓸 수 있음

# 3. 컬럼별 필터링 및 OR 연산 누적
 # OR 누적: False | X = X (첫 루프), X | Y = X or Y (이후 루프)
  # df[col].fillna(' ').str.contains(filter_keyword, case=False, na=False)
  # .fillna(' ')와 na=False가 함께 있어 안전하지만,
  # fillna를 이미 썼다면 na=False는 선택 사항입니다. (현 코드에서는 이중 안전 장치로 간주)

 # 존재하지 않는 컬럼은 건너뛰고 경고 출력
    
# 4. 필터링된 데이터프레임 생성 및 복사본(copy)으로 독립성 확보

# 5. 데이터 없음 예외 처리 (조기 종료)

# 6. 최종 성공 결과 반환

# 7. 파일 로드 실패 예외 처리 (조기 종료)