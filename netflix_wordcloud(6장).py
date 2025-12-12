import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from PIL import Image
import random
from typing import Optional, Dict, Any, Set, List, Tuple
import re # 정규표현식 사용

# --- 상수 정의 (유지보수 용이성 확보) ---
CSV_FILE = 'netflix_preprocessed.csv'
MASK_FILE = 'netflix_logo.jpg'
PLOT_TITLE_WC = 'Keywords in KOREAN Netflix Content Descriptions'
PLOT_TITLE_GENRE = 'Top 10 Genres in KOREAN Netflix Content'
TEXT_COLUMN = 'description'
COUNTRY_COLUMN = 'country'
GENRE_COLUMN = 'listed_in'
RANDOM_SEED = 42

# 💡 U+00A0 오류 방지: 한 줄로 정의함
DEFAULT_STOPWORDS = {'series', 'film', 'movie', 'show', 'story', 'life', 'new', 'world', 'us', 'based', 'one', 'two', 'young', 'old', 'about', 'from', 'with', 'who', 'when', 'what', 'where', 'their', 'they', 'them', 'this', 'that', 'these', 'those', 'also', 'after', 'before', 'just', 'much', 'many', 'more', 'most', 'very', 'get', 'got', 'make', 'made', 'take', 'takes', 'find', 'found', 'come', 'comes', 'go', 'goes', 'see', 'saw', 'said', 'say', 'into', 'through', 'while', 'upon', 'among', 'across', 'always', 'ever', 'never', 'might', 'must', 'should', 'could', 'would', 'can', 'will', 'may', 'way', 'time', 'years', 'first', 'their', 'all', 'its', 'her', 'his', 'which', 'had', 'etc', 'korean', 'korea', 'drama', 'kdrama', 'a', 'an', 'to', 'of', 'for', 'in', 'on', 'at', 'and', 'the', 'is', 'but', 'as', 'by', 'he', 'she', 'out', 'up'}

# --- 1. 데이터 로드 및 필터링 ---

def load_and_filter_data(file_path: str, country_col: str, genre_col: str) -> Optional[pd.DataFrame]:
    """지정된 CSV 파일을 로드하고 'Korea' 관련 콘텐츠를 필터링합니다."""
    try:
        df = pd.read_csv(file_path)
        korea_df = df[
            df[country_col].fillna('').str.contains('Korea', case=False, na=False) |
            df[genre_col].fillna('').str.contains('Korean', case=False, na=False)
        ].copy()
        
        if korea_df.empty:
            print("🚨 경고: 'Korea' 관련 콘텐츠를 찾을 수 없습니다.")
            return None
        return korea_df
    except FileNotFoundError:
        print(f"🚨 오류: {file_path} 파일을 찾을 수 없습니다. 프로그램 실행 중단.")
        return None
    except KeyError as e:
        print(f"🚨 오류: 필수 컬럼 '{e}'을(를) 찾을 수 없습니다. 데이터셋 확인 필요.")
        return None

# --- 2. 피처 엔지니어링 ---

def engineer_korea_features(df: pd.DataFrame, text_col: str, genre_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """KOREA 콘텐츠 데이터프레임에 새로운 피처를 엔지니어링하고, 장르 빈도 데이터를 추출합니다."""
    all_k_genres = df[genre_col].str.split(', ').explode().dropna()
    k_genre_counts = all_k_genres.value_counts().head(10)
    df['K_Drama_Flag'] = df[text_col].fillna('').apply(
        lambda x: 1 if 'drama' in x.lower() or 'kdrama' in x.lower() else 0
    )
    return df, k_genre_counts

# --- 3. 텍스트 전처리 ---

def preprocess_text_for_wordcloud(df: pd.DataFrame, text_col: str, stopwords: Set[str]) -> str:
    """워드 클라우드용으로 텍스트를 추출하고 전처리합니다."""
    clean_descriptions = df[text_col].fillna('')
    combined_text = clean_descriptions.str.cat(sep=' ')
    combined_text = re.sub(r'[^가-힣a-zA-Z\s]', '', combined_text) 
    combined_text = combined_text.lower()
    
    words = combined_text.split()
    filtered_words = [word for word in words if word not in stopwords and len(word) > 1] 
    
    return ' '.join(filtered_words)

# --- 4. 워드 클라우드 관련 유틸리티 ---

def load_mask(mask_path: str) -> Optional[np.ndarray]:
    """마스크 이미지를 로드하고 NumPy 배열로 변환합니다."""
    try:
        return np.array(Image.open(mask_path))
    except FileNotFoundError:
        print(f"🚨 경고: {mask_path} 파일을 찾을 수 없어 마스크 없이 진행합니다.")
        return None

def netflix_color_func(word: str, font_size: int, position: tuple, orientation: int, 
                       random_state: Optional[int] = None, **kwargs: Dict[str, Any]) -> str:
    """WordCloud 객체를 위한 넷플릭스 테마(레드, 블랙) 무작위 색상 선택 함수."""
    colors = ['#221F1F', '#B20710'] 
    return random.choice(colors)

def generate_wordcloud_object(text: str, mask: Optional[np.ndarray], stopwords: Set[str]) -> WordCloud:
    """결합된 텍스트와 마스크를 사용하여 WordCloud 객체를 생성합니다."""
    return WordCloud(
        background_color='white',
        width=1400,
        height=1400,
        max_words=170,
        mask=mask,
        color_func=netflix_color_func,
        collocations=False, 
        stopwords=stopwords, 
        random_state=RANDOM_SEED
    ).generate(text)

# --- 5. 시각화 함수 ---

def plot_genre_distribution(genre_counts: pd.Series, title: str) -> None:
    """Seaborn을 사용하여 KOREA 콘텐츠의 장르 분포를 시각화하고 화면에 표시합니다."""
    plt.figure(figsize=(12, 6))
    
    sns.barplot(
        x=genre_counts.index, 
        y=genre_counts.values,
        hue=genre_counts.index, 
        palette='viridis',
        legend=False
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def save_wordcloud_image_final(wordcloud: WordCloud, title: str, filename: str = "korean_netflix_wordcloud.png") -> None:
    """
    워드 클라우드 결과물을 파일로 직접 저장하며, bbox_inches='tight'로 제목 잘림을 방지하고 화면에 출력합니다.
    """
    plt.figure(figsize=(15, 6)) 
    
    # suptitle 설정 (y 값을 수동으로 조정할 필요 없이 기본값 사용)
    plt.suptitle(title, fontweight='bold', fontfamily='serif', fontsize=18) 
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # ⭐ Critical Path Solution: bbox_inches='tight'로 모든 요소가 포함되도록 저장합니다.
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ 워드 클라우드 이미지가 '{filename}' 파일로 성공적으로 저장되었습니다.")
        print("파일이 스크립트 실행 폴더에 저장되었는지 확인해주세요.")
    except Exception as e:
        print(f"🚨 파일 저장 중 오류가 발생했습니다: {e}")
    
    # 💡 화면 출력 추가: 이 때 화면상에서는 제목이 잘려 보일 수 있으나, 파일은 완벽합니다.
    plt.show() 
    plt.close()

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    print("--- 넷플릭스 KOREA 콘텐츠 분석 시작 ---")

    # 1. 데이터 로드 및 필터링
    korea_df_filtered = load_and_filter_data(CSV_FILE, COUNTRY_COLUMN, GENRE_COLUMN)
    if korea_df_filtered is None:
        exit()

    # 2. 피처 엔지니어링
    korea_df_processed, k_top_genres = engineer_korea_features(
        korea_df_filtered, TEXT_COLUMN, GENRE_COLUMN
    )
    print(f"\nKOREA 콘텐츠 총 {len(korea_df_processed)}개 발견.")
    
    # 3. 텍스트 전처리
    wordcloud_text = preprocess_text_for_wordcloud(
        korea_df_processed, TEXT_COLUMN, DEFAULT_STOPWORDS
    )
    if not wordcloud_text: 
        print("🚨 오류: 워드 클라우드를 생성할 텍스트가 충분하지 않습니다. 프로그램 종료.")
        exit()

    # 4. 마스크 로드
    mask_array = load_mask(MASK_FILE)

    # 5. 워드 클라우드 객체 생성
    wordcloud_obj = generate_wordcloud_object(
        wordcloud_text, mask_array, DEFAULT_STOPWORDS
    )

    # 6. 결과 시각화
    print("\n--- 분석 결과 시각화 ---")
    
    # 6.1. 장르 분포 그래프 (화면 표시)
    plot_genre_distribution(k_top_genres, PLOT_TITLE_GENRE)
    
    # 6.2. 워드 클라우드 (파일 저장 및 화면 출력)
    save_wordcloud_image_final(wordcloud_obj, PLOT_TITLE_WC)

    print("--- 넷플릭스 KOREA 콘텐츠 분석 완료 ---")