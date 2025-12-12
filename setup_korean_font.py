import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform

def setup_korean_font():
    """
    시스템에서 사용 가능한 한글 폰트를 찾아 Matplotlib의 기본 폰트로 설정합니다.
    (Windows: Malgun Gothic, Mac: AppleGothic, 그 외: NanumGothic 등)
    """
    
    # 폰트 우선순위 설정 (시스템에 있을 확률이 높은 순)
    # *HYGothic-Medium, Malgun Gothic은 Windows에 주로 존재
    # *NanumGothic은 Jupyter/Colab/Linux에서 가장 흔함
    # *AppleGothic은 Mac OS에 존재
    font_preferences = ['Malgun Gothic', 'HYGothic-Medium', 'NanumGothic', 'AppleGothic', 'DejaVu Sans']
    selected_font = 'DejaVu Sans'
    
    # 1. Matplotlib 설정 초기화 및 마이너스 부호 설정
    plt.rcParams['axes.unicode_minus'] = False
    
    # 2. 시스템 폰트 검색 및 설정
    for preferred_font in font_preferences:
        font_path = None
        # 폰트 파일 경로 검색
        for font in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
            # 폰트 이름에 선호 폰트 이름이 포함되어 있는지 확인
            if preferred_font.lower().replace(' ', '') in font.lower().replace(' ', ''):
                font_path = font
                selected_font = preferred_font
                break
        
        if font_path:
            # 폰트 경로를 Matplotlib에 추가하고 기본 폰트로 설정
            try:
                font_manager.fontManager.addfont(font_path)
                rc('font', family=selected_font)
                print(f"✅ Matplotlib 한글 폰트가 '{selected_font}'으로 설정되었습니다.")
                return selected_font
            except Exception as e:
                # 폰트 로딩 실패 시 다음 폰트 시도
                print(f"경고: 폰트 '{selected_font}' 로딩 실패 ({e}). 다음 폰트를 시도합니다.")
                continue

    # 3. 모든 폰트 검색 실패 시
    rc('font', family='DejaVu Sans')
    print(f"경고: 한글 폰트({', '.join(font_preferences)})를 찾을 수 없어 기본 폰트(DejaVu Sans)로 설정되었습니다. 한글이 깨질 수 있습니다.")
    return 'DejaVu Sans'

# 함수 사용 예시:
# setup_korean_font()