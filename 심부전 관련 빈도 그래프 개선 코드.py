import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import sys  # 🔧 exit() 대신 sys.exit() 사용 (더 안전함)

# ==============================================================================
# 1. 환경 설정: Matplotlib 한글 폰트 설정
# ------------------------------------------------------------------------------
font_path = 'C:/Windows/Fonts/malgun.ttf'  # Windows 기본 폰트 경로
try:
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
except FileNotFoundError:
    print("⚠️ 경고: 'Malgun Gothic' 폰트를 찾을 수 없습니다. 그래프에 한글이 깨질 수 있습니다.")
# ==============================================================================


# 2. 데이터 로드 및 비율 계산
try:
    heart = pd.read_csv('heart.csv')
except FileNotFoundError:
    print("🚨 heart.csv 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
    sys.exit(1)  # 🔧 exit() 대신 sys.exit 사용


# a. HeartDisease와 ChestPainType 별 빈도 계산
cp_counts = heart.groupby(['HeartDisease', 'ChestPainType']).size()

# b. HeartDisease 그룹 내부 비율 계산
cp_proportions = cp_counts.groupby(level=0).apply(lambda x: x / x.sum() * 100)

# c. ChestPainType을 컬럼으로 변환
cp_ratio_for_plot = cp_proportions.unstack(level=1)

# 3. 데이터 순서 정리 (KeyError 방지)
order = ["ASY", "NAP", "ATA", "TA"]
cp_ratio_for_plot = cp_ratio_for_plot.reindex(columns=order)
cp_ratio_for_plot.columns.name = None


# ==============================================================================
# 4. 시각화 (Stacked Bar Plot)
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

cp_ratio_for_plot.plot(
    kind='bar',
    stacked=True,
    ax=ax,
    color=sns.color_palette('viridis', n_colors=len(order)),
    width=0.7
)

# 🔧 Critical Path Fix: X축 눈금 수동 지정
ax.set_xticks([0, 1])
ax.set_xticklabels(['정상 (0)', '심장병 (1)'], rotation=0, fontsize=11)

ax.set_ylabel('흉통 유형 비율 (%)', fontsize=13)
ax.set_xlabel('심장병 유무', fontsize=13)
ax.set_yticks(range(0, 101, 20))


# ==============================================================================
# 5. 비율 텍스트 추가
# ------------------------------------------------------------------------------
for i, col in enumerate(cp_ratio_for_plot.columns):
    y_offset = 0
    for j, val in enumerate(cp_ratio_for_plot[col]):
        if val > 5:
            ax.text(j,
                    y_offset + val / 2,
                    f'{val:.1f}%',
                    ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        y_offset += val


# ==============================================================================
# 6. 제목 및 범례
# ------------------------------------------------------------------------------
plt.suptitle(
    '심장병 유무에 따른 흉통 유형별 비율 분석',
    fontsize=16, fontweight='bold', color='darkslategray'
)
plt.legend(title='흉통 유형', loc='upper right', fontsize=9, title_fontsize=10)


# ==============================================================================
# 7. 플롯 출력 (환경별 안정화)
# ------------------------------------------------------------------------------
# Jupyter 환경이라면 plt.show(block=False) + plt.pause 사용
# 터미널 실행이라면 plt.show(block=True)가 더 자연스러움
try:
    plt.show(block=True)
except Exception:
    plt.show(block=False)
    plt.pause(3)
finally:
    plt.close(fig)
# ==============================================================================
