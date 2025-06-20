## 1. utils/visualization.py

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import platform
import streamlit as st
import io
import base64

# 폰트 설정 함수 (한 번만 실행되도록 캐싱)
@st.cache_resource
def setup_korean_font():
    # 1. 프로젝트 내 포함된 폰트 우선 적용
    import os
    font_path = os.path.join("fonts", "NanumGothic.ttf")

    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        mpl.rcParams["font.family"] = "NanumGothic"
        return font_path

    # 2. 시스템별 fallback (폰트가 없을 경우)
    system = platform.system()
    if system == "Windows":
        mpl.rcParams["font.family"] = "Malgun Gothic"
    elif system == "Darwin":
        mpl.rcParams["font.family"] = "AppleGothic"
    else:
        fallback_fonts = ["Noto Sans CJK KR", "NanumGothic", "Droid Sans Fallback", "UnDotum", "Liberation Sans"]
        available_fonts = set(f.name for f in fm.fontManager.ttflist)
        matched = next((font for font in fallback_fonts if font in available_fonts), None)

        if matched:
            mpl.rcParams["font.family"] = matched
        else:
            mpl.rcParams["font.family"] = "sans-serif"
            st.warning("⚠️ 한글 폰트가 시스템에 없어 기본 폰트로 대체됩니다. (한글 깨질 수 있음)")

    mpl.rcParams["axes.unicode_minus"] = False
    return None  # fallback일 경우 경로 반환 안 함

# 그래프 다운로드 기능
def get_image_download_link(fig, filename, text):
    """그래프를 이미지로 변환하고 다운로드 링크 생성"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# 한글 폰트가 적용된 그림 객체 생성
def create_figure(figsize=(10, 6), dpi=150):
    setup_korean_font()  # 폰트 설정 (캐싱되므로 첫 번째 호출만 실행됨)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax

# 메뉴별 색상 테마 정의
def get_color_theme(menu_name):
    color_themes = {
        "대시보드": "Blues",
        "고장 유형 분석": "Greens",
        "브랜드/모델 분석": "Oranges",
        "고장 예측": "viridis"
    }
    return color_themes.get(menu_name, "Blues")

