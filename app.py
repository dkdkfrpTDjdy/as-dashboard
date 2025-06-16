import streamlit as st
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import urllib.request
import platform

# 페이지 설정이 가장 먼저 와야 함
st.set_page_config(
    page_title="산업장비 AS 분석 대시보드",
    layout="wide"
)

def setup_korean_font_test():
    # 1. 프로젝트 내 포함된 폰트 우선 적용
    font_path = os.path.join("fonts", "NanumGothic.ttf")
    
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        mpl.rcParams["font.family"] = "NanumGothic"
        # 디버깅용
        # st.info("✅ NanumGothic.ttf 폰트를 직접 등록하여 사용합니다.")
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
            # 디버깅용
            # st.info(f"✅ 시스템에서 발견된 한글 폰트 사용: {matched}")
        else:
            mpl.rcParams["font.family"] = "sans-serif"
            
            # 디버깅용
            st.warning("⚠️ 한글 폰트가 시스템에 없어 기본 폰트로 대체됩니다. (한글 깨질 수 있음)")

    mpl.rcParams["axes.unicode_minus"] = False
    return None  # fallback일 경우 경로 반환 안 함

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import re
import matplotlib.font_manager as fm
from collections import Counter
from wordcloud import WordCloud
import folium
from streamlit_folium import folium_static
import io
import base64
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
import requests
import subprocess

# 폰트 설정 실행
font_path = setup_korean_font_test()
# 이후 코드에서 사용 가능
if font_path and os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
 
# 그래프 다운로드 기능 추가
def get_image_download_link(fig, filename, text):
    """그래프를 이미지로 변환하고 다운로드 링크 생성"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}"> {text}</a>'
    return href
  
def create_figure_with_korean(figsize=(10, 6), dpi=300):
    """한글 폰트가 적용된 그림 객체 생성""" 
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax

# 메뉴별 색상 테마 설정
color_themes = {
    "대시보드": "Blues",
    "고장 유형 분석": "Greens",
    "브랜드/모델 분석": "Oranges",
    "정비내용 분석": "YlOrRd",
    "고장 예측": "viridis"
}

# 사이드바 설정
st.sidebar.title("데이터 업로드 및 메뉴 클릭")

# 파일 업로더 - 두 개의 파일 업로드 기능 추가
uploaded_file1 = st.sidebar.file_uploader("**AS 데이터 업로드**: AS 데이터: 구 ERP > 자산관리 > 정비관리 > 건설장비AS", type=["xlsx"])
uploaded_file2 = st.sidebar.file_uploader("**자산조회 파일 업로드**: SAP > 자산조회", type=["xlsx"])

# 데이터 로드 함수
@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        return None

# 지역 추출 함수
def extract_first_two_chars(address):
    if isinstance(address, str) and len(address) >= 2:
        return address[:2]
    return None

# 문자열 리스트로 변환하는 함수 (NaN과 혼합 유형 처리용)
def convert_to_str_list(arr):
    return [str(x) for x in arr if not pd.isna(x)]

# 작은 비율 항목을 '기타'로 그룹화하는 함수
def group_small_categories(series, threshold=0.03):
    total = series.sum()
    mask = series / total < threshold
    if mask.any():
        others = pd.Series({'기타': series[mask].sum()})
        return pd.concat([series[~mask], others])
    return series

# 두 데이터프레임 병합 함수
def merge_dataframes(df1, df2):
    if df1 is None or df2 is None:
        return None
    
    try:
        # 관리번호 컬럼을 기준으로 두 데이터프레임 병합
        # 필요한 컬럼만 선택하여 병합
        df2_subset = df2[['관리번호', '제조사명', '제조년도', '취득가', '자재내역']]
        
        # 왼쪽 조인으로 병합 (AS 데이터는 모두 유지)
        merged_df = pd.merge(df1, df2_subset, on='관리번호', how='left')
        
        # 중복 행 제거
        merged_df = merged_df.drop_duplicates()
        
        # 자재내역 컬럼 분할
        if '자재내역' in merged_df.columns:
            # 자재내역에서 추가 정보 추출 (공백으로 나누기)
            merged_df[['연료', '운전방식', '적재용량', '마스트형']] = merged_df['자재내역'].str.split(' ', n=3, expand=True)
        
        return merged_df
    except Exception as e:
        st.error(f"데이터 병합 중 오류 발생: {e}")
        return df1  # 오류 발생시 원본 데이터프레임 반환

# 샘플 데이터 경로 또는 업로드된 파일 사용
if uploaded_file1 is not None:
    df1 = load_data(uploaded_file1)
    file_name1 = uploaded_file1.name

    if uploaded_file2 is not None:
        df2 = load_data(uploaded_file2)
        # 두 데이터프레임 병합
        df = merge_dataframes(df1, df2)
        if df is not None:
            st.sidebar.success("두 파일이 성공적으로 병합되었습니다.")
            file_name = f"{file_name1} (병합됨)"
        else:
            df = df1
            file_name = file_name1
    else:
        df = df1
        file_name = file_name1
        st.sidebar.warning("자산조회 파일이 업로드되지 않았습니다. AS 데이터만 사용합니다.")
else:
    df = None
    file_name = None

# 데이터가 로드된 경우 분석 시작
if df is not None:
    # 메뉴 선택 - 데이터 개요와 KPI 대시보드를 합침
    menu = st.sidebar.selectbox(
        "분석 메뉴",
        ["대시보드", "고장 유형 분석", "브랜드/모델 분석", "정비내용 분석", "고장 예측"]
    )
    
    # 현재 메뉴의 색상 테마 설정
    current_theme = color_themes[menu]
    
    # 데이터 전처리
    try:
        # 날짜 변환
        df['접수일자'] = pd.to_datetime(df['접수일자'], errors='coerce')
        df['정비일자'] = pd.to_datetime(df['정비일자'], errors='coerce')
        df['최근정비일자'] = pd.to_datetime(df['최근정비일자'], errors='coerce')
        
        # 6. 처리일수에서 이상치인 -1 값 삭제
        df['AS처리일수'] = (df['정비일자'] - df['접수일자']).dt.days
        df = df[df['AS처리일수'] >= 0]  # 음수 제거
        
        # 재정비 간격 계산 (정비일자 - 최근정비일자)
        df['재정비간격'] = (df['정비일자'] - df['최근정비일자']).dt.days
        
        # 30일 내 재정비 여부
        df['30일내재정비'] = (df['재정비간격'] <= 30) & (df['재정비간격'] > 0)
        
        # 브랜드 컬럼 처리 - 제조사명 컬럼이 있으면 그것을 사용, 없으면 '기타'로 채움
        if '제조사명' in df.columns:
            df['브랜드'] = df['제조사명'].fillna('기타')
        else:
            df['브랜드'] = '기타'
        
        # 지역 추출
        if 'ADDR' in df.columns:
            df['지역'] = df['ADDR'].apply(extract_first_two_chars)

        df.rename(columns={
        '대분류': '작업유형',
        '중분류': '정비대상',
        '소분류': '정비작업'
        }, inplace=True)
        
        
        # 고장유형 조합
        if all(col in df.columns for col in ['작업유형', '정비대상', '정비작업']):
            df['고장유형'] = df['작업유형'].astype(str) + '_' + df['정비대상'].astype(str) + '_' + df['정비작업'].astype(str)
            df['브랜드_모델'] = df['브랜드'].astype(str) + '_' + df['모델명'].astype(str)
    except Exception as e:
        st.warning(f"일부 데이터 전처리 중 오류가 발생했습니다: {e}")
    
    # 메뉴별 콘텐츠 표시
    if menu == "대시보드":
        st.title("대시보드")
        
        # 지표 카드용 컬럼 생성
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_cases = len(df)
            st.metric("총 AS 건수", f"{total_cases:,}")
        
        with col2:
            if 'AS처리일수' in df.columns:
                avg_days = df['AS처리일수'].mean()
                st.metric("평균 처리일수", f"{avg_days:.2f}일")
            else:
                st.metric("평균 처리일수", "데이터 없음")
        
        with col3:
            if 'AS처리일수' in df.columns:
                within_3_days = (df['AS처리일수'] <= 2).mean() * 100
                st.metric("2일 이내 처리율", f"{within_3_days:.1f}%")
            else:
                st.metric("2일 이내 처리율", "데이터 없음")
        
        with col4:
            if '접수일자' in df.columns:
                last_month = df['접수일자'].max().strftime('%Y-%m')
                last_month_count = df[df['접수일자'].dt.strftime('%Y-%m') == last_month].shape[0]
                st.metric("최근 월 AS 건수", f"{last_month_count:,}")
            else:
                st.metric("최근 월 AS 건수", "데이터 없음")
                
        st.markdown("---")
        
        # 1. 월별 AS건수 + 월별 평균 AS 처리일수 + AS 처리 일수 분포
        col1, col2, col3 = st.columns(3)
            
        with col1:
            # 월별 AS 건수
            st.subheader("월별 AS 건수")
            if '접수일자' in df.columns:
                df_time = df.copy()
                df_time['월'] = df_time['접수일자'].dt.to_period('M')
                monthly_counts = df_time.groupby('월').size().reset_index(name='건수')
                monthly_counts['월'] = monthly_counts['월'].astype(str)
                
                # 고해상도 그래프를 위한 설정
                fig, ax = create_figure_with_korean(figsize=(10, 6), dpi=300)
                sns.barplot(x='월', y='건수', data=monthly_counts, ax=ax, palette='Blues')
                
                # 막대 위에 텍스트 표시
                for i, v in enumerate(monthly_counts['건수']):
                    ax.text(i, v + max(monthly_counts['건수']) * 0.01, str(v), ha='center')
                
                plt.xticks(rotation=45)
                ax.set_ylabel('건수')
                plt.tight_layout()  # 여백 자동 조정
                
                # 고품질 표시 옵션 추가
                st.pyplot(fig, use_container_width=True)
                
                # 다운로드 링크 추가
                st.markdown(get_image_download_link(fig, '월별_AS_건수.png', '월별 AS 건수 다운로드'), unsafe_allow_html=True)
                
        with col2:
            if 'AS처리일수' in df.columns:
                st.subheader("평균 AS 처리일수")
                df['월'] = df['접수일자'].dt.to_period('M').astype(str)
                monthly_avg = df.groupby('월')['AS처리일수'].mean().reset_index()
                
                fig, ax = create_figure_with_korean(figsize=(10, 6), dpi=300)
                sns.barplot(data=monthly_avg, x='월', y='AS처리일수', ax=ax, palette=f"{current_theme}")
                
                # 평균값 텍스트 표시
                for index, row in monthly_avg.iterrows():
                    ax.text(index, row['AS처리일수'] + 0.02, f"{row['AS처리일수']:.1f}일", ha='center')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig, use_container_width=True)
                
                # 다운로드 링크 추가
                st.markdown(get_image_download_link(fig, '월별_평균_처리일수.png', '월별 평균 처리일수 다운로드'), unsafe_allow_html=True)
            
        with col3:
            if 'AS처리일수' in df.columns:
                st.subheader("AS 처리 일수 분포")
                
                days_counts = df['AS처리일수'].value_counts().sort_index()
                days_counts = days_counts[days_counts.index <= 10]  # 10일 이하만 표시
                
                fig, ax = create_figure_with_korean(figsize=(10, 6), dpi=300)
                sns.barplot(x=days_counts.index, y=days_counts.values, ax=ax, palette=f"{current_theme}")
                
                # 막대 위에 텍스트 표시
                for i, v in enumerate(days_counts.values):
                    ax.text(i, v + max(days_counts.values) * 0.02, str(v),
                           ha='center', fontsize=12)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                
                # 다운로드 링크 추가
                st.markdown(get_image_download_link(fig, 'AS_처리일수_분포.png', 'AS 처리일수 분포 다운로드'), unsafe_allow_html=True)
                
        st.markdown("---")
        
        # 2. 30일 내 재정비 현장 TOP 10 + 30일 내 재정비율 + 2일 이내 처리율
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("30일 내 재정비 현장 TOP 10")
            
            # 현장별 30일 내 재정비 건수 계산
            if '현장명' in df.columns:
                repair_by_site = df[df['30일내재정비'] == True].groupby('현장명').size().sort_values(ascending=False).head(10)
                
                if len(repair_by_site) > 0:
                    fig, ax = create_figure_with_korean(figsize=(10, 10), dpi=300)
                    sns.barplot(y=repair_by_site.index, x=repair_by_site.values, ax=ax, palette=f"{current_theme}_r")
                    
                    # 막대 끝에 텍스트 표시
                    for i, v in enumerate(repair_by_site.values):
                        ax.text(v + 0.1, i, str(v), va='center')
                    
                    ax.set_xlabel('재정비 건수')
                    ax.set_ylabel('현장명')
                    plt.tight_layout()
                    
                    st.pyplot(fig, use_container_width=True)
                    
                    # 다운로드 링크 추가
                    st.markdown(get_image_download_link(fig, '30일내_재정비_현장_TOP10.png', '30일 내 재정비 현장 TOP10 다운로드'), unsafe_allow_html=True)
                else:
                    st.warning("30일 내 재정비 데이터가 충분하지 않습니다.")
            else:
                st.warning("현장명 데이터가 없습니다.")
                
        with col2:
            # 30일 내 재정비율 추가
            st.subheader("재정비율")
            
            # 유효한 데이터만 사용
            valid_repair = df.dropna(subset=['정비일자', '최근정비일자']).copy()
            valid_repair = valid_repair[valid_repair['재정비간격'] > 0]  # 정비간격이 0보다 큰 경우만
            
            if len(valid_repair) > 0:
                # 30일 내 재정비 여부에 따른 건수 계산
                within_30_days = valid_repair['30일내재정비'].sum()
                over_30_days = len(valid_repair) - within_30_days
                
                labels = ['30일 이내 재정비', '30일 초과 재정비']
                sizes = [within_30_days, over_30_days]
                
                # 파이 차트 그리기
                fig, ax = create_figure_with_korean(figsize=(8, 8), dpi=300)
                ax.pie(
                    sizes,
                    labels=labels,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=[sns.color_palette(current_theme)[0], sns.color_palette(current_theme)[2]],
                    textprops={'fontsize': 14}
                )
                ax.axis('equal')
                plt.tight_layout()
                
                st.pyplot(fig, use_container_width=True)
                
                # 다운로드 링크 추가
                st.markdown(get_image_download_link(fig, '30일내_재정비율.png', '30일 내 재정비율 다운로드'), unsafe_allow_html=True)
            else:
                st.warning("재정비 데이터가 부족합니다.")
                
        with col3:
            if 'AS처리일수' in df.columns:
                st.subheader("2일 이내 처리율")
                
                threshold_days = 2
                within_threshold = df['AS처리일수'] <= threshold_days
                counts = within_threshold.value_counts()
                labels = [f'{threshold_days}일 이내 처리', f'{threshold_days}일 초과 처리']
                sizes = [counts.get(True, 0), counts.get(False, 0)]
                colors = [sns.color_palette(current_theme)[0], sns.color_palette(current_theme)[2]]
                
                # 파이 차트 그리기
                fig, ax = create_figure_with_korean(figsize=(8, 8), dpi=300)
                ax.pie(
                    sizes,
                    labels=labels,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors,
                    textprops={'fontsize': 14}
                )
                plt.tight_layout()
                
                st.pyplot(fig, use_container_width=True)
                
                # 다운로드 링크 추가
                st.markdown(get_image_download_link(fig, '2일이내_처리율.png', '2일 이내 처리율 다운로드'), unsafe_allow_html=True)
                
        st.markdown("---")
        
        # 3. 업체별 AS 건수 Top 15 + 현장별 AS 건수 Top 15 + 지역별 AS 건수
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if '업체명' in df.columns:
                st.subheader("업체별 AS 건수 Top 15")
                top_vendors = df['업체명'].value_counts().nlargest(15)
                
                # 업체별, 현장별 그래프 크기 맞춤
                fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                sns.barplot(x=top_vendors.values, y=top_vendors.index, ax=ax, palette=f"{current_theme}")
                
                for index, value in enumerate(top_vendors.values):
                    ax.text(value + 1, index, str(value), va='center')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                
                # 다운로드 링크 추가
                st.markdown(get_image_download_link(fig, '업체별_AS_건수.png', '업체별 AS 건수 다운로드'), unsafe_allow_html=True)
            else:
                st.warning("업체명 데이터가 없습니다.")
        
        with col2:
            if '현장명' in df.columns:
                st.subheader("현장별 AS 건수 Top 15")
                site_counts = df['현장명'].value_counts().nlargest(15)
                
                # 업체별, 현장별 그래프 크기 맞춤 - 정확히 동일한 크기와 스타일로 설정
                fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                sns.barplot(x=site_counts.values, y=site_counts.index, ax=ax, palette=f"{current_theme}")
                
                for index, value in enumerate(site_counts.values):
                    ax.text(value + 1, index, str(value), va='center')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                
                # 다운로드 링크 추가
                st.markdown(get_image_download_link(fig, '현장별_AS_건수.png', '현장별 AS 건수 다운로드'), unsafe_allow_html=True)
            else:
                st.warning("현장명 데이터가 없습니다.")
                
        with col3:
            # 지역별 빈도 분석
            st.subheader("지역별 AS 건수")
            if 'ADDR' in df.columns and '지역' in df.columns:
                df_clean = df.dropna(subset=['ADDR']).copy()
                
                region_counts = df_clean['지역'].value_counts()
                region_counts = region_counts.dropna()
                
                # 최소 빈도수 처리 및 상위 15개 표시
                others_count = region_counts[region_counts < 3].sum()
                region_counts = region_counts[region_counts >= 3]
                if others_count > 0:
                    region_counts['기타'] = others_count
                
                region_counts = region_counts.sort_values(ascending=False).nlargest(20)
                
                # 시각화
                fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                blue_palette = sns.color_palette(f"{current_theme}", n_colors=len(region_counts))
                
                sns.barplot(x=region_counts.index, y=region_counts.values, ax=ax, palette=blue_palette)
                
                # 막대 위에 텍스트 표시
                for i, v in enumerate(region_counts.values):
                    ax.text(i, v + max(region_counts.values) * 0.02, str(v),
                           ha='center', fontsize=12)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                
                # 다운로드 링크 추가
                st.markdown(get_image_download_link(fig, '지역별_AS_현황.png', '지역별 AS 현황 다운로드'), unsafe_allow_html=True)
            else:
                st.warning("지역 정보가 없습니다.")
                
        st.markdown("---")
        
        # 4. 현장 지도 (전체 너비로 표시) - 경도/위도 컬럼이 있는 경우에만 표시
        if all(col in df.columns for col in ['경도', '위도']):
            st.subheader("현장 지도")
            
            # NA 값 제거 - 좌표가 없는 데이터는 제외
            df_map = df.dropna(subset=['경도', '위도']).copy()
            
            if len(df_map) > 0:
                st.write(f"총 {len(df):,}건 중 {len(df_map):,}건의 위치 데이터가 있습니다.")
                
                # 중심 좌표 계산
                center_lat = df_map['위도'].mean()
                center_lon = df_map['경도'].mean()
                
                # 지도 생성
                m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
                
                # 마커 클러스터링 적용
                from folium.plugins import MarkerCluster
                marker_cluster = MarkerCluster().add_to(m)
                
                # 현재 테마에 맞는 색상 선택
                if current_theme == "Blues":
                    marker_color = 'blue'
                elif current_theme == "Greens":
                    marker_color = 'green'
                elif current_theme == "Oranges":
                    marker_color = 'orange'
                elif current_theme == "Purples":
                    marker_color = 'purple'
                else:
                    marker_color = 'red'
                
                # 모든 마커 추가
                for idx, row in df_map.iterrows():
                    # 위도/경도가 모두 숫자인지 확인 (추가 검증)
                    if pd.notna(row['위도']) and pd.notna(row['경도']):
                        try:
                            lat = float(row['위도'])
                            lon = float(row['경도'])
                            
                            # 유효한 범위인지 확인 (위도: -90~90, 경도: -180~180)
                            if -90 <= lat <= 90 and -180 <= lon <= 180:
                                # 마커 팝업 텍스트 생성
                                popup_text = f"현장명: {row.get('현장명', '정보 없음')}<br>"
                                if 'ADDR' in df_map.columns and pd.notna(row['ADDR']):
                                    popup_text += f"주소: {row['ADDR']}<br>"
                                if '정비일자' in df_map.columns and pd.notna(row['정비일자']):
                                    popup_text += f"수리일: {row['정비일자']}<br>"
                                if '정비대상' in df_map.columns and pd.notna(row['정비대상']):
                                    popup_text += f"증상: {row['정비대상']}<br>"
                                if '정비작업' in df_map.columns and pd.notna(row['정비작업']):
                                    popup_text += f"상세: {row['정비작업']}<br>"
                                
                                folium.CircleMarker(
                                    location=[lat, lon],
                                    radius=4,
                                    popup=folium.Popup(popup_text, max_width=300),
                                    fill=True,
                                    color=marker_color,
                                    fill_color=marker_color,
                                    fill_opacity=1.0,
                                    weight=1,
                                    opacity=1.0
                                ).add_to(marker_cluster)
                        except (ValueError, TypeError):
                            # 좌표 변환에 실패한 경우 무시
                            continue
                
                # 타일 레이어 추가
                folium.TileLayer(
                    'OpenStreetMap',
                    attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                ).add_to(m)
                
                # 레이어 컨트롤 추가
                folium.LayerControl().add_to(m)
                
                # 지도 표시 (전체 너비로 표시)
                folium_static(m, width=1200, height=600)
            else:
                st.warning("지도에 표시할 좌표 데이터가 없습니다.")
    
    elif menu == "고장 유형 분석":
        st.title("고장 유형 분석")

        # 필요한 컬럼 존재 여부 확인
        if all(col in df.columns for col in ['작업유형', '정비대상', '정비작업', '고장유형']):
            
            # 탭 구조 정의
            category_tabs = {
                "작업유형": "작업유형",
                "정비대상": "정비대상",
                "정비작업": "정비작업"
            }

            tabs = st.tabs(list(category_tabs.keys()))

            for tab, colname in zip(tabs, category_tabs.values()):
                with tab:
                    st.subheader(f"{colname}")
                    category_counts = df[colname].value_counts().head(15)
                    category_values = convert_to_str_list(df[colname].unique())
                    selected_category = st.selectbox(f"{colname} 선택", ["전체"] + sorted(category_values), key=f"sel_{colname}")

                    if selected_category != "전체":
                        filtered_df = df[df[colname].astype(str) == selected_category]
                    else:
                        filtered_df = df

                    top_faults = filtered_df['고장유형'].value_counts().nlargest(15).index
                    df_filtered = filtered_df[filtered_df['고장유형'].isin(top_faults)]
                    top_combos = df_filtered['브랜드_모델'].value_counts().nlargest(15).index
                    df_filtered = df_filtered[df_filtered['브랜드_모델'].isin(top_combos)]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"**{colname} 분포**")
                        fig1, ax1 = create_figure_with_korean(figsize=(10, 8), dpi=300)
                        sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax1, palette=f"{current_theme}_r")
                        plt.xticks(rotation=45, ha='right')
                        for i, v in enumerate(category_counts.values):
                            ax1.text(i, v + max(category_counts.values) * 0.01, str(v), ha='center', fontsize=12)
                        plt.tight_layout()
                        st.pyplot(fig1, use_container_width=True)
                        st.markdown(get_image_download_link(fig1, f'고장유형_{colname}_분포.png', f'{colname} 분포 다운로드'), unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"**{colname}별 비율**")

                        # 5% 미만은 기타로 그룹화
                        category_counts_ratio = category_counts / category_counts.sum()
                        small_categories = category_counts_ratio[category_counts_ratio < 0.05]
                        if not small_categories.empty:
                            others_sum = small_categories.sum() * category_counts.sum()
                            category_counts = category_counts[category_counts_ratio >= 0.05]
                            category_counts['기타'] = int(others_sum)

                        fig2, ax2 = create_figure_with_korean(figsize=(8, 8), dpi=300)
                        category_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2,
                                             colors=sns.color_palette(current_theme, n_colors=len(category_counts)))
                        ax2.set_ylabel('')
                        plt.tight_layout()
                        st.pyplot(fig2, use_container_width=True)
                        st.markdown(get_image_download_link(fig2, f'고장유형_{colname}_비율.png', f'{colname} 비율 다운로드'), unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"**모델에 따른 고장유형 증상**")
                        try:
                            pivot_df = df_filtered.pivot_table(
                                index='고장유형',
                                columns='브랜드_모델',
                                aggfunc='size',
                                fill_value=0
                            )
                            fig3, ax3 = create_figure_with_korean(figsize=(14, 10), dpi=300)
                            sns.heatmap(pivot_df, cmap=current_theme, annot=True, fmt='d', linewidths=0.5, ax=ax3)
                            plt.xticks(rotation=90)
                            plt.yticks(rotation=0)
                            plt.tight_layout()
                            st.pyplot(fig3, use_container_width=True)
                            st.markdown(get_image_download_link(fig3, f'고장유형_{colname}_히트맵.png', f'{colname} 히트맵 다운로드'), unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"히트맵 생성 중 오류가 발생했습니다: {e}")
                            st.info("선택한 필터에 맞는 데이터가 충분하지 않을 수 있습니다.")

            # 자재내역 분석 섹션 추가 - 자재내역 컬럼이 있는 경우만 표시
            st.subheader("모델 타입 분석")
            
            # 탭으로 분석 항목 구분
            tabs = st.tabs(["연료", "운전방식", "적재용량", "마스트"])
            
            # 연료별 분석
            with tabs[0]:
                if '연료' in df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 연료별 AS 건수
                        st.subheader("연료별 AS 건수")
                        fuel_type_counts = df['연료'].value_counts().dropna()
                        
                        if len(fuel_type_counts) > 0:
                            fig, ax = create_figure_with_korean(figsize=(10, 9), dpi=300)
                            sns.barplot(x=fuel_type_counts.index, y=fuel_type_counts.values, ax=ax, palette=f"{current_theme}_r")
                            
                            # 막대 위에 텍스트 표시
                            for i, v in enumerate(fuel_type_counts.values):
                                ax.text(i, v + max(fuel_type_counts.values) * 0.01, str(v),
                                      ha='center', fontsize=12)
                                
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            
                            # 다운로드 링크 추가
                            st.markdown(get_image_download_link(fig, '연료별_AS_건수.png', '연료별 AS 건수 다운로드'), unsafe_allow_html=True)
                        else:
                            st.warning("연료 데이터가 없습니다.")
                    
                    with col2:
                        # 연료별 고장유형 Top 10
                        st.subheader("연료별 고장유형")
                        
                        # 연료 선택
                        fuel_types = ["전체"] + df['연료'].value_counts().index.tolist()
                        selected_fuel = st.selectbox("연료", fuel_types)
                        
                        if selected_fuel != "전체":
                            filtered_df_fuel = df[df['연료'] == selected_fuel]
                        else:
                            filtered_df_fuel = df
                            
                        if '고장유형' in filtered_df_fuel.columns:
                            top_faults_by_fuel = filtered_df_fuel['고장유형'].value_counts().head(10)
                            
                            fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                            sns.barplot(x=top_faults_by_fuel.values, y=top_faults_by_fuel.index, ax=ax, palette=f"{current_theme}_r")
                            
                            # 막대 위에 텍스트 표시
                            for i, v in enumerate(top_faults_by_fuel.values):
                                ax.text(v + max(top_faults_by_fuel.values) * 0.002, i, str(v),
                                      va='center', fontsize=12)
                                
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            
                            # 다운로드 링크 추가
                            st.markdown(get_image_download_link(fig, f'{selected_fuel}_고장유형_TOP10.png', f'{selected_fuel} 고장유형 TOP10 다운로드'), unsafe_allow_html=True)
                        else:
                            st.warning("고장유형 데이터가 없습니다.")
                else:
                    st.warning("연료 데이터가 없습니다.")
            
            # 운전방식별 분석
            with tabs[1]:
                if '운전방식' in df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 운전방식별 AS 건수
                        st.subheader("운전방식별 AS 건수")
                        driving_type_counts = df['운전방식'].value_counts().head(15)
                        
                        if len(driving_type_counts) > 0:
                            fig, ax = create_figure_with_korean(figsize=(10, 9), dpi=300)
                            sns.barplot(x=driving_type_counts.index, y=driving_type_counts.values, ax=ax, palette=f"{current_theme}_r")
                            
                            # 막대 위에 텍스트 표시
                            for i, v in enumerate(driving_type_counts.values):
                                ax.text(i, v + max(driving_type_counts.values) * 0.01, str(v),
                                      ha='center', fontsize=12)
                                
                            plt.tight_layout()
                            plt.xticks(rotation=45)
                            st.pyplot(fig, use_container_width=True)
                            
                            # 다운로드 링크 추가
                            st.markdown(get_image_download_link(fig, '운전방식별_AS_건수.png', '운전방식별 AS 건수 다운로드'), unsafe_allow_html=True)
                        else:
                            st.warning("운전방식 데이터가 없습니다.")
                    
                    with col2:
                        # 운전방식별 고장유형 Top 10
                        st.subheader("운전방식별 고장유형")
                        
                        # 운전방식 선택
                        driving_types = ["전체"] + df['운전방식'].value_counts().index.tolist()
                        selected_driving = st.selectbox("운전방식", driving_types)
                        
                        if selected_driving != "전체":
                            filtered_df_driving = df[df['운전방식'] == selected_driving]
                        else:
                            filtered_df_driving = df
                            
                        if '고장유형' in filtered_df_driving.columns:
                            top_faults_by_driving = filtered_df_driving['고장유형'].value_counts().head(10)
                            
                            fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                            sns.barplot(x=top_faults_by_driving.values, y=top_faults_by_driving.index, ax=ax, palette=f"{current_theme}_r")
                            
                            # 막대 위에 텍스트 표시
                            for i, v in enumerate(top_faults_by_driving.values):
                                ax.text(v + max(top_faults_by_driving.values) * 0.002, i, str(v),
                                      va='center', fontsize=12)
                                
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            
                            # 다운로드 링크 추가
                            st.markdown(get_image_download_link(fig, f'{selected_driving}_고장유형_TOP10.png', f'{selected_driving} 고장유형 TOP10 다운로드'), unsafe_allow_html=True)
                        else:
                            st.warning("고장유형 데이터가 없습니다.")
                else:
                    st.warning("운전방식 데이터가 없습니다.")

            # 적재용량별 분석
            with tabs[2]:
                if '적재용량' in df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 적재용량별 AS 건수
                        st.subheader("적재용량별 AS 건수")
                        load_capacity_counts = df['적재용량'].value_counts().head(15)
                        
                        if len(load_capacity_counts) > 0:
                            fig, ax = create_figure_with_korean(figsize=(10, 9), dpi=300)
                            sns.barplot(x=load_capacity_counts.index, y=load_capacity_counts.values, ax=ax, palette=f"{current_theme}_r")
                            
                            # 막대 위에 텍스트 표시
                            for i, v in enumerate(load_capacity_counts.values):
                                ax.text(i, v + max(load_capacity_counts.values) * 0.02, str(v),
                                      ha='center', fontsize=12)
                                
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            
                            # 다운로드 링크 추가
                            st.markdown(get_image_download_link(fig, '적재용량별_AS_건수.png', '적재용량별 AS 건수 다운로드'), unsafe_allow_html=True)
                        else:
                            st.warning("적재용량 데이터가 없습니다.")
                    
                    with col2:
                        # 적재용량별 고장유형 Top 10
                        st.subheader("적재용량별 고장유형")
                        
                        # 적재용량 선택
                        load_capacities = ["전체"] + df['적재용량'].value_counts().index.tolist()
                        selected_capacity = st.selectbox("적재용량", load_capacities)
                        
                        if selected_capacity != "전체":
                            filtered_df_capacity = df[df['적재용량'] == selected_capacity]
                        else:
                            filtered_df_capacity = df
                            
                        if '고장유형' in filtered_df_capacity.columns:
                            top_faults_by_capacity = filtered_df_capacity['고장유형'].value_counts().head(10)
                            
                            fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                            sns.barplot(x=top_faults_by_capacity.values, y=top_faults_by_capacity.index, ax=ax, palette=f"{current_theme}_r")
                            
                            # 막대 위에 텍스트 표시
                            for i, v in enumerate(top_faults_by_capacity.values):
                                ax.text(v + max(top_faults_by_capacity.values) * 0.002, i, str(v),
                                      va='center', fontsize=12)
                                
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            
                            # 다운로드 링크 추가
                            st.markdown(get_image_download_link(fig, f'{selected_capacity}_고장유형_TOP10.png', f'{selected_capacity} 고장유형 TOP10 다운로드'), unsafe_allow_html=True)
                        else:
                            st.warning("고장유형 데이터가 없습니다.")
                else:
                    st.warning("적재용량 데이터가 없습니다.")

            # 마스트별 분석
            with tabs[3]:
                if '마스트' in df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 마스트별 AS 건수
                        st.subheader("마스트별 AS 건수")
                        mast_type_counts = df['마스트'].value_counts().head(15)
                        
                        if len(mast_type_counts) > 0:
                            fig, ax = create_figure_with_korean(figsize=(10, 9), dpi=300)
                            sns.barplot(x=mast_type_counts.index, y=mast_type_counts.values, ax=ax, palette=f"{current_theme}_r")
                            
                            # 막대 위에 텍스트 표시
                            for i, v in enumerate(mast_type_counts.values):
                                ax.text(i, v + max(mast_type_counts.values) * 0.02, str(v),
                                      ha='center', fontsize=12)
                                
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            
                            # 다운로드 링크 추가
                            st.markdown(get_image_download_link(fig, '마스트별_AS_건수.png', '마스트별 AS 건수 다운로드'), unsafe_allow_html=True)
                        else:
                            st.warning("마스트 데이터가 없습니다.")
                    
                    with col2:
                        # 마스트별 고장유형 Top 10
                        st.subheader("마스트별 고장유형")
                        
                        # 마스트 선택
                        mast_types = ["전체"] + df['마스트형'].value_counts().index.tolist()
                        selected_mast = st.selectbox("마스트", mast_types)
                        
                        if selected_mast != "전체":
                            filtered_df_mast = df[df['마스트형'] == selected_mast]
                        else:
                            filtered_df_mast = df
                            
                        if '고장유형' in filtered_df_mast.columns:
                            top_faults_by_mast = filtered_df_mast['고장유형'].value_counts().head(10)
                            
                            fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                            sns.barplot(x=top_faults_by_mast.values, y=top_faults_by_mast.index, ax=ax, palette=f"{current_theme}_r")
                            
                            # 막대 위에 텍스트 표시
                            for i, v in enumerate(top_faults_by_mast.values):
                                ax.text(v + max(top_faults_by_mast.values) * 0.002, i, str(v),
                                      va='center', fontsize=12)
                                
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            
                            # 다운로드 링크 추가
                            st.markdown(get_image_download_link(fig, f'{selected_mast}_고장유형_TOP10.png', f'{selected_mast} 고장유형 TOP10 다운로드'), unsafe_allow_html=True)
                        else:
                            st.warning("고장유형 데이터가 없습니다.")
                else:
                    st.warning("마스트 데이터가 없습니다.")
                    
            # 상위 고장 유형 리스트
            st.subheader("상위 고장 유형")
            top_40_faults = filtered_df['고장유형'].value_counts().nlargest(40)
            fault_df = pd.DataFrame({
                '고장유형': top_40_faults.index,
                '건수': top_40_faults.values
            })
            st.dataframe(fault_df)
        else:
            st.warning("고장 유형 분석에 필요한 컬럼(작업유형, 정비대상, 정비작업)이 데이터에 없습니다.")

    elif menu == "브랜드/모델 분석":
        st.title("브랜드 및 모델 분석")
        
        # 그래프 행 1: 브랜드 분포와 모델별 분석 (높이 제한)
        col1, col2 = st.columns(2)
        
        with col1:
            # 브랜드 분포
            st.subheader("브랜드 분포")
            brand_counts = df['브랜드'].value_counts()
            
            # 3% 이하는 '기타'로 통합
            brand_counts = group_small_categories(brand_counts, threshold=0.03)
            brand_counts = brand_counts.nlargest(15) # 상위 15개만 표시
            
            # 높이를 제한하여 더 작게 표시
            fig, ax = create_figure_with_korean(figsize=(8, 6), dpi=300)
            sns.barplot(x=brand_counts.index, y=brand_counts.values, ax=ax, palette=f"{current_theme}_r")
            
            # 막대 위에 텍스트 표시
            for i, v in enumerate(brand_counts.values):
                ax.text(i, v + max(brand_counts.values) * 0.01, str(int(v)),
                       ha='center', fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            # 다운로드 링크 추가
            st.markdown(get_image_download_link(fig, '브랜드_분포.png', '브랜드 분포 다운로드'), unsafe_allow_html=True)
            
        with col2:
            # 브랜드별 AS 비율 파이 차트
            st.subheader("브랜드별 AS 비율")
            
            # 3% 이하는 '기타'로 통합
            brand_counts_pie = df['브랜드'].value_counts()
            brand_counts_pie = group_small_categories(brand_counts_pie, threshold=0.03)
            
            # 파이 차트 크기 조정 - 더 작게
            fig, ax = create_figure_with_korean(figsize=(6, 6), dpi=300)
            brand_counts_pie.plot(kind='pie', autopct='%1.1f%%', ax=ax, 
                             colors=sns.color_palette(current_theme, n_colors=len(brand_counts_pie)))
            ax.set_ylabel('')
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            
            # 다운로드 링크 추가
            st.markdown(get_image_download_link(fig, '브랜드_AS_비율.png', '브랜드 AS 비율 다운로드'), unsafe_allow_html=True)
        
        # 브랜드별 모델 분석
        st.subheader("브랜드별 모델 분석")
        
        # 브랜드 목록 정의 - 특정 브랜드를 우선순위로 배치하고 나머지는 빈도수 순으로
        priority_brands = ['도요타', '두산', '현대', '클라크']

        # 전체 브랜드 빈도 계산
        brand_counts = df['브랜드'].value_counts()

        # 우선순위 브랜드 목록 생성
        brand_list = [brand for brand in priority_brands if brand in brand_counts.index]

        # 나머지 브랜드: 우선순위와 '기타'를 제외하고 빈도수 순으로 정렬
        other_brands = [brand for brand in brand_counts.index
                        if brand not in priority_brands and brand != '기타']
        brand_list.extend(other_brands)

        # '기타'가 있으면 마지막에 추가
        if '기타' in brand_counts.index:
            brand_list.append('기타')

        # "전체"를 맨 앞에 추가
        brand_list = ["전체"] + brand_list

        # 선택 UI
        selected_brand = st.selectbox("브랜드 선택", brand_list)

        # 모델별 AS 시각화
        col1, col2 = st.columns(2)

        with col1:
            # 브랜드별 필터링
            if selected_brand != "전체":
                brand_df = df[df['브랜드'] == selected_brand]
            else:
                brand_df = df

            # 모델별 AS 건수 계산 (빈도수 순 정렬 유지)
            model_counts = brand_df['모델명'].value_counts().head(15)

            st.subheader(f"{selected_brand if selected_brand != '전체' else '전체'} 모델별 AS 건수")

            # 그래프 생성
            fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
            sns.barplot(x=model_counts.values, y=model_counts.index, ax=ax, palette=f"{current_theme}_r")

            # 값 텍스트로 표시
            for i, v in enumerate(model_counts.values):
                ax.text(v + max(model_counts.values) * 0.0025, i, str(v),
                        va='center', fontsize=12)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            # 이미지 다운로드 링크
            st.markdown(get_image_download_link(fig, f'{selected_brand}_모델별_AS_건수.png',
                                                f'{selected_brand} 모델별 AS 건수 다운로드'),
                        unsafe_allow_html=True)
        
        with col2:
            # 브랜드별 고장 유형 분석
            if '고장유형' in brand_df.columns:
                st.subheader(f"{selected_brand if selected_brand != '전체' else '전체'} 브랜드 고장 유형 분석")
                brand_faults = brand_df['고장유형'].value_counts().head(15)
                
                # 그래프 크기 맞춤
                fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                sns.barplot(x=brand_faults.values, y=brand_faults.index, ax=ax, palette=f"{current_theme}_r")
                
                # 막대 옆에 텍스트 표시
                for i, v in enumerate(brand_faults.values):
                    ax.text(v + max(brand_faults.values) * 0.0025, i, str(v),
                           va='center', fontsize=12)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                
                # 다운로드 링크 추가
                st.markdown(get_image_download_link(fig, f'{selected_brand}_고장유형_분석.png', f'{selected_brand} 고장유형 분석 다운로드'), unsafe_allow_html=True)

        # 제조년도별 분석 추가
        if '제조년도' in df.columns:
            # 브랜드별 필터링
            if selected_brand != "전체":
                brand_df = df[df['브랜드'] == selected_brand]
            else:
                brand_df = df

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"{selected_brand if selected_brand != '전체' else '전체'} 제조년도 건수")
                # 제조년도별 AS 건수
                year_counts = brand_df['제조년도'].dropna().astype(int).value_counts().sort_index()

                if len(year_counts) > 0:
                    fig, ax = create_figure_with_korean(figsize=(10, 6), dpi=300)
                    sns.barplot(x=year_counts.index.astype(str), y=year_counts.values, ax=ax, palette=f"{current_theme}_r")

                    for i, v in enumerate(year_counts.values):
                        ax.text(i, v + max(year_counts.values) * 0.02, str(v),
                                ha='center', fontsize=12)

                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

                    st.markdown(get_image_download_link(fig, '제조년도별_AS_건수.png', '제조년도별 AS 건수 다운로드'), unsafe_allow_html=True)
                else:
                    st.warning("제조년도 데이터가 없습니다.")

            with col2:
                st.subheader(f"{selected_brand if selected_brand != '전체' else '전체'} 제조년도별 처리일수")

                if 'AS처리일수' in df.columns:
                    df_clean = brand_df.dropna(subset=['제조년도', 'AS처리일수'])

                    if len(df_clean) > 0:
                        df_clean['제조년도'] = df_clean['제조년도'].astype(int)
                        year_avg_days = df_clean.groupby('제조년도')['AS처리일수'].mean().sort_index()

                        fig, ax = create_figure_with_korean(figsize=(10, 6), dpi=300)
                        sns.barplot(x=year_avg_days.index.astype(str), y=year_avg_days.values, ax=ax, palette=f"{current_theme}")

                        for i, v in enumerate(year_avg_days.values):
                            ax.text(i, v + max(year_avg_days.values) * 0.02, f"{v:.1f}",
                                    ha='center', fontsize=12)

                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)

                        st.markdown(get_image_download_link(fig, '제조년도별_평균처리일수.png', '제조년도별 평균 처리일수 다운로드'), unsafe_allow_html=True)
                    else:
                        st.warning("제조년도 및 AS처리일수 데이터가 충분하지 않습니다.")
                else:
                    st.warning("AS처리일수 데이터가 없습니다.")


    elif menu == "정비내용 분석":
        st.title("정비내용 분석")
        
        # 텍스트 데이터 확인
        if '정비내용' in df.columns:
            # 정비내용 데이터 준비
            text_data = ' '.join(df['정비내용'].dropna().astype(str))
            
            if not text_data:
                st.warning("정비내용 데이터가 없습니다.")
            else:
                # 그래프 행 1: 전체 워드클라우드와 분류별 워드클라우드
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("전체 정비내용 워드클라우드")
                    
                    # 불용어 목록 업데이트
                    stopwords = ["및", "있음", "없음", "함", "을", "후", "함", "접수", "취소", "확인", "위해", "통해", "오류", "완료", "작업", "실시", "진행", "수리"]
                    
                    try:
                        # 워드클라우드 생성 (font_path 사용하지 않음)
                        wordcloud = WordCloud(
                            width=1200, 
                            height=800,
                            background_color='white',
                            font_path=font_path,  
                            colormap=current_theme,
                            max_words=100,
                            stopwords=set(stopwords),
                            min_font_size=10,
                            max_font_size=150,
                            random_state=42
                        ).generate(text_data)
                        
                        # 워드클라우드 시각화
                        fig, ax = create_figure_with_korean(figsize=(10, 10), dpi=300)
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        plt.tight_layout()
                        
                        st.pyplot(fig, use_container_width=True)
                        
                        # 다운로드 링크 추가
                        st.markdown(get_image_download_link(fig, '정비내용_워드클라우드.png', '정비내용 워드클라우드 다운로드'), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"워드클라우드 생성 중 오류가 발생했습니다: {e}")
                        if "font_path" in str(e).lower():
                            st.info("한글 폰트 경로를 확인해주세요.")
                
                with col2:
                    # 주요 단어 표시
                    word_freq = wordcloud.words_
                    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:30])
                    
                    st.subheader("주요 단어 Top 30")
                    word_df = pd.DataFrame({
                        '단어': list(top_words.keys()),
                        '가중치': list(top_words.values())
                    })
                    
                    fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                    sns.barplot(x=word_df['가중치'].head(20), y=word_df['단어'].head(20), ax=ax, palette=f"{current_theme}_r")
                    plt.tight_layout()
                    
                    st.pyplot(fig, use_container_width=True)
                    
                    # 다운로드 링크 추가
                    st.markdown(get_image_download_link(fig, '주요단어_TOP30.png', '주요단어 TOP30 다운로드'), unsafe_allow_html=True)
                
                # 분류별 정비내용 워드클라우드
                st.subheader("분류별 정비내용 워드클라우드")
                
                if all(col in df.columns for col in ['작업유형', '정비대상', '정비작업', '정비내용']):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # 문자열 변환으로 정렬 오류 방지
                        categories = ["전체"] + sorted(convert_to_str_list(df['작업유형'].dropna().unique()))
                        selected_category = st.selectbox("작업유형", categories)
                    
                    # 선택된 작업유형에 따라 데이터 필터링
                    if selected_category != "전체":
                        filtered_df = df[df['작업유형'].astype(str) == selected_category]
                        
                        with col2:
                            subcategories = ["전체"] + sorted(convert_to_str_list(filtered_df['정비대상'].dropna().unique()))
                            selected_subcategory = st.selectbox("정비대상", subcategories)
                        
                        # 선택된 정비대상에 따라 추가 필터링
                        if selected_subcategory != "전체":
                            filtered_df = filtered_df[filtered_df['정비대상'].astype(str) == selected_subcategory]
                            
                            with col3:
                                detailed_categories = ["전체"] + sorted(convert_to_str_list(filtered_df['정비작업'].dropna().unique()))
                                selected_detailed = st.selectbox("정비작업", detailed_categories)
                            
                            # 선택된 정비작업에 따라 최종 필터링
                            if selected_detailed != "전체":
                                filtered_df = filtered_df[filtered_df['정비작업'].astype(str) == selected_detailed]
                        else:
                            selected_detailed = "전체"
                            with col3:
                                st.selectbox("정비작업", ["전체"])
                    else:
                        filtered_df = df
                        selected_subcategory = "전체"
                        selected_detailed = "전체"
                        
                        with col2:
                            st.selectbox("정비대상", ["전체"])
                        
                        with col3:
                            st.selectbox("정비작업", ["전체"])
                    
                    # 필터링된 정비내용 결합
                    filtered_text = ' '.join(filtered_df['정비내용'].dropna().astype(str))
                    
                    if not filtered_text:
                        st.warning("선택한 분류에 대한 정비내용 데이터가 없습니다.")
                    else:
                        st.write(f"선택: {selected_category} > {selected_subcategory} > {selected_detailed}")
                        st.write(f"선택된 AS 건수: {len(filtered_df)}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            try:
                                # 워드클라우드 생성 (font_path 사용하지 않음)
                                wordcloud = WordCloud(
                                    width=1200, 
                                    height=800,
                                    background_color='white',
                                    font_path=font_path,  # 다운로드한 폰트 경로 사용
                                    colormap=current_theme,
                                    max_words=100,
                                    stopwords=set(stopwords),
                                    min_font_size=10,
                                    max_font_size=150,
                                    random_state=42
                                ).generate(filtered_text)
                                
                                # 워드클라우드 시각화
                                fig, ax = create_figure_with_korean(figsize=(10, 10), dpi=300)
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                plt.tight_layout()
                                
                                st.pyplot(fig, use_container_width=True)
                                
                                # 다운로드 링크 추가
                                st.markdown(get_image_download_link(fig, f'{selected_category}_{selected_subcategory}_{selected_detailed}_워드클라우드.png', 
                                           '분류별 워드클라우드 다운로드'), unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"워드클라우드 생성 중 오류가 발생했습니다: {e}")
                        
                        with col2:
                            # 주요 단어 표시
                            word_freq = wordcloud.words_
                            top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:30])
                            
                            word_df = pd.DataFrame({
                                '단어': list(top_words.keys()),
                                '가중치': list(top_words.values())
                            })
                            
                            fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                            sns.barplot(x=word_df['가중치'].head(20), y=word_df['단어'].head(20), ax=ax, palette=f"{current_theme}_r")
                            plt.tight_layout()
                            
                            st.pyplot(fig, use_container_width=True)
                            
                            # 다운로드 링크 추가
                            st.markdown(get_image_download_link(fig, f'{selected_category}_{selected_subcategory}_{selected_detailed}_주요단어.png', 
                                       '분류별 주요단어 다운로드'), unsafe_allow_html=True)
                else:
                    st.warning("분류별 분석에 필요한 컬럼이 데이터에 없습니다.")
        else:
            st.warning("정비내용 컬럼이 데이터에 없습니다.")
    
    elif menu == "고장 예측":
        st.title("고장 예측")

        @st.cache_resource
        def prepare_prediction_model(df):
            try:
                # 필수 컬럼 체크
                required_cols = ['브랜드', '모델명', '작업유형', '정비대상', '정비작업', 'AS처리일수', '제조년도']
                if not all(col in df.columns for col in required_cols):
                    return [None] * 10

                model_df = df.dropna(subset=required_cols[:-1]).copy()  # 제조년도는 후처리로

                if len(model_df) < 100:
                    return [None] * 10

                # 범주형 인코딩
                le_brand = LabelEncoder()
                le_model = LabelEncoder()
                le_category = LabelEncoder()
                le_subcategory = LabelEncoder()
                le_detail = LabelEncoder()

                model_df['브랜드_인코딩'] = le_brand.fit_transform(model_df['브랜드'])
                model_df['모델_인코딩'] = le_model.fit_transform(model_df['모델명'])
                model_df['작업유형_인코딩'] = le_category.fit_transform(model_df['작업유형'])
                model_df['정비대상_인코딩'] = le_subcategory.fit_transform(model_df['정비대상'])
                model_df['정비작업_인코딩'] = le_detail.fit_transform(model_df['정비작업'])

                # 제조년도 처리
                model_df['제조년도_정수'] = pd.to_numeric(model_df['제조년도'], errors='coerce')
                if model_df['제조년도_정수'].isna().any():
                    mode_year = model_df['제조년도_정수'].mode().iloc[0]
                    model_df['제조년도_정수'] = model_df['제조년도_정수'].fillna(mode_year)

                def year_to_range(year):
                    if year <= 2005:
                        return "2005이하"
                    elif year <= 2010:
                        return "2006-2010"
                    elif year <= 2015:
                        return "2011-2015"
                    elif year <= 2020:
                        return "2016-2020"
                    else:
                        return "2021-2025"

                model_df['제조년도_구간'] = model_df['제조년도_정수'].apply(year_to_range)
                le_year_range = LabelEncoder()
                model_df['제조년도_구간_인코딩'] = le_year_range.fit_transform(model_df['제조년도_구간'])

                # 피처
                features = ['브랜드_인코딩', '모델_인코딩', '작업유형_인코딩', '정비대상_인코딩', '정비작업_인코딩', 'AS처리일수', '제조년도_구간_인코딩']

                # 타겟
                model_df['재정비간격_타겟'] = model_df['재정비간격'].fillna(365).clip(0, 365)
                X = model_df[features]
                y_interval = model_df['재정비간격_타겟']

                # 회귀 모델 학습
                X_train, X_test, y_train, y_test = train_test_split(X, y_interval, test_size=0.2, random_state=42)
                rf_interval_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_interval_model.fit(X_train, y_train)

                # 분류 모델들 학습
                category_model = RandomForestClassifier(n_estimators=100, random_state=42)
                category_model.fit(X, model_df['작업유형_인코딩'])

                subcategory_model = RandomForestClassifier(n_estimators=100, random_state=42)
                subcategory_model.fit(X, model_df['정비대상_인코딩'])

                detail_model = RandomForestClassifier(n_estimators=100, random_state=42)
                detail_model.fit(X, model_df['정비작업_인코딩'])

                return (
                    rf_interval_model, category_model, subcategory_model, detail_model,
                    le_brand, le_model, le_category, le_subcategory, le_detail, le_year_range
                )

            except Exception as e:
                st.error(f"모델 준비 중 오류 발생: {e}")
                return [None] * 10
    
        # 모델 준비 (백그라운드에서 실행)
        interval_model, category_model, subcategory_model, detail_model, le_brand, le_model, le_category, le_subcategory, le_detail, le_year_range = prepare_prediction_model(df)
        
        if interval_model is not None:
            st.info("다음 고장 시기 예측과 확률이 높은 고장 유형을 예측합니다.")
            
        col1, col2, col3, col4, col5 = st.columns(5)

        # 브랜드 목록 정의 - 특정 브랜드를 우선순위로 배치하고 나머지는 알파벳 순서로
        priority_brands = ['도요타', '두산', '현대', '클라크']
        
        # 우선순위 브랜드 목록 생성
        brand_list = [brand for brand in priority_brands if brand in df['브랜드'].unique()]
        
        # 나머지 브랜드 추가 (우선순위와 '기타'를 제외하고 정렬)
        other_brands = sorted([brand for brand in df['브랜드'].unique() 
                              if brand not in priority_brands and brand != '기타'])
        brand_list.extend(other_brands)
        
        # '기타'가 있으면 마지막에 추가
        if '기타' in df['브랜드'].unique():
            brand_list.append('기타')
            
        with col1:
            selected_brand = st.selectbox("브랜드(필수)", brand_list)

        with col2:
            brand_models = df[df['브랜드'] == selected_brand]['모델명'].unique()
            selected_model = st.selectbox("모델(필수)", brand_models)

        # 브랜드/모델 선택 이후 필터링
        filtered_df = df[(df['브랜드'] == selected_brand) & (df['모델명'] == selected_model)]

        if not filtered_df.empty:
            with col3:
                existing_ids = filtered_df['관리번호'].dropna().unique()
                selected_id = st.selectbox("관리번호(선택)", ["전체"] + list(existing_ids), index=0)

                if selected_id != "전체":
                    filtered_df = filtered_df[filtered_df['관리번호'] == selected_id]
            
            with col4:
                id_placeholder = f"예: {existing_ids[0]}" if len(existing_ids) > 0 else ""
                input_id = st.text_input("관리번호(직접 입력)", placeholder=id_placeholder).strip()
                # 선택된 ID 또는 입력된 ID 사용
                final_id = selected_id if selected_id else input_id
    
            with col5:
                if '제조년도' in filtered_df.columns:
                    years = filtered_df['제조년도'].dropna().astype(int)

                    def year_to_range(year):
                        if year <= 2005: return "2005년 이하"
                        elif year <= 2010: return "2006-2010"
                        elif year <= 2015: return "2011-2015"
                        elif year <= 2020: return "2016-2020"
                        else: return "2021-2025"

                    year_ranges = sorted(set(year_to_range(y) for y in years))
                    year_ranges = ["전체"] + year_ranges  # "전체" 옵션 추가
                    selected_year_range = st.selectbox("제조년도(선택)", year_ranges, index=0)
                else:
                    selected_year_range = "전체"


            # 브랜드 + 모델 기준 1차 필터링
            filtered_df = df[(df['브랜드'] == selected_brand) & (df['모델명'] == selected_model)]

            # 관리번호 추가 필터링
            if selected_id != "전체":
                filtered_df = filtered_df[filtered_df['관리번호'] == selected_id]

            # 제조년도 범위로 필터링 (정의된 구간 내 값만)
            if selected_year_range != "전체":
                def year_in_range(year):
                    if selected_year_range == "2005년 이하": return year <= 2005
                    elif selected_year_range == "2006-2010": return 2006 <= year <= 2010
                    elif selected_year_range == "2011-2015": return 2011 <= year <= 2015
                    elif selected_year_range == "2016-2020": return 2016 <= year <= 2020
                    elif selected_year_range == "2021-2025": return 2021 <= year <= 2025
                    return False

                filtered_df = filtered_df[filtered_df['제조년도'].dropna().astype(int).apply(year_in_range)]

            if len(filtered_df) > 0:
                latest_record = filtered_df.sort_values('정비일자', ascending=False).iloc[0]

                # 최근 정비 내용 표시
                st.subheader("장비 최근 정비 정보")
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**최근 정비일:** {latest_record['정비일자'].strftime('%Y-%m-%d')}")
                    st.write(f"**고장 유형:** {latest_record['작업유형']} > {latest_record['정비대상']} > {latest_record['정비작업']}")
                    st.write(f"**종류:** {latest_record.get('자재내역', '정보 없음')}")
                    st.write(f"**정비사:** {latest_record.get('정비자', '정보 없음')}")

                with col2:
                    st.write(f"**이전 정비일:** {latest_record.get('최근정비일자', '정보 없음')}")
                    st.write(f"**정비 내용:** {latest_record.get('정비내용', '정보 없음')}")
                    st.write(f"**현장명:** {latest_record.get('현장명', '정보 없음')}")
        
            # 예측 실행
            if st.button("고장 예측 실행"):
                with st.spinner("예측 분석 중..."):
                    try:
                        # 선택한 값을 인코딩
                        brand_code = le_brand.transform([selected_brand])[0]
                        model_code = le_model.transform([selected_model])[0]

                        # 최근 정비 데이터 가져오기
                        latest_data = df[(df['브랜드'] == selected_brand) & (df['모델명'] == selected_model)] \
                            .sort_values('정비일자', ascending=False).iloc[0]

                        category_code = le_category.transform([latest_data['작업유형']])[0]
                        subcat_code = le_subcategory.transform([latest_data['정비대상']])[0]
                        detail_code = le_detail.transform([latest_data['정비작업']])[0]

                        # AS 처리일수 평균
                        avg_repair_days = df[(df['브랜드'] == selected_brand) & (df['모델명'] == selected_model)]['AS처리일수'].mean()
                        if pd.isna(avg_repair_days):
                            avg_repair_days = df['AS처리일수'].mean()

                        # 제조년도 구간 → 인코딩
                        if selected_year_range == "전체":
                            mode_range = df['제조년도'].dropna().astype(int).apply(year_to_range).mode().iloc[0]
                        else:
                            mode_range = selected_year_range
                        year_range_encoded = le_year_range.transform([mode_range])[0]

                        # 예측할 데이터 준비
                        pred_data = np.array([[ 
                            brand_code, model_code, category_code, subcat_code, detail_code, 
                            avg_repair_days, year_range_encoded
                        ]])

                        # 예측 수행
                        predicted_days = interval_model.predict(pred_data)[0]
                        predicted_category_code = category_model.predict(pred_data)[0]
                        predicted_subcategory_code = subcategory_model.predict(pred_data)[0]
                        predicted_detail_code = detail_model.predict(pred_data)[0]

                        predicted_category = le_category.inverse_transform([predicted_category_code])[0]
                        predicted_subcategory = le_subcategory.inverse_transform([predicted_subcategory_code])[0]
                        predicted_detail = le_detail.inverse_transform([predicted_detail_code])[0]

                        # 결과 출력
                        st.success("**예측 분석 완료** ※ 해당 모델에 따른 예측모델 구현")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("다음 정비 시기")
                            prediction_date = datetime.datetime.now() + datetime.timedelta(days=int(predicted_days))
                            st.markdown(f"""
                            **장비 정보**: {selected_brand} {selected_model}  
                            **예상 재정비 기간**: 약 **{int(predicted_days)}일** 후  
                            **예상 고장 날짜**: {prediction_date.strftime('%Y-%m-%d')}
                            """)

                            risk_level = "낮음"
                            risk_color = "green"
                            if predicted_days < 30:
                                risk_level = "매우 높음"
                                risk_color = "red"
                            elif predicted_days < 90:
                                risk_level = "높음"
                                risk_color = "orange"
                            elif predicted_days < 180:
                                risk_level = "중간"
                                risk_color = "yellow"

                            st.markdown(f"<h4 style='color: {risk_color};'>재정비 위험도: {risk_level}</h4>", unsafe_allow_html=True)

                        with col2:
                            st.subheader("고장 유형 예측")
                            st.markdown(f"""
                            **작업유형**: {predicted_category}  
                            **정비대상**: {predicted_subcategory}  
                            **정비작업**: {predicted_detail}
                            """)

                    except Exception as e:
                        st.error(f"예측 중 오류가 발생했습니다: {e}")
                        st.info("선택한 데이터에 대한 학습 정보가 부족할 수 있습니다.")
        else:
            st.warning("""
            예측 모델을 준비할 수 없습니다. 다음 사항을 확인해주세요:
            1. 충분한 데이터(최소 100개 이상의 기록)가 있는지 확인
            2. 필요한 컬럼(브랜드, 모델명, 작업유형, 정비대상, 정비작업, AS처리일수)이 모두 있는지 확인
            3. 재정비 간격 정보가 있는지 확인
            """)

else:
    st.header("산업장비 AS 대시보드")
    st.info("좌측에 데이터 파일을 업로드해 주세요.")
    
    # 대시보드 설명 표시
    st.markdown("""
    ### 분석 메뉴
    
    1. **대시보드**: 핵심 성과 지표, 지역별 분포, 월별 AS 건수, 30일 내 재정비율
    2. **고장 유형 분석**: 고장 유형 분포 및 브랜드-모델별 고장 패턴 히트맵
    3. **브랜드/모델 분석**: 브랜드 및 모델별 특성 분석
    4. **정비내용 분석**: 정비내용 워드클라우드 및 분류별 정비내용 분석
    5. **고장 예측**: 기계학습 모델을 활용한 재정비 기간 및 증상 예측
    """)
