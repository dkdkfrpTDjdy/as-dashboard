import streamlit as st
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import urllib.request
import platform
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter
from wordcloud import WordCloud
import io
import base64
import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests
import subprocess

# 페이지 설정이 가장 먼저 와야 함
st.set_page_config(
    page_title="산업장비 AS 분석 대시보드",
    layout="wide"
)

# 한글 폰트 설정 함수
def setup_korean_font_test():
    # 1. 프로젝트 내 포함된 폰트 우선 적용
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

# 주소에서 지역 추출 함수 개선 버전
def extract_region_from_address(address):
    if not isinstance(address, str):
        return None
    
    # 주소 형태인 경우만 처리
    if len(address) >= 3:  # 최소 "시/도 " 형태 (3글자 이상) 필요
        first_two = address[:2]
        
        # 시/도 약칭 리스트
        regions = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', 
                  '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
        
        # 첫 두 글자가 시/도이고 그 뒤에 공백이 있으면 주소로 간주
        if first_two in regions and address[2] == ' ':
            return first_two
    return None

# 현장 컬럼에서 지역 추출 및 적용
def extract_and_apply_region(df):
    if 'current' in df.columns:
        # 지역 추출
        df['지역'] = df['현장'].apply(extract_region_from_address)
        st.sidebar.success("지역 정보 추출이 완료되었습니다.")
    return df

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
            merged_df[['연료', '운전방식', '적재용량', '마스트']] = merged_df['자재내역'].str.split(' ', n=3, expand=True)
            st.sidebar.success(f"자재내역 분할 완료: 연료, 운전방식, 적재용량, 마스트 컬럼이 생성되었습니다.")
        
        return merged_df
    except Exception as e:
        st.error(f"데이터 병합 중 오류 발생: {e}")
        return df1  # 오류 발생시 원본 데이터프레임 반환

# 최근 정비일자 계산 함수 (관리번호 기준)
def calculate_previous_maintenance_dates(df):
    if '관리번호' not in df.columns or '정비일자' not in df.columns:
        return df
    
    # 정비일자 정렬 및 그룹화
    df = df.sort_values(['관리번호', '정비일자'])
    
    # 각 관리번호별로 이전 정비일자 계산
    df['최근정비일자'] = df.groupby('관리번호')['정비일자'].shift(1)
    
    return df

# 조직도 데이터와 정비자번호/출고자 매핑 함수
def map_employee_data(df, org_df):
    if org_df is None or df is None:
        return df
        
    try:
        # 정비일지 데이터인 경우 (정비자번호 있음)
        if '정비자번호' in df.columns and '사번' in org_df.columns:
            # 데이터 타입 변환 - 문자열로 통일
            df['정비자번호'] = df['정비자번호'].astype(str)
            org_df['사번'] = org_df['사번'].astype(str)
            
            # 사번과 정비자번호 매핑
            df = pd.merge(df, org_df[['사번', '소속']],
                        left_on='정비자번호', right_on='사번', how='left')
                        
            # 컬럼명 변경
            df.rename(columns={'소속': '정비자소속'}, inplace=True)
            
            # 중복 컬럼 제거 (사번_y)
            if '사번_y' in df.columns:
                df = df.drop('사번_y', axis=1)
            if '사번_x' in df.columns:
                df = df.rename(columns={'사번_x': '사번'})
                    
        # 수리비 데이터인 경우 (출고자 있음)
        if '출고자' in df.columns and '사번' in org_df.columns:
            # 출고자와 사번 데이터 타입 통일
            if '정비자번호' in df.columns:
                df['정비자번호'] = df['정비자번호'].astype(str)
            org_df['사번'] = org_df['사번'].astype(str)
            
            # 사번과 출고자 매핑
            df = pd.merge(df, org_df[['사번', '소속']],
                        left_on='정비자번호', right_on='사번', how='left')
                        
            # 컬럼명 변경
            df.rename(columns={'소속': '출고자소속'}, inplace=True)
            
            # 중복 컬럼 제거
            if '사번_y' in df.columns:
                df = df.drop('사번_y', axis=1)
            if '사번_x' in df.columns:
                df = df.rename(columns={'사번_x': '사번'})
                    
        return df
    except Exception as e:
        st.error(f"직원 데이터 매핑 중 오류 발생: {e}")
        return df

# 폰트 설정 실행
font_path = setup_korean_font_test()
# 이후 코드에서 사용 가능
if font_path and os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

# 메뉴별 색상 테마 설정
color_themes = {
    "정비일지 대시보드": "Blues",
    "수리비 대시보드": "Purples",
    "고장 유형 분석": "Greens",
    "브랜드/모델 분석": "Oranges",
    "정비내용 분석": "YlOrRd",
    "고장 예측": "viridis"
}

# 사이드바 설정
st.sidebar.title("데이터 업로드 및 메뉴 클릭")

@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file)
        
        # 컬럼명 정리 (줄바꿈 제거 및 공백 제거)
        df.columns = [str(col).strip().replace('\n', '') for col in df.columns]
        
        return df
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        return None

# 변수 초기화 (한 번만 초기화)
df1 = None  # 정비일지 데이터 (사용자 업로드)
df2 = None  # 자산조회 데이터
df3 = None  # 수리비 데이터 (사용자 업로드)
df4 = None  # 조직도 데이터

# 파일 업로더 - 사용자가 업로드할 파일만 표시
uploaded_file1 = st.sidebar.file_uploader("**정비일지 데이터 업로드**", type=["xlsx"])
uploaded_file3 = st.sidebar.file_uploader("**수리비 데이터 업로드**", type=["xlsx"])

# 사용자 업로드 파일 처리
if uploaded_file1 is not None:
    df1 = load_data(uploaded_file1)
    file_name1 = uploaded_file1.name

if uploaded_file3 is not None:
    df3 = load_data(uploaded_file3)
    file_name3 = uploaded_file3.name

# 깃허브 레포지토리 내 파일 직접 로드
try:
    # 자산조회 데이터 로드 (같은 레포지토리 내 파일)
    asset_data_path = "data/자산조회데이터.xlsx"
    df2 = pd.read_excel(asset_data_path)
except Exception as e:
    df2 = None

try:
    # 조직도 데이터 로드 (같은 레포지토리 내 파일)
    org_data_path = "data/조직도데이터.xlsx"
    df4 = pd.read_excel(org_data_path)
except Exception as e:
    df4 = None

# 데이터 병합 및 전처리
if df1 is not None:
    # 자산 데이터와 병합
    if df2 is not None:
        df1 = merge_dataframes(df1, df2)
        st.sidebar.success("정비일지와 자산조회 파일이 성공적으로 병합되었습니다.")
    
    # 최근 정비일자 계산
    df1 = calculate_previous_maintenance_dates(df1)
    
    # 조직도 데이터 매핑
    if df4 is not None:
        df1 = map_employee_data(df1, df4)
        st.sidebar.success("정비일지에 조직도 정보가 매핑되었습니다.")
    
    # 현장 컬럼에서 지역 정보 추출
    df1 = extract_and_apply_region(df1)
    
    # 날짜 변환 - 오류 수정
    try:
        date_columns = ['정비일자', '최근정비일자']
        for col in date_columns:
            if col in df1.columns:
                try:
                    # 기본 날짜 변환 시도
                    df1[col] = pd.to_datetime(df1[col], errors='coerce')
                except Exception as e:  # 구체적인 예외 처리
                    st.warning(f"{col} 기본 날짜 변환 실패: {e}")
                    try:
                        # Excel 날짜 숫자 처리 시도
                        df1[col] = pd.to_datetime(df1[col], origin='1899-12-30', unit='D', errors='coerce')
                    except Exception as e2:
                        st.warning(f"{col} Excel 날짜 변환도 실패: {e2}")
        
        # 재정비 간격 계산 (정비일자 - 최근정비일자)
        if '최근정비일자' in df1.columns and '정비일자' in df1.columns:
            df1['재정비간격'] = (df1['정비일자'] - df1['최근정비일자']).dt.days
            # 30일 내 재정비 여부
            df1['30일내재정비'] = (df1['재정비간격'] <= 30) & (df1['재정비간격'] > 0)
    
    except Exception as main_error:
        st.error(f"정비일지 데이터 전처리 중 오류 발생: {main_error}")

# 수리비 데이터 전처리
if df3 is not None:
    # 조직도 데이터 매핑
    if df4 is not None:
        df3 = map_employee_data(df3, df4)

    try:
        # 날짜 변환
        if '출고일자' in df3.columns:
            df3['출고일자'] = pd.to_datetime(df3['출고일자'], errors='coerce')
        
        # 금액 컬럼 숫자로 변환
        for col in df3.columns:
            if '금액' in col or '비용' in col or '단가' in col:
                df3[col] = pd.to_numeric(df3[col], errors='coerce')
    except Exception as e:
        st.warning(f"수리비 데이터 전처리 중 오류가 발생했습니다: {e}")

# 이제 데이터 로딩과 전처리가 모두 완료되었으므로, 대시보드 표시 함수를 정의합니다.

# 정비일지 대시보드 표시 함수
def display_maintenance_dashboard(df, category_name):
    # 지표 카드용 컬럼 생성
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cases = len(df)
        st.metric(f"{category_name} AS 건수", f"{total_cases:,}")
    
    with col2:
        # 가동시간 컬럼이 있는지 확인하고, 없으면 비슷한 이름의 컬럼 찾기
        operation_col = None
        for col in df.columns:
            if '가동시간' in col:
                operation_col = col
                break
                
        if operation_col:
            avg_operation = df[operation_col].mean()
            st.metric("평균 가동시간", f"{avg_operation:.2f}시간")
        else:
            st.metric("평균 가동시간", "데이터 없음")
    
    with col3:
        # 수리시간 컬럼이 있는지 확인하고, 없으면 비슷한 이름의 컬럼 찾기
        repair_col = None
        for col in df.columns:
            if '수리시간' in col:
                repair_col = col
                break
                
        if repair_col:
            avg_repair = df[repair_col].mean()
            st.metric("평균 수리시간", f"{avg_repair:.2f}시간")
        else:
            st.metric("평균 수리시간", "데이터 없음")
    
    with col4:
        # 정비일자 컬럼이 있는지 확인하고, 없으면 비슷한 이름의 컬럼 찾기
        date_col = None
        for col in df.columns:
            if '정비일자' in col:
                date_col = col
                break
                
        if date_col and not df[date_col].empty:
            last_month = df[date_col].max().strftime('%Y-%m')
            last_month_count = df[df[date_col].dt.strftime('%Y-%m') == last_month].shape[0]
            st.metric("최근 월 AS 건수", f"{last_month_count:,}")
        else:
            st.metric("최근 월 AS 건수", "데이터 없음")
            
    st.markdown("---")
    
    # 1. 월별 AS건수 + 월별 평균 가동 및 수리시간 + 수리시간 분포
    col1, col2, col3 = st.columns(3)
        
    with col1:
        # 월별 AS 건수
        st.subheader("월별 AS 건수")
        if '정비일자' in df.columns:
            df_time = df.copy()
            df_time['월'] = df_time['정비일자'].dt.to_period('M')
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
            st.markdown(get_image_download_link(fig, f'{category_name}_월별_AS_건수.png', '월별 AS 건수 다운로드'), unsafe_allow_html=True)
            
    with col2:
        if '가동시간' in df.columns:
            st.subheader("월별 평균 가동시간")
            if '정비일자' in df.columns:
                df['월'] = df['정비일자'].dt.to_period('M')
                # 이후 그래프 생성
            else:
                st.warning("정비일자 컬럼이 없습니다. 월별 분석을 건너뜁니다.")
                
            monthly_avg = df.groupby('월')['가동시간'].mean().reset_index()
            
            fig, ax = create_figure_with_korean(figsize=(10, 6), dpi=300)
            sns.barplot(data=monthly_avg, x='월', y='가동시간', ax=ax, palette="Blues")
            
            # 평균값 텍스트 표시
            for index, row in monthly_avg.iterrows():
                ax.text(index, row['가동시간'] + 0.2, f"{row['가동시간']:.1f}시간", ha='center')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            
            # 다운로드 링크 추가
            st.markdown(get_image_download_link(fig, f'{category_name}_월별_평균_가동시간.png', '월별 평균 가동시간 다운로드'), unsafe_allow_html=True)
        
    with col3:
        if '수리시간' in df.columns:
            st.subheader("수리시간 분포")

            # 수리시간 구간화
            bins = [0, 2, 4, 8, 12, 24, float('inf')]
            labels = ['0-2시간', '2-4시간', '4-8시간', '8-12시간', '12-24시간', '24시간 이상']
            df['수리시간_구간'] = pd.cut(df['수리시간'], bins=bins, labels=labels)
            repair_time_counts = df['수리시간_구간'].value_counts().sort_index()

            fig, ax = create_figure_with_korean(figsize=(10, 6), dpi=300)
            sns.barplot(x=repair_time_counts.index, y=repair_time_counts.values, ax=ax, palette="Blues")

            # 막대 위에 텍스트 표시
            for i, v in enumerate(repair_time_counts.values):
                ax.text(i, v + max(repair_time_counts.values) * 0.02, str(v),
                        ha='center', fontsize=12)

            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            st.markdown(get_image_download_link(fig, f'{category_name}_수리시간_분포.png', '수리시간 분포 다운로드'), unsafe_allow_html=True)
            
    st.markdown("---")
    
    # 2. 가동률 분석 + 수리시간 분석 + 지역별 AS 건수
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 가동률 분석 추가
        st.subheader("장비 가동률")
        
        # 유효한 데이터만 사용
        valid_operation = df.copy()
        
        # 필요한 컬럼 찾기
        operation_col = None
        repair_col = None
        
        for col in df.columns:
            if '가동시간' in col:
                operation_col = col
            if '수리시간' in col:
                repair_col = col
        
        # 두 컬럼이 모두 있는 경우에만 가동률 분석 수행
        if operation_col and repair_col:
            valid_operation = valid_operation.dropna(subset=[operation_col, repair_col])
            # 이후 코드에서도 'operation_col'과 'repair_col' 변수를 사용
        else:
            st.warning("가동시간 또는 수리시간 컬럼을 찾을 수 없습니다.")
        
        if len(valid_operation) > 0:
            # 가동률 = 가동시간 / (가동시간 + 수리시간)
            valid_operation['가동률'] = valid_operation['가동시간'] / (valid_operation['가동시간'] + valid_operation['수리시간'])
            valid_operation['가동률'] = valid_operation['가동률'] * 100  # 퍼센트로 변환
            
            # 가동률 구간화
            bins = [0, 80, 90, 95, 100]
            labels = ['80% 미만', '80-90%', '90-95%', '95% 이상']
            valid_operation['가동률_구간'] = pd.cut(valid_operation['가동률'], bins=bins, labels=labels)
            operation_rate_counts = valid_operation['가동률_구간'].value_counts().sort_index()
            
            fig, ax = create_figure_with_korean(figsize=(10, 6), dpi=300)
            sns.barplot(x=operation_rate_counts.index, y=operation_rate_counts.values, ax=ax, palette="Blues")
            
            # 막대 위에 텍스트 표시
            for i, v in enumerate(operation_rate_counts.values):
                ax.text(i, v + max(operation_rate_counts.values) * 0.02, str(v),
                      ha='center', fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            # 다운로드 링크 추가
            st.markdown(get_image_download_link(fig, f'{category_name}_가동률_분포.png', '가동률 분포 다운로드'), unsafe_allow_html=True)
            
            # 평균 가동률 정보 추가
            avg_operation_rate = valid_operation['가동률'].mean()
            st.info(f"평균 가동률: {avg_operation_rate:.2f}%")
            
        else:
            st.warning("가동시간 및 수리시간 데이터가 부족합니다.")
            
    with col2:
        if '수리시간' in df.columns and '가동시간' in df.columns:
            st.subheader("수리시간 vs 가동시간")
            
            # 산점도 데이터 준비
            scatter_df = df.dropna(subset=['가동시간', '수리시간']).copy()
            
            if len(scatter_df) > 0:
                # 일부 데이터만 표시 (최대 1000개)
                if len(scatter_df) > 1000:
                    scatter_df = scatter_df.sample(1000, random_state=42)
                
                fig, ax = create_figure_with_korean(figsize=(10, 6), dpi=300)
                scatter = ax.scatter(
                    scatter_df['가동시간'], 
                    scatter_df['수리시간'],
                    alpha=0.6,
                    c=scatter_df['가동시간'] / (scatter_df['가동시간'] + scatter_df['수리시간']),
                    cmap='viridis'
                )
                
                # 추세선 추가
                z = np.polyfit(scatter_df['가동시간'], scatter_df['수리시간'], 1)
                p = np.poly1d(z)
                ax.plot(scatter_df['가동시간'], p(scatter_df['가동시간']), "r--", alpha=0.8)
                
                ax.set_xlabel('가동시간 (시간)')
                ax.set_ylabel('수리시간 (시간)')
                
                # 컬러바 추가
                cbar = plt.colorbar(scatter)
                cbar.set_label('가동률')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                
                # 다운로드 링크 추가
                st.markdown(get_image_download_link(fig, f'{category_name}_가동시간_수리시간_산점도.png', '가동시간 vs 수리시간 다운로드'), unsafe_allow_html=True)
                
                # 상관계수 표시
                corr = scatter_df['가동시간'].corr(scatter_df['수리시간'])
                st.info(f"가동시간과 수리시간의 상관계수: {corr:.3f}")
            else:
                st.warning("가동시간 및 수리시간 데이터가 부족합니다.")
    
    with col3:
        # 지역별 빈도 분석
        st.subheader("지역별 AS 건수")
        if '지역' in df.columns:
            df_clean = df.dropna(subset=['지역']).copy()
            
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
            blue_palette = sns.color_palette("Blues", n_colors=len(region_counts))
            
            sns.barplot(x=region_counts.index, y=region_counts.values, ax=ax, palette=blue_palette)
            
            # 막대 위에 텍스트 표시
            for i, v in enumerate(region_counts.values):
                ax.text(i, v + max(region_counts.values) * 0.02, str(v),
                       ha='center', fontsize=12)
            
            plt.tight_layout()
            plt.xticks(rotation=45)
            st.pyplot(fig, use_container_width=True)
            
            # 다운로드 링크 추가
            st.markdown(get_image_download_link(fig, f'{category_name}_지역별_AS_현황.png', '지역별 AS 현황 다운로드'), unsafe_allow_html=True)
        else:
            st.warning("지역 정보가 없습니다.")
    
    # 추가: 브랜드별 평균 수리시간
    if '브랜드' in df.columns and '수리시간' in df.columns:
        st.subheader("브랜드별 평균 수리시간")
        
        # 브랜드별 평균 수리시간 계산
        brand_repair_times = df.groupby('브랜드')['수리시간'].agg(['mean', 'count']).reset_index()
        brand_repair_times = brand_repair_times.sort_values(by='mean', ascending=False)
        brand_repair_times = brand_repair_times[brand_repair_times['count'] >= 5]  # 데이터가 충분한 브랜드만
        
        if len(brand_repair_times) > 0:
            fig, ax = create_figure_with_korean(figsize=(12, 6), dpi=300)
            
            # 막대 그래프 생성
            bars = sns.barplot(x='브랜드', y='mean', data=brand_repair_times.head(15), ax=ax, palette="Blues_r")
            
            # 막대 위에 텍스트 표시
            for i, v in enumerate(brand_repair_times['mean'].head(15)):
                ax.text(i, v + 0.1, f"{v:.2f}시간", ha='center')
            
            ax.set_xlabel('브랜드')
            ax.set_ylabel('평균 수리시간 (시간)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            
            # 다운로드 링크 추가
            st.markdown(get_image_download_link(fig, f'{category_name}_브랜드별_평균수리시간.png', '브랜드별 평균수리시간 다운로드'), unsafe_allow_html=True)
        else:
            st.warning("브랜드별 수리시간 데이터가 부족합니다.")
    
    # 정비자 소속별 분석 (조직도 데이터가 있는 경우)
    if '정비자소속' in df.columns:
        st.subheader("정비자 소속별 분석")
                
        col1, col2 = st.columns(2)
                
        with col1:
            # 소속별 정비 건수
            dept_counts = df['정비자소속'].value_counts().head(10)
                        
            fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
            sns.barplot(x=dept_counts.values, y=dept_counts.index, ax=ax, palette="Blues_r")
                        
            # 막대 위에 텍스트 표시
            for i, v in enumerate(dept_counts.values):
                ax.text(v + 0.5, i, str(v), va='center')
                        
            ax.set_xlabel('정비 건수')
            plt.tight_layout()
                        
            st.pyplot(fig, use_container_width=True)
            st.markdown(get_image_download_link(fig, f'{category_name}_소속별_정비건수.png', '소속별 정비건수 다운로드'), unsafe_allow_html=True)
                
        with col2:
            # 소속별 평균 수리시간
            if '수리시간' in df.columns:
                dept_avg_repair = df.groupby('정비자소속')['수리시간'].mean().sort_values(ascending=False).head(10)
                                
                fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                sns.barplot(x=dept_avg_repair.values, y=dept_avg_repair.index, ax=ax, palette="Blues_r")
                                
                # 막대 위에 텍스트 표시
                for i, v in enumerate(dept_avg_repair.values):
                    ax.text(v + 0.1, i, f"{v:.1f}시간", va='center')
                                
                ax.set_xlabel('평균 수리시간')
                plt.tight_layout()
                                
                st.pyplot(fig, use_container_width=True)
                st.markdown(get_image_download_link(fig, f'{category_name}_소속별_평균수리시간.png', '소속별 평균수리시간 다운로드'), unsafe_allow_html=True)
            else:
                st.warning("수리시간 데이터가 없습니다.")

# 수리비 대시보드 표시 함수
def display_repair_cost_dashboard(df):
    if df is None:
        st.warning("수리비 데이터가 없습니다.")
        return
    
    # 기본 지표
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cases = len(df)
        st.metric("총 수리 건수", f"{total_cases:,}")
    
    with col2:
        if '금액' in df.columns:
            total_cost = df['금액'].sum()
            st.metric("총 수리비용", f"{total_cost:,.0f}원")
        else:
            for col in df.columns:
                if '금액' in col or '비용' in col:
                    total_cost = df[col].sum()
                    st.metric("총 수리비용", f"{total_cost:,.0f}원")
                    break
            else:
                st.metric("총 수리비용", "데이터 없음")
    
    with col3:
        if '출고일자' in df.columns:
            last_month = df['출고일자'].max().strftime('%Y-%m')
            last_month_count = df[df['출고일자'].dt.strftime('%Y-%m') == last_month].shape[0]
            st.metric("최근 월 수리 건수", f"{last_month_count:,}")
        else:
            st.metric("최근 월 수리 건수", "데이터 없음")
    
    with col4:
        if '출고자소속' in df.columns:
            dept_counts = df['출고자소속'].value_counts()
            top_dept = dept_counts.index[0] if not dept_counts.empty else "정보 없음"
            top_dept_count = dept_counts.iloc[0] if not dept_counts.empty else 0
            st.metric("최다 출고 소속", f"{top_dept} ({top_dept_count}건)")
        else:
            st.metric("최다 출고 소속", "데이터 없음")
    
    st.markdown("---")
    
    # 월별 수리 건수 및 비용
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("월별 수리 건수")
        if '출고일자' in df.columns:
            df_time = df.copy()
            df_time['월'] = df_time['출고일자'].dt.to_period('M')
            monthly_counts = df_time.groupby('월').size().reset_index(name='건수')
            monthly_counts['월'] = monthly_counts['월'].astype(str)
            
            fig, ax = create_figure_with_korean(figsize=(10, 6), dpi=300)
            sns.barplot(x='월', y='건수', data=monthly_counts, ax=ax, palette='Purples')
            
            # 막대 위에 텍스트 표시
            for i, v in enumerate(monthly_counts['건수']):
                ax.text(i, v + max(monthly_counts['건수']) * 0.01, str(v), ha='center')
            
            plt.xticks(rotation=45)
            ax.set_ylabel('건수')
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            st.markdown(get_image_download_link(fig, '월별_수리_건수.png', '월별 수리 건수 다운로드'), unsafe_allow_html=True)
    
    with col2:
        st.subheader("월별 수리 비용")
        if '출고일자' in df.columns and ('금액' in df.columns or any('금액' in col for col in df.columns)):
            df_time = df.copy()
            df_time['월'] = df_time['출고일자'].dt.to_period('M')
            
            # 금액 컬럼 찾기
            cost_col = '금액' if '금액' in df.columns else next((col for col in df.columns if '금액' in col), None)
            
            if cost_col:
                monthly_costs = df_time.groupby('월')[cost_col].sum().reset_index()
                monthly_costs['월'] = monthly_costs['월'].astype(str)
                
                fig, ax = create_figure_with_korean(figsize=(10, 6), dpi=300)
                sns.barplot(x='월', y=cost_col, data=monthly_costs, ax=ax, palette='Purples')
                
                # 막대 위에 텍스트 표시 (천 단위 구분)
                for i, v in enumerate(monthly_costs[cost_col]):
                    ax.text(i, v + max(monthly_costs[cost_col]) * 0.01, f"{v:,.0f}", ha='center', fontsize=8)
                
                plt.xticks(rotation=45)
                ax.set_ylabel('비용 (원)')
                plt.tight_layout()
                
                st.pyplot(fig, use_container_width=True)
                st.markdown(get_image_download_link(fig, '월별_수리_비용.png', '월별 수리 비용 다운로드'), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 소속별 수리 건수 및 비용
    if '출고자소속' in df.columns:
        st.subheader("소속별 수리 현황")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 소속별 수리 건수
            dept_counts = df['출고자소속'].value_counts().head(10)
            
            fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
            sns.barplot(x=dept_counts.values, y=dept_counts.index, ax=ax, palette="Purples_r")
            
            # 막대 위에 텍스트 표시
            for i, v in enumerate(dept_counts.values):
                ax.text(v + 0.5, i, str(v), va='center')
            
            ax.set_xlabel('수리 건수')
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            st.markdown(get_image_download_link(fig, '소속별_수리건수.png', '소속별 수리건수 다운로드'), unsafe_allow_html=True)
        
        with col2:
            # 소속별 수리 비용
            cost_col = '금액' if '금액' in df.columns else next((col for col in df.columns if '금액' in col), None)
            
            if cost_col:
                dept_costs = df.groupby('출고자소속')[cost_col].sum().sort_values(ascending=False).head(10)
                
                fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                sns.barplot(x=dept_costs.values, y=dept_costs.index, ax=ax, palette="Purples_r")
                
                # 막대 위에 텍스트 표시 (천 단위 구분)
                for i, v in enumerate(dept_costs.values):
                    ax.text(v + max(dept_costs.values) * 0.01, i, f"{v:,.0f}", va='center', fontsize=8)
                
                ax.set_xlabel('수리 비용 (원)')
                plt.tight_layout()
                
                st.pyplot(fig, use_container_width=True)
                st.markdown(get_image_download_link(fig, '소속별_수리비용.png', '소속별 수리비용 다운로드'), unsafe_allow_html=True)
    
    # 소속별 인원 대비 수리 건수 (조직도 데이터가 있는 경우)
    if '출고자소속' in df.columns and df4 is not None and '소속' in df4.columns:
        st.subheader("소속별 인원 대비 수리 건수")
        
        # 소속별 인원 수 계산
        dept_staff = df4.groupby('소속').size()
        
        # 소속별 수리 건수
        repair_by_dept = df.groupby('출고자소속').size()
        
        # 공통 소속만 추출
        common_depts = sorted(set(dept_staff.index) & set(repair_by_dept.index))
        
        if common_depts:
            # 데이터 준비
            dept_comparison = pd.DataFrame({
                '소속': common_depts,
                '인원수': [dept_staff.get(dept, 0) for dept in common_depts],
                '수리건수': [repair_by_dept.get(dept, 0) for dept in common_depts]
            })
            
            # 인원 대비 수리 건수 비율 계산
            dept_comparison['인원당수리건수'] = (dept_comparison['수리건수'] / dept_comparison['인원수']).round(2)
            dept_comparison = dept_comparison.sort_values('인원당수리건수', ascending=False).head(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 소속별 인원 및 수리 건수 비교
                fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                
                x = np.arange(len(dept_comparison))
                width = 0.4
                
                # 인원 수 (정규화)
                max_staff = dept_comparison['인원수'].max()
                max_repair = dept_comparison['수리건수'].max()
                scale_factor = max_repair / max_staff if max_staff > 0 else 1
                
                ax.bar(x - width/2, dept_comparison['인원수'] * scale_factor, width, 
                      label='인원수 (정규화)', color='#8ECAE6', alpha=0.7)
                ax.bar(x + width/2, dept_comparison['수리건수'], width, 
                      label='수리건수', color='#9370DB')
                
                # 축 설정
                ax.set_xticks(x)
                ax.set_xticklabels(dept_comparison['소속'], rotation=45, ha='right')
                ax.legend()
                
                # 보조 y축 추가 (실제 인원수)
                ax2 = ax.twinx()
                ax2.set_ylim(0, max_staff * 1.1)
                ax2.set_ylabel('실제 인원수')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                st.markdown(get_image_download_link(fig, '소속별_인원수리건수_비교.png', '소속별 인원수리건수 비교 다운로드'), unsafe_allow_html=True)
            
            with col2:
                # 인원당 수리 건수
                fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                sns.barplot(x=dept_comparison['인원당수리건수'], y=dept_comparison['소속'], ax=ax, palette="Purples_r")
                
                # 막대 위에 텍스트 표시
                for i, v in enumerate(dept_comparison['인원당수리건수']):
                    ax.text(v + 0.1, i, f"{v:.2f}", va='center')
                
                ax.set_xlabel('인원당 수리 건수')
                plt.tight_layout()
                
                st.pyplot(fig, use_container_width=True)
                st.markdown(get_image_download_link(fig, '소속별_인원당수리건수.png', '소속별 인원당수리건수 다운로드'), unsafe_allow_html=True)

# 데이터가 로드된 경우 분석 시작
if df1 is not None or df3 is not None:
    # 메뉴 선택 - 정비일지와 수리비 대시보드 분리
    menu_options = []
    
    if df1 is not None:
        menu_options.append("정비일지 대시보드")
        menu_options.extend(["고장 유형 분석", "브랜드/모델 분석", "정비내용 분석", "고장 예측"])
    
    if df3 is not None:
        menu_options.append("수리비 대시보드")
    
    menu = st.sidebar.selectbox("분석 메뉴", menu_options)
    
    # 현재 메뉴의 색상 테마 설정
    current_theme = color_themes[menu]
    
    # 메뉴별 콘텐츠 표시
    if menu == "정비일지 대시보드":
        st.title("정비일지 대시보드")
        
        # 정비구분 컬럼 확인 및 값 검증
        if '정비구분' in df1.columns and df1['정비구분'].notna().any():
            # 실제 존재하는 정비구분 값 확인
            maintenance_types = df1['정비구분'].dropna().unique()
            
            # 내부, 외부 값이 있는지 확인
            has_internal = '내부' in maintenance_types
            has_external = '외부' in maintenance_types
            
            # 탭 생성
            tabs = st.tabs(["전체", "내부", "외부"])
            
            # 전체 탭
            with tabs[0]:
                st.header("전체 정비 현황")
                # 전체 데이터 표시
                display_maintenance_dashboard(df1, "전체")
            
            # 내부 탭
            with tabs[1]:
                st.header("내부 정비 현황")
                if has_internal:
                    df_internal = df1[df1['정비구분'] == '내부']
                    display_maintenance_dashboard(df_internal, "내부")
                else:
                    st.info("내부 정비 데이터가 없습니다.")
            
            # 외부 탭
            with tabs[2]:
                st.header("외부 정비 현황")
                if has_external:
                    df_external = df1[df1['정비구분'] == '외부']
                    display_maintenance_dashboard(df_external, "외부")
                else:
                    st.info("외부 정비 데이터가 없습니다.")
        else:
            # 정비구분 컬럼이 없는 경우 전체 데이터만 표시
            st.header("정비 현황")
            display_maintenance_dashboard(df1, "전체")

    elif menu == "수리비 대시보드":
        st.title("수리비 대시보드")
        display_repair_cost_dashboard(df3)
    
    # 나머지 메뉴 구현 - 여기서 구현할 수 있습니다
    elif menu == "고장 유형 분석":
        st.title("고장 유형 분석")
        st.info("고장 유형 분석 기능이 구현 중입니다.")
    
    elif menu == "브랜드/모델 분석":
        st.title("브랜드/모델 분석")
        st.info("브랜드/모델 분석 기능이 구현 중입니다.")
    
    elif menu == "정비내용 분석":
        st.title("정비내용 분석")
        st.info("정비내용 분석 기능이 구현 중입니다.")
    
    elif menu == "고장 예측":
        st.title("고장 예측")
        st.info("고장 예측 기능이 구현 중입니다.")

else:
    st.header("산업장비 AS 대시보드")
    st.info("좌측에 데이터 파일을 업로드해 주세요.")
    
    # 대시보드 설명 표시
    st.markdown("""
    ### 분석 메뉴
    
    1. **정비일지 대시보드**: 정비일지 데이터 기반의 AS 분석 (정비구분별 탭 제공)
    2. **수리비 대시보드**: 수리비 데이터 기반의 비용 분석
    3. **고장 유형 분석**: 고장 유형 분포 및 브랜드-모델별 고장 패턴 히트맵
    4. **브랜드/모델 분석**: 브랜드 및 모델별 특성 분석
    5. **정비내용 분석**: 정비내용 워드클라우드 및 분류별 정비내용 분석
    6. **고장 예측**: 기계학습 모델을 활용한 재정비 기간 및 증상 예측
    
    ### 필요한 파일
    
    - **정비일지 데이터**: 장비 AS 정보
    - **수리비 데이터**: 수리비용 정보
    """)
