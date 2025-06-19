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

def extract_region_from_address(address):
    if not isinstance(address, str):
        return None, None
    
    # 주소 형태인 경우만 처리
    if len(address) >= 3:  # 최소 "시/도 " 형태 (3글자 이상) 필요
        first_two = address[:2]
        
        # 시/도 약칭 리스트
        regions = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', 
                  '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
        
        # 첫 두 글자가 시/도이고 그 뒤에 공백이 있으면 주소로 간주
        if first_two in regions and address[2] == ' ':
            return first_two, address  # 지역과 전체 주소 반환
    
    return None, address  # 지역 없음, 원본 값은 현장명으로 간

# 현장 컬럼에서 지역과 현장명 추출 및 적용
def extract_and_apply_region(df):
    if '현장' in df.columns:
        # 지역과 주소/현장명 추출
        results = df['현장'].apply(extract_region_from_address)
        df['지역'] = [r[0] for r in results]
        df['현장명'] = [r[1] for r in results]
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

# 최근 정비일자 계산 함수 (관리번호 기준)
def calculate_previous_maintenance_dates(df):
    if '관리번호' not in df.columns or '정비일자' not in df.columns:
        return df

    # 정비일자 정렬 및 그룹화
    df = df.sort_values(['관리번호', '정비일자'])

    # 각 관리번호별로 이전 정비일자 계산
    df['최근정비일자'] = df.groupby('관리번호')['정비일자'].shift(1)

    return df

# 조직도 데이터와 정비자번호/출고자 매핑 함수 (통합 버전)
def map_employee_data(df, org_df):
    if org_df is None or df is None:
        return df

    try:
        # 결과 데이터프레임 복사
        result_df = df.copy()
        org_temp = org_df.copy()

        # 조직도의 사번을 문자열로 통일
        org_temp['사번'] = org_temp['사번'].astype(str)

        # 정비일지 데이터인 경우 (정비자번호 있음)
        if '정비자번호' in result_df.columns:
            # 정비자번호를 문자열로 변환
            result_df['정비자번호'] = result_df['정비자번호'].astype(str)

            # 소속 정보만 가져오기 (left join)
            result_df = pd.merge(
                result_df,
                org_temp[['사번', '소속']],
                left_on='정비자번호',
                right_on='사번',
                how='left'
            )

            # 소속 컬럼명 변경 및 중복 컬럼 제거
            result_df.rename(columns={'소속': '정비자소속'}, inplace=True)
            if '사번' in result_df.columns:
                result_df.drop('사번', axis=1, inplace=True)

        # 수리비 데이터인 경우 (출고자 있음)
        elif '출고자' in result_df.columns:
            # 출고자를 문자열로 변환
            result_df['출고자'] = result_df['출고자'].astype(str)

            # 소속 정보만 가져오기 (left join)
            result_df = pd.merge(
                result_df,
                org_temp[['사번', '소속']],
                left_on='출고자',
                right_on='사번',
                how='left'
            )

            # 소속 컬럼명 변경 및 중복 컬럼 제거
            result_df.rename(columns={'소속': '출고자소속'}, inplace=True)
            if '사번' in result_df.columns:
                result_df.drop('사번', axis=1, inplace=True)

        return result_df

    except Exception as e:
        st.error(f"직원 데이터 매핑 중 오류 발생: {e}")
        import traceback
        st.error(traceback.format_exc())  # 상세 오류 정보 출력
        return df
    
# 수리비 데이터와 정비일지 데이터 매핑 함수
def map_repair_costs(df1, df3):
    if df1 is None or df3 is None:
        return df1
    
    try:
        # 데이터 타입 통일
        df1 = df1.copy()
        df3 = df3.copy()
        
        df1['관리번호'] = df1['관리번호'].astype(str)
        df3['관리번호'] = df3['관리번호'].astype(str)
        
        df1['정비자번호'] = df1['정비자번호'].astype(str)
        df3['출고자'] = df3['출고자'].astype(str)
        
        # 날짜 형식 통일
        if '정비일자' in df1.columns and '출고일자' in df3.columns:
            df1['정비일자'] = pd.to_datetime(df1['정비일자'], errors='coerce')
            df3['출고일자'] = pd.to_datetime(df3['출고일자'], errors='coerce')
        
        # 결과 데이터프레임 초기화
        df1['수리비'] = 0
        df1['자재명_목록'] = ''
        
        # 매핑 조건에 맞는 데이터 찾기
        for idx, row in df1.iterrows():
            # 관리번호 및 정비자번호가 동일한 df3의 행들 찾기
            matching_repairs = df3[
                (df3['관리번호'] == row['관리번호']) & 
                (df3['출고자'] == row['정비자번호'])
            ]
            
            # 30일 전후로 일치하는지 확인
            if '정비일자' in row and not pd.isna(row['정비일자']):
                date_condition = (
                    (matching_repairs['출고일자'] >= row['정비일자'] - pd.Timedelta(days=30)) &
                    (matching_repairs['출고일자'] <= row['정비일자'] + pd.Timedelta(days=30))
                )
                matching_repairs = matching_repairs[date_condition]
            
            # 매칭된 행이 있는 경우
            if not matching_repairs.empty:
                # 출고금액 합산
                total_cost = matching_repairs['출고금액'].sum()
                
                # 자재명 목록 생성
                materials = ', '.join(matching_repairs['자재명'].dropna().unique())
                
                # 값 업데이트
                df1.at[idx, '수리비'] = total_cost
                df1.at[idx, '자재명_목록'] = materials
        
        return df1
    
    except Exception as e:
        st.error(f"수리비 데이터 매핑 중 오류 발생: {e}")
        import traceback
        st.error(traceback.format_exc())
        return df1

# 폰트 설정 실행
font_path = setup_korean_font_test()
# 이후 코드에서 사용 가능
if font_path and os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

# 메뉴별 색상 테마 설정
color_themes = {
    "정비일지 대시보드": "Blues",
    "고장 유형 분석": "Purples",
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

        # 컬럼명 매핑 (정비일지 데이터인 경우)
        try:
            # 대분류, 중분류, 소분류가 있는 경우 작업유형, 정비대상, 정비작업으로 변환
            if all(col in df.columns for col in ['대분류', '중분류', '소분류']):
                df.rename(columns={
                    '대분류': '작업유형',
                    '중분류': '정비대상',
                    '소분류': '정비작업'
                }, inplace=True)
                
            # 고장유형 조합은 나중에 브랜드 정보가 적절히 설정된 후에 수행
            
        except Exception as e:
            st.warning(f"일부 데이터 전처리 중 오류가 발생했습니다: {e}")

        return df
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        return None

# 두 데이터프레임 병합 함수 - 브랜드 매핑 문제 해결
def merge_dataframes(df1, df2):
    if df1 is None or df2 is None:
        return None

    try:
        # 데이터 타입 통일 - 관리번호를 문자열로 변환
        df1['관리번호'] = df1['관리번호'].astype(str)
        df2['관리번호'] = df2['관리번호'].astype(str)
        
        # 중복 관리번호 확인 및 제거 (자산 데이터에서)
        if df2['관리번호'].duplicated().any():
            # 디버깅 정보
            st.sidebar.info(f"자산조회 데이터에 중복된 관리번호가 있습니다: {df2['관리번호'].duplicated().sum()}개 - 첫번째 값만 사용합니다.")
            # 중복 제거 (첫 번째 값 유지)
            df2 = df2.drop_duplicates(subset='관리번호')
            
        # 자산 데이터에서 필요한 컬럼만 선택
        df2_subset = df2[['관리번호', '제조사명', '제조사모델명', '제조년도', '취득가', '자재내역']]
        
        # 컬럼명 표준화: 제조사명 -> 브랜드, 제조사모델명 -> 모델명
        df2_subset = df2_subset.rename(columns={
            '제조사명': '브랜드',
            '제조사모델명': '모델명'
        })
        
        # 관리번호 컬럼을 기준으로 왼쪽 조인으로 병합 (AS 데이터는 모두 유지)
        # 브랜드와 모델명이 이미 존재할 경우 _x, _y로 구분됨
        merged_df = pd.merge(df1, df2_subset, on='관리번호', how='left')
            
        # 브랜드 컬럼 처리
        # 기존에 브랜드(_x)가 있고 값도 있으면 유지, 없으면 자산데이터의 브랜드(_y)를 사용
        if '브랜드_x' in merged_df.columns and '브랜드_y' in merged_df.columns:
            # 두 컬럼이 모두 있는 경우 - 병합 처리
            merged_df['브랜드'] = merged_df['브랜드_x'].fillna(merged_df['브랜드_y'])
            # 원본 컬럼 삭제
            merged_df = merged_df.drop(['브랜드_x', '브랜드_y'], axis=1)
        elif '브랜드_y' in merged_df.columns:
            # 자산 데이터의 브랜드만 있는 경우
            merged_df['브랜드'] = merged_df['브랜드_y']
            merged_df = merged_df.drop(['브랜드_y'], axis=1)
        elif '브랜드_x' in merged_df.columns:
            # AS 데이터의 브랜드만 있는 경우
            merged_df['브랜드'] = merged_df['브랜드_x']
            merged_df = merged_df.drop(['브랜드_x'], axis=1)
        
        # 브랜드에 여전히 NaN이 있으면 '기타'로 채움
        if '브랜드' in merged_df.columns:
            merged_df['브랜드'] = merged_df['브랜드'].fillna('기타')
        else:
            # 브랜드 컬럼이 없는 경우 새로 생성
            merged_df['브랜드'] = '기타'
        
        # 모델명 처리 (브랜드와 동일한 방식)
        if '모델명_x' in merged_df.columns and '모델명_y' in merged_df.columns:
            merged_df['모델명'] = merged_df['모델명_x'].fillna(merged_df['모델명_y'])
            merged_df = merged_df.drop(['모델명_x', '모델명_y'], axis=1)
        elif '모델명_y' in merged_df.columns:
            merged_df['모델명'] = merged_df['모델명_y']
            merged_df = merged_df.drop(['모델명_y'], axis=1)
        elif '모델명_x' in merged_df.columns:
            merged_df['모델명'] = merged_df['모델명_x']
            merged_df = merged_df.drop(['모델명_x'], axis=1)
            
        # 자재내역 컬럼 분할 (있는 경우만)
        if '자재내역' in merged_df.columns and merged_df['자재내역'].notna().any():
            # 자재내역에서 추가 정보 추출 (공백으로 나누기)
            split_result = merged_df['자재내역'].str.split(' ', n=3, expand=True)
            # 결과가 있을 때만 컬럼 추가
            if len(split_result.columns) >= 4:
                merged_df[['연료', '운전방식', '적재용량', '마스트']] = split_result
            else:
                # 결과 컬럼 수가 부족한 경우 빈 컬럼 생성
                for i, col_name in enumerate(['연료', '운전방식', '적재용량', '마스트']):
                    if i < len(split_result.columns):
                        merged_df[col_name] = split_result[i]
                    else:
                        merged_df[col_name] = None

        # 브랜드와 모델명으로 브랜드_모델 컬럼 생성
        if '브랜드' in merged_df.columns and '모델명' in merged_df.columns:
            mask = merged_df['브랜드'].notna() & merged_df['모델명'].notna()
            merged_df.loc[mask, '브랜드_모델'] = merged_df.loc[mask, '브랜드'].astype(str) + '_' + merged_df.loc[mask, '모델명'].astype(str)
        
        # 고장유형 조합 (이제 브랜드가 적절히 설정되었으므로 수행)
        if all(col in merged_df.columns for col in ['작업유형', '정비대상', '정비작업']):
            # nan 값을 가진 행 필터링하여 처리
            mask = merged_df['작업유형'].notna() & merged_df['정비대상'].notna() & merged_df['정비작업'].notna()
            merged_df.loc[mask, '고장유형'] = (merged_df.loc[mask, '작업유형'].astype(str) + '_' + 
                                            merged_df.loc[mask, '정비대상'].astype(str) + '_' + 
                                            merged_df.loc[mask, '정비작업'].astype(str))

        if merged_df['브랜드'].nunique() < 2:
            st.sidebar.warning("브랜드 값이 너무 적습니다. 병합이 제대로 되지 않았을 수 있습니다.")

        return merged_df
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"데이터 병합 중 오류 발생: {e}\n\n{error_details}")
        return df1  # 오류 발생시 원본 데이터프레임 반환

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
    
    # 컬럼명 정리 (자산데이터도 동일하게 처리)
    df2.columns = [str(col).strip().replace('\n', '') for col in df2.columns]
    
except Exception as e:
    df2 = None
    st.sidebar.warning(f"자산조회 데이터를 로드할 수 없습니다: {e}")

try:
    # 조직도 데이터 로드 (같은 레포지토리 내 파일)
    org_data_path = "data/조직도데이터.xlsx"
    df4 = pd.read_excel(org_data_path)
    
    # 컬럼명 정리
    df4.columns = [str(col).strip().replace('\n', '') for col in df4.columns]
    
except Exception as e:
    df4 = None
    st.sidebar.warning(f"조직도 데이터를 로드할 수 없습니다: {e}")

# 데이터 병합 및 전처리
if df1 is not None:
    # 1. 먼저 자산 데이터 병합
    if df2 is not None:
        # 자산 데이터와 병합
        df1 = merge_dataframes(df1, df2)
        
    # 2. 최근 정비일자 계산
    df1 = calculate_previous_maintenance_dates(df1)

    # 3. 조직도 데이터 매핑
    if df4 is not None:
        df1 = map_employee_data(df1, df4)

    # 4. 현장 컬럼에서 지역 정보 추출
    df1 = extract_and_apply_region(df1)
        
    # 5. 수리비 데이터 매핑 (모든 전처리 후)
    if df3 is not None:
        df1 = map_repair_costs(df1, df3)

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

# 정비일지 대시보드 표시 함수 (수리비 정보 포함하도록 수정함)
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
        # 수리비 컬럼이 있는지 확인
        if '수리비' in df.columns:
            total_repair_cost = df['수리비'].sum()
            st.metric("총 수리비용", f"{total_repair_cost:,.0f}원")
        else:
            st.metric("총 수리비용", "데이터 없음")

    st.markdown("---")

    # 1. 월별 AS건수 + 월별 평균 수리비 + 수리시간 분포
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
        # 월별 수리비 평균 그래프 (수리비 데이터가 있는 경우)
        st.subheader("월별 평균 수리비")
        if '정비일자' in df.columns and '수리비' in df.columns:
            df_cost = df.copy()
            df_cost['월'] = df_cost['정비일자'].dt.to_period('M')
            
            # 월별 평균 수리비 계산
            monthly_costs = df_cost.groupby('월')['수리비'].mean().reset_index()
            monthly_costs['월'] = monthly_costs['월'].astype(str)
            
            fig, ax = create_figure_with_korean(figsize=(10, 6), dpi=300)
            sns.barplot(x='월', y='수리비', data=monthly_costs, ax=ax, palette="Blues")
            
            # 평균값 텍스트 표시
            for i, row in monthly_costs.iterrows():
                ax.text(i, row['수리비'] + 100, f"{row['수리비']:,.0f}원", ha='center', fontsize=8)
            
            plt.xticks(rotation=45)
            ax.set_ylabel('평균 수리비 (원)')
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            st.markdown(get_image_download_link(fig, f'{category_name}_월별_평균_수리비.png', '월별 평균 수리비 다운로드'), unsafe_allow_html=True)
        else:
            st.info("수리비 데이터가 없습니다.")
        
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

    # 지역별 AS 건수, 수리비를 많이 쓴 현장, 인당 수리비 소속 분석을 세 개의 컬럼으로 구성
    col1, col2, col3 = st.columns(3)

    with col1:
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

            region_counts = region_counts.sort_values(ascending=False).nlargest(15)

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

    with col2:
        # 수리비 많은 현장 순위 (새로 추가)
        st.subheader("수리비 많은 현장 TOP 15")
        if '현장명' in df.columns and '수리비' in df.columns:
            # 현장별 총 수리비 계산
            site_costs = df.groupby('현장명')['수리비'].sum().sort_values(ascending=False).head(15)
            
            fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
            sns.barplot(x=site_costs.values, y=site_costs.index, ax=ax, palette="Blues_r")
            
            # 막대 위에 텍스트 표시
            for i, v in enumerate(site_costs.values):
                ax.text(v + max(site_costs.values) * 0.01, i, f"{v:,.0f}원", va='center', fontsize=8)
            
            ax.set_xlabel('총 수리비 (원)')
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            st.markdown(get_image_download_link(fig, f'{category_name}_현장별_수리비.png', '현장별 수리비 다운로드'), unsafe_allow_html=True)
        else:
            st.warning("현장명 또는 수리비 정보가 없습니다.")

    with col3:
        # 인당 수리비 많은 소속 순위 (새로 추가)
        if '정비자소속' in df.columns and '수리비' in df.columns and df4 is not None:
            st.subheader("소속별 인당 수리비 TOP 15")
            
            # 전체 소속별 인원수 계산
            total_staff_by_dept = df4['소속'].value_counts()
            
            # 소속별 총 수리비 계산
            dept_costs = df.groupby('정비자소속')['수리비'].sum().sort_values(ascending=False)
            
            # 소속별 수리비 및 인원 비율 계산
            dept_comparison = pd.DataFrame({
                '소속': dept_costs.index,
                '총수리비': dept_costs.values,
                '소속인원수': [total_staff_by_dept.get(dept, 0) for dept in dept_costs.index]
            })
            
            # 인원이 0이면 1로 설정하여 나누기 오류 방지
            dept_comparison['소속인원수'] = dept_comparison['소속인원수'].replace(0, 1)
            
            # 인원당 수리비 계산
            dept_comparison['인원당수리비'] = (dept_comparison['총수리비'] / dept_comparison['소속인원수']).round(0)
            
            # 결과 소트하고 상위 15개 선택
            dept_comparison = dept_comparison.sort_values('인원당수리비', ascending=False).head(15)
            
            fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
            sns.barplot(x=dept_comparison['인원당수리비'], y=dept_comparison['소속'], ax=ax, palette="Blues_r")
            
            # 막대 위에 텍스트 표시
            for i, row in enumerate(dept_comparison.itertuples()):
                ax.text(row.인원당수리비 + 100, i, f"{row.인원당수리비:,.0f}원/인", va='center', fontsize=8)
            
            ax.set_xlabel('인원당 수리비 (원)')
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            st.markdown(get_image_download_link(fig, f'{category_name}_소속별_인원당수리비.png', '소속별 인원당수리비 다운로드'), unsafe_allow_html=True)
        else:
            st.info("정비자소속 또는 수리비 정보가 없습니다.")
            
            # 정비자 소속별 정비건수 분석은 수리비 정보가 없을 때만 표시
            if '정비자소속' in df.columns and df4 is not None:
                st.subheader("정비자 소속별 건수")
                
                # 전체 소속별 인원수 계산
                total_staff_by_dept = df4['소속'].value_counts()
                total_staff = len(df4)
                
                # 정비 소속별 건수 및 비율
                dept_counts = df['정비자소속'].value_counts().head(15)
                
                # 소속별 정비 건수 및 인원 비율 계산
                dept_comparison = pd.DataFrame({
                    '소속': dept_counts.index,
                    '정비건수': dept_counts.values,
                    '소속인원수': [total_staff_by_dept.get(dept, 0) for dept in dept_counts.index]
                })
                
                # 인원이 0이면 1로 설정하여 나누기 오류 방지
                dept_comparison['소속인원수'] = dept_comparison['소속인원수'].replace(0, 1)
                
                # 인원당 정비 건수 계산
                dept_comparison['인원당건수'] = (dept_comparison['정비건수'] / dept_comparison['소속인원수']).round(1)
                
                # 결과 소트
                dept_comparison = dept_comparison.sort_values('인원당건수', ascending=False)

                fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                sns.barplot(x=dept_comparison['인원당건수'], y=dept_comparison['소속'], ax=ax, palette="Blues_r")

                # 막대 위에 텍스트 표시 (인원당 건수, 총 건수, 소속 인원수)
                for i, row in enumerate(dept_comparison.itertuples()):
                    ax.text(row.인원당건수 + 0.1, i, 
                           f"{row.인원당건수:.1f}건/인", 
                           va='center', fontsize=8)

                ax.set_xlabel('인원당 정비 건수')
                plt.tight_layout()

                st.pyplot(fig, use_container_width=True)
                st.markdown(get_image_download_link(fig, f'{category_name}_소속별_인원당정비건수.png', '소속별 인원당정비건수 다운로드'), unsafe_allow_html=True)

    st.markdown("---")

# 수리비 대시보드 표시 함수 (수정됨)
def display_repair_cost_dashboard(df):
    if df is None:
        st.warning("수리비 데이터가 없습니다.")
        return

    # 비용 컬럼 찾기
    cost_col = None
    for col in df.columns:
        if '금액' in col or '비용' in col:
            cost_col = col
            break
    
    if not cost_col:
        st.warning("비용 관련 컬럼을 찾을 수 없습니다.")
        return
    
    # 기본 지표 (총 수리비용 제거함)
    col1, col2, col3 = st.columns(3)

    with col1:
        total_cases = len(df)
        st.metric("총 수리 건수", f"{total_cases:,}")

    with col2:
        if '출고일자' in df.columns:
            last_month = df['출고일자'].max().strftime('%Y-%m')
            last_month_count = df[df['출고일자'].dt.strftime('%Y-%m') == last_month].shape[0]
            st.metric("최근 월 수리 건수", f"{last_month_count:,}")
        else:
            st.metric("최근 월 수리 건수", "데이터 없음")
    
    with col3:
        if '출고자소속' in df.columns:
            dept_counts = df['출고자소속'].value_counts()
            top_dept = dept_counts.index[0] if not dept_counts.empty else "정보 없음"
            top_dept_count = dept_counts.iloc[0] if not dept_counts.empty else 0
            st.metric("최다 출고 소속", f"{top_dept} ({top_dept_count}건)")
        else:
            st.metric("최다 출고 소속", "데이터 없음")

    st.markdown("---")

    # 요청대로 월별 수리 건수, 월별 수리 비용, 소속별 수리 비용 현황을 한 줄에 배치
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("월별 수리 건수")
        if '출고일자' in df.columns:
            df_time = df.copy()
            df_time['월'] = df_time['출고일자'].dt.to_period('M')
            monthly_counts = df_time.groupby('월').size().reset_index(name='건수')
            monthly_counts['월'] = monthly_counts['월'].astype(str)

            fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
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
        if '출고일자' in df.columns and cost_col:
            df_time = df.copy()
            df_time['월'] = df_time['출고일자'].dt.to_period('M')
            
            monthly_costs = df_time.groupby('월')[cost_col].sum().reset_index()
            monthly_costs['월'] = monthly_costs['월'].astype(str)
            
            # % 제거
            fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
            sns.barplot(x='월', y=cost_col, data=monthly_costs, ax=ax, palette='Purples')

            # 막대 위에 텍스트 표시 (비용만)
            for i, v in enumerate(monthly_costs[cost_col]):
                ax.text(i, v + max(monthly_costs[cost_col]) * 0.01, f"{v:,.0f}", ha='center', fontsize=8)

            plt.xticks(rotation=45)
            ax.set_ylabel('비용 (원)')
            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)
            st.markdown(get_image_download_link(fig, '월별_수리_비용.png', '월별 수리 비용 다운로드'), unsafe_allow_html=True)
            
    with col3:
        # 소속별 수리비 분석
        if '출고자소속' in df.columns and df4 is not None:
            st.subheader("소속별 수리 비용 현황")
            
            # 전체 소속별 인원수 계산
            total_staff_by_dept = df4['소속'].value_counts()
            total_staff = len(df4)
            
            # 소속별 수리 비용
            dept_costs = df.groupby('출고자소속')[cost_col].sum().sort_values(ascending=False).head(10)
            
            # 소속별 수리 비용 및 인원 비율 계산
            dept_comparison = pd.DataFrame({
                '소속': dept_costs.index,
                '총수리비용': dept_costs.values,
                '소속인원수': [total_staff_by_dept.get(dept, 0) for dept in dept_costs.index]
            })
            
            # 인원이 0이면 1로 설정하여 나누기 오류 방지
            dept_comparison['소속인원수'] = dept_comparison['소속인원수'].replace(0, 1)
            
            # 인원당 수리 비용 계산
            dept_comparison['인원당비용'] = (dept_comparison['총수리비용'] / dept_comparison['소속인원수']).round(0)
            
            # 결과 소트
            dept_comparison = dept_comparison.sort_values('인원당비용', ascending=False)
            
            fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
            sns.barplot(x=dept_comparison['인원당비용'], y=dept_comparison['소속'], ax=ax, palette="Purples_r")
            
            # 막대 위에 텍스트 표시 (인원당 비용, 총 비용, 소속 인원수)
            for i, row in enumerate(dept_comparison.itertuples()):
                ax.text(row.인원당비용 + 100, i, 
                       f"{row.인원당비용:,.0f}원/인)", 
                       va='center', fontsize=8)
            
            ax.set_xlabel('인원당 수리 비용 (원)')
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            st.markdown(get_image_download_link(fig, '소속별_인원당수리비용.png', '소속별 인원당수리비용 다운로드'), unsafe_allow_html=True)
        else:
            st.warning("소속 정보가 없습니다.")
    
    # 고가 부품과 소속별 고가 부품 지출을 한 줄에 배치
    if '모델명' in df.columns and '단가' in df.columns and '출고자소속' in df.columns and '출고금액' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("고가 부품")
            # 모델별 평균 단가 계산
            model_prices = df.groupby('모델명')['단가'].mean().sort_values(ascending=False).head(15)
            
            fig, ax = create_figure_with_korean(figsize=(12, 8), dpi=300)
            sns.barplot(x=model_prices.values, y=model_prices.index, ax=ax, palette="Purples_r")
            
            # 막대 위에 텍스트 표시
            for i, v in enumerate(model_prices.values):
                ax.text(v + max(model_prices.values) * 0.01, i, f"{v:,.0f}원", va='center', fontsize=8)
            
            ax.set_xlabel('평균 단가 (원)')
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            st.markdown(get_image_download_link(fig, '고가부품_Top15.png', '고가부품 Top15 다운로드'), unsafe_allow_html=True)
        
        with col2:
            st.subheader("소속별 고가 부품 지출")
            # Top 5 고가 모델만 필터링
            top_models = model_prices.head(15).index.tolist()
            high_cost_df = df[df['모델명'].isin(top_models)]
            
            # 소속별, 모델별 총 지출 계산
            pivot_data = pd.pivot_table(
                high_cost_df, 
                values='출고금액', 
                index='모델명', 
                columns='출고자소속',
                aggfunc='sum',
                fill_value=0
            ).head(10)
            
            fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
            sns.heatmap(pivot_data, annot=True, fmt=',d', cmap='Purples', ax=ax, linewidths=.5, cbar=False)
            
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            st.markdown(get_image_download_link(fig, '소속별_고가모델_지출.png', '소속별 고가모델 지출 다운로드'), unsafe_allow_html=True)

# 고장 유형 분석 함수 - 수정: 내부/외부/전체 탭 추가
def display_fault_analysis(df, maintenance_type=None):
    # 필터링 (내부/외부/전체)
    if maintenance_type and maintenance_type != "전체":
        filtered_df = df[df['정비구분'] == maintenance_type]
        title_prefix = f"{maintenance_type} "
    else:
        filtered_df = df
        title_prefix = ""

    # 필요한 컬럼 존재 여부 확인
    if all(col in filtered_df.columns for col in ['작업유형', '정비대상', '정비작업', '고장유형']):
        # 탭 구조 정의
        category_tabs = {
            "작업유형": "작업유형",
            "정비대상": "정비대상",
            "정비작업": "정비작업"
        }

        tabs = st.tabs(list(category_tabs.keys()))

        current_theme = color_themes["고장 유형 분석"]

        for tab, colname in zip(tabs, category_tabs.values()):
            with tab:
                st.subheader(f"{title_prefix}{colname}")
                category_counts = filtered_df[colname].value_counts().head(15)
                category_values = convert_to_str_list(filtered_df[colname].unique())
                selected_category = st.selectbox(f"{colname} 선택", ["전체"] + sorted(category_values), key=f"sel_{colname}_{maintenance_type}")

                if selected_category != "전체":
                    tab_filtered_df = filtered_df[filtered_df[colname].astype(str) == selected_category]
                else:
                    tab_filtered_df = filtered_df

                top_faults = tab_filtered_df['고장유형'].value_counts().nlargest(15).index
                df_filtered = tab_filtered_df[tab_filtered_df['고장유형'].isin(top_faults)]
                top_combos = df_filtered['브랜드_모델'].value_counts().nlargest(15).index
                df_filtered = df_filtered[df_filtered['브랜드_모델'].isin(top_combos)]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**{colname} 분포**")
                    fig1, ax1 = create_figure_with_korean(figsize=(8, 8), dpi=300)
                    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax1, palette=f"{current_theme}_r")
                    plt.xticks(rotation=45, ha='right')
                    for i, v in enumerate(category_counts.values):
                        ax1.text(i, v + max(category_counts.values) * 0.01, str(v), ha='center', fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig1, use_container_width=True)
                    st.markdown(get_image_download_link(fig1, f'{title_prefix}고장유형_{colname}_분포.png', f'{colname} 분포 다운로드'), unsafe_allow_html=True)

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
                    st.markdown(get_image_download_link(fig2, f'{title_prefix}고장유형_{colname}_비율.png', f'{colname} 비율 다운로드'), unsafe_allow_html=True)

                with col3:
                    st.markdown(f"**{colname}에 따른 고장 증상**")
                    try:
                        pivot_df = df_filtered.pivot_table(
                            index='고장유형',
                            columns='브랜드_모델',
                            aggfunc='size',
                            fill_value=0
                        )
                        fig3, ax3 = create_figure_with_korean(figsize=(8, 8), dpi=300)
                        sns.heatmap(pivot_df, cmap=current_theme, annot=True, fmt='d', linewidths=0.5, ax=ax3, cbar=False)
                        plt.xticks(rotation=90)
                        plt.yticks(rotation=0)
                        plt.tight_layout()
                        st.pyplot(fig3, use_container_width=True)
                        st.markdown(get_image_download_link(fig3, f'{title_prefix}고장유형_{colname}_히트맵.png', f'{colname} 히트맵 다운로드'), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"히트맵 생성 중 오류가 발생했습니다: {e}")
                        st.info("선택한 필터에 맞는 데이터가 충분하지 않을 수 있습니다.")

        # 자재내역 분석 섹션 추가 - 자재내역 컬럼이 있는 경우만 표시
        st.subheader(f"{title_prefix}모델 타입 분석")
        
        # 탭으로 분석 항목 구분
        tabs = st.tabs(["연료", "운전방식", "적재용량", "마스트"])
        
        # 연료별 분석
        with tabs[0]:
            if '연료' in filtered_df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # 연료별 AS 건수
                    st.subheader(f"{title_prefix}연료별 AS 건수")
                    fuel_type_counts = filtered_df['연료'].value_counts().dropna()
                    
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
                        st.markdown(get_image_download_link(fig, f'{title_prefix}연료별_AS_건수.png', '연료별 AS 건수 다운로드'), unsafe_allow_html=True)
                    else:
                        st.warning("연료 데이터가 없습니다.")
                
                with col2:
                    # 연료별 고장유형 Top 10
                    st.subheader(f"{title_prefix}연료별 고장유형")
                    
                    # 연료 선택
                    fuel_types = ["전체"] + filtered_df['연료'].value_counts().index.tolist()
                    selected_fuel = st.selectbox("연료", fuel_types, key=f"fuel_{maintenance_type}")
                    
                    if selected_fuel != "전체":
                        filtered_df_fuel = filtered_df[filtered_df['연료'] == selected_fuel]
                    else:
                        filtered_df_fuel = filtered_df
                        
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
                        st.markdown(get_image_download_link(fig, f'{title_prefix}{selected_fuel}_고장유형_TOP10.png', f'{selected_fuel} 고장유형 TOP10 다운로드'), unsafe_allow_html=True)
                    else:
                        st.warning("고장유형 데이터가 없습니다.")
            else:
                st.warning("연료 데이터가 없습니다.")
        
        # 운전방식별 분석 (tabs[1]), 적재용량별 분석 (tabs[2]), 마스트별 분석 (tabs[3])은 
        # 연료별 분석과 비슷한 구조로 구현할 수 있습니다.
        
        # 상위 고장 유형 리스트
        st.subheader(f"{title_prefix}상위 고장 유형")
        top_40_faults = filtered_df['고장유형'].value_counts().nlargest(40)
        fault_df = pd.DataFrame({
            '고장유형': top_40_faults.index,
            '건수': top_40_faults.values
        })
        st.dataframe(fault_df)
    else:
        st.warning("고장 유형 분석에 필요한 컬럼(작업유형, 정비대상, 정비작업)이 데이터에 없습니다.")

# 정비내용 분석 함수
def display_maintenance_text_analysis(df, maintenance_type=None):
    # 데이터 필터링 (정비구분에 따라)
    if maintenance_type and maintenance_type != "전체":
        filtered_df = df[df['정비구분'] == maintenance_type]
        title_prefix = f"{maintenance_type} "
    else:
        filtered_df = df
        title_prefix = ""
    
    # 텍스트 데이터 확인
    if '정비내용' in filtered_df.columns:
        # 정비내용 데이터 준비
        from kiwipiepy import Kiwi
        kiwi = Kiwi()

        text_data_raw = ' '.join(filtered_df['정비내용'].dropna().astype(str))

        # 형태소 분석 + 명사 추출
        tokens = kiwi.tokenize(text_data_raw)
        nouns = [token.form for token in tokens if token.tag.startswith('N')]

        stopwords = ["및", "있음", "없음", "함", "을", "후", "함", "접수", "취소", "확인", "위해", "통해", "오류", "완료", "작업", "실시", "진행", "수리", '정상작동', '정상작동확인', '조치완료']

        # 불용어 제거
        filtered_nouns = [word for word in nouns if word not in stopwords and len(word) > 1]

        # 단어 빈도 계산
        from collections import Counter
        word_counts = Counter(filtered_nouns)
        
        # 상위 100개 단어만 선별
        top_100_words = dict(word_counts.most_common(100))
        
        # 워드클라우드용 문자열 생성 (상위 100개 단어만으로)
        text_data = ' '.join([word for word in filtered_nouns if word in top_100_words])

        if not text_data:
            st.warning(f"{title_prefix}정비내용 데이터가 없습니다.")
        else:
            # 그래프 행 1: 전체 워드클라우드와 분류별 워드클라우드
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"{title_prefix}정비내용 워드클라우드")

                try:
                    # 워드클라우드 생성
                    wordcloud = WordCloud(
                        width=1200, 
                        height=800,
                        background_color='white',
                        font_path=font_path,  
                        colormap=current_theme,
                        max_words=100,  # 최대 100개 단어만 표시
                        stopwords=set(stopwords),
                        min_font_size=10,
                        max_font_size=150,
                        random_state=42
                    ).generate_from_frequencies(top_100_words)

                    # 워드클라우드 시각화
                    fig, ax = create_figure_with_korean(figsize=(10, 10), dpi=300)
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    plt.tight_layout()

                    st.pyplot(fig, use_container_width=True)

                    # 다운로드 링크 추가
                    st.markdown(get_image_download_link(fig, f'{title_prefix}정비내용_워드클라우드.png', '정비내용 워드클라우드 다운로드'), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"워드클라우드 생성 중 오류가 발생했습니다: {e}")
                    if "font_path" in str(e).lower():
                        st.info("한글 폰트 경로를 확인해주세요.")

            with col2:
                # 주요 단어 표시
                st.subheader(f"{title_prefix}주요 단어 Top 30")
                word_df = pd.DataFrame({
                    '단어': list(top_100_words.keys())[:30],
                    '빈도': list(top_100_words.values())[:30]
                })

                fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                sns.barplot(x=word_df['빈도'], y=word_df['단어'], ax=ax, palette=f"{current_theme}_r")
                plt.tight_layout()

                st.pyplot(fig, use_container_width=True)

                # 다운로드 링크 추가
                st.markdown(get_image_download_link(fig, f'{title_prefix}주요단어_TOP30.png', '주요단어 TOP30 다운로드'), unsafe_allow_html=True)

            # 분류별 정비내용 워드클라우드
            st.subheader(f"{title_prefix}분류별 정비내용 워드클라우드")

            if all(col in filtered_df.columns for col in ['작업유형', '정비대상', '정비작업', '정비내용']):
                col1, col2, col3 = st.columns(3)

                with col1:
                    # 문자열 변환으로 정렬 오류 방지
                    categories = ["전체"] + sorted(convert_to_str_list(filtered_df['작업유형'].dropna().unique()))
                    selected_category = st.selectbox("작업유형", categories, key=f"text_cat_{maintenance_type}")

                # 선택된 작업유형에 따라 데이터 필터링
                if selected_category != "전체":
                    text_filtered_df = filtered_df[filtered_df['작업유형'].astype(str) == selected_category]

                    with col2:
                        subcategories = ["전체"] + sorted(convert_to_str_list(text_filtered_df['정비대상'].dropna().unique()))
                        selected_subcategory = st.selectbox("정비대상", subcategories, key=f"text_subcat_{maintenance_type}")

                    # 선택된 정비대상에 따라 추가 필터링
                    if selected_subcategory != "전체":
                        text_filtered_df = text_filtered_df[text_filtered_df['정비대상'].astype(str) == selected_subcategory]

                        with col3:
                            detailed_categories = ["전체"] + sorted(convert_to_str_list(text_filtered_df['정비작업'].dropna().unique()))
                            selected_detailed = st.selectbox("정비작업", detailed_categories, key=f"text_detail_{maintenance_type}")

                        # 선택된 정비작업에 따라 최종 필터링
                        if selected_detailed != "전체":
                            text_filtered_df = text_filtered_df[text_filtered_df['정비작업'].astype(str) == selected_detailed]
                    else:
                        selected_detailed = "전체"
                        with col3:
                            st.selectbox("정비작업", ["전체"], key=f"text_detail_empty_{maintenance_type}")
                else:
                    text_filtered_df = filtered_df
                    selected_subcategory = "전체"
                    selected_detailed = "전체"

                    with col2:
                        st.selectbox("정비대상", ["전체"], key=f"text_subcat_empty_{maintenance_type}")

                    with col3:
                        st.selectbox("정비작업", ["전체"], key=f"text_detail_empty2_{maintenance_type}")

                # 필터링된 정비내용 결합 및 명사 추출
                raw_filtered_text = ' '.join(text_filtered_df['정비내용'].dropna().astype(str))
                
                if not raw_filtered_text:
                    st.warning(f"선택한 분류에 대한 {title_prefix}정비내용 데이터가 없습니다.")
                else:
                    # 형태소 분석 및 명사 추출
                    filtered_tokens = kiwi.tokenize(raw_filtered_text)
                    filtered_nouns = [token.form for token in filtered_tokens if token.tag.startswith('N') and token.form not in stopwords and len(token.form) > 1]
                    
                    # 단어 빈도 계산
                    filtered_word_counts = Counter(filtered_nouns)
                    
                    # 상위 100개 단어만 선별
                    filtered_top_100_words = dict(filtered_word_counts.most_common(100))
                    
                    st.write(f"선택: {selected_category} > {selected_subcategory} > {selected_detailed}")
                    st.write(f"선택된 AS 건수: {len(text_filtered_df)}")

                    col1, col2 = st.columns(2)

                    with col1:
                        try:
                            # 워드클라우드 생성 (상위 100개 단어만)
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
                            ).generate_from_frequencies(filtered_top_100_words)

                            # 워드클라우드 시각화
                            fig, ax = create_figure_with_korean(figsize=(10, 10), dpi=300)
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            plt.tight_layout()

                            st.pyplot(fig, use_container_width=True)

                            # 다운로드 링크 추가
                            st.markdown(get_image_download_link(fig, f'{title_prefix}{selected_category}_{selected_subcategory}_{selected_detailed}_워드클라우드.png', 
                                    '분류별 워드클라우드 다운로드'), unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"워드클라우드 생성 중 오류가 발생했습니다: {e}")

                    with col2:
                        # 주요 단어 표시
                        st.subheader("주요 단어 Top 30")
                        word_df = pd.DataFrame({
                            '단어': list(filtered_top_100_words.keys())[:30],
                            '빈도': list(filtered_top_100_words.values())[:30]
                        })

                        fig, ax = create_figure_with_korean(figsize=(10, 8), dpi=300)
                        sns.barplot(x=word_df['빈도'], y=word_df['단어'], ax=ax, palette=f"{current_theme}_r")
                        plt.tight_layout()

                        st.pyplot(fig, use_container_width=True)

                        # 다운로드 링크 추가
                        st.markdown(get_image_download_link(fig, f'{title_prefix}{selected_category}_{selected_subcategory}_{selected_detailed}_주요단어.png', 
                                '분류별 주요단어 다운로드'), unsafe_allow_html=True)
                        
            else:
                st.warning("분류별 분석에 필요한 컬럼이 데이터에 없습니다.")
    else:
        st.warning("정비내용 컬럼이 데이터에 없습니다.")

# 데이터가 로드된 경우 분석 시작
if df1 is not None or df3 is not None:
    # 메뉴 선택 - 정비일지 대시보드로 통합
    menu_options = []

    if df1 is not None:
        menu_options.append("정비일지 대시보드")  # 통합된 대시보드
        menu_options.extend(["고장 유형 분석", "브랜드/모델 분석", "정비내용 분석", "고장 예측"])

    # 원하는 순서대로 메뉴를 정렬
    sorted_menu_options = []
    desired_order = ["정비일지 대시보드", "고장 유형 분석", "브랜드/모델 분석", "정비내용 분석", "고장 예측"]

    for item in desired_order:
        if item in menu_options:
            sorted_menu_options.append(item)

    menu = st.sidebar.selectbox("분석 메뉴", sorted_menu_options)

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

    elif menu == "고장 유형 분석":
        st.title("고장 유형 분석")

        # 정비구분 컬럼 확인 및 값 검증
        if '정비구분' in df1.columns and df1['정비구분'].notna().any():
            # 실제 존재하는 정비구분 값 확인
            maintenance_types = df1['정비구분'].dropna().unique()
            has_internal = '내부' in maintenance_types
            has_external = '외부' in maintenance_types
            
            # 탭 생성
            tabs = st.tabs(["전체", "내부", "외부"])
            
            # 전체 탭
            with tabs[0]:
                st.header("전체 고장 유형 분석")
                display_fault_analysis(df1, None)
            
            # 내부 탭
            with tabs[1]:
                st.header("내부 고장 유형 분석")
                if has_internal:
                    display_fault_analysis(df1, "내부")
                else:
                    st.info("내부 정비 데이터가 없습니다.")
            
            # 외부 탭
            with tabs[2]:
                st.header("외부 고장 유형 분석")
                if has_external:
                    display_fault_analysis(df1, "외부")
                else:
                    st.info("외부 정비 데이터가 없습니다.")
        else:
            # 정비구분 컬럼이 없는 경우 전체 데이터만
            display_fault_analysis(df1, None)

    elif menu == "브랜드/모델 분석":
        st.title("브랜드 및 모델 분석")

        # 자산 데이터 확인
        has_asset_data = df2 is not None and '제조사명' in df2.columns and '제조사모델명' in df2.columns

        if has_asset_data:
            # 자산 데이터 기본 정보
            total_assets = len(df2)
            asset_brand_counts = df2['제조사명'].value_counts()
            asset_model_counts = df2['제조사모델명'].value_counts()
            asset_year_counts = df2['제조년도'].value_counts()

            # 비율 계산
            asset_brand_ratio = asset_brand_counts / total_assets * 100
            asset_model_ratio = asset_model_counts / total_assets * 100
            asset_year_ratio = asset_year_counts / total_assets * 100
        else:
            st.warning("자산조회 파일이 업로드되지 않았거나 필요한 컬럼이 없습니다. AS 건수만 표시합니다.")

        # AS 데이터 전체 건수
        total_as = len(df1)

        # 섹션 1: 브랜드 분석
        st.header("브랜드 분석")

        # 브랜드별 AS 비율 계산
        brand_counts = df1['브랜드'].value_counts()
        brand_as_ratio = brand_counts / total_as * 100
        brand_as_ratio = group_small_categories(brand_as_ratio, threshold=0.03)
        brand_as_ratio = brand_as_ratio.nlargest(15)  # 상위 15개만 표시

        # 브랜드 분석 그래프 - 세 개의 그래프를 나란히 배치
        col1, col2, col3 = st.columns(3)

        with col1:
            # 브랜드별 AS 비율 파이 차트
            st.subheader("브랜드별 AS 비율")

            fig, ax = create_figure_with_korean(figsize=(6, 6), dpi=200)
            wedges, texts, autotexts = ax.pie(
                brand_as_ratio.values, 
                labels=brand_as_ratio.index, 
                autopct='%1.1f%%',
                textprops={'fontsize': 8},
                colors=sns.color_palette(f"{current_theme}_r", n_colors=len(brand_as_ratio))
            )

            # 레이블 가독성 향상
            for text in texts:
                text.set_fontsize(8)
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_color('white')

            plt.tight_layout()
            st.pyplot(fig)

            # 다운로드 링크
            st.markdown(get_image_download_link(fig, '브랜드_AS_비율_파이.png', '파이차트 다운로드'), unsafe_allow_html=True)

        # 자산 데이터가 있는 경우 특정 브랜드에 대한 비교 그래프 표시
        if has_asset_data:
            # 특정 브랜드만 선택
            selected_brands = ['도요타', '클라크', '두산', '현대', '니찌유', '비와이디']

            # 선택된 브랜드에 대한 데이터만 필터링
            filtered_brands = []
            for brand in selected_brands:
                if brand in brand_as_ratio.index and brand in asset_brand_ratio.index:
                    filtered_brands.append(brand)

            if filtered_brands:
                # 브랜드별 자산 비교 데이터 준비
                brand_comparison = pd.DataFrame({
                    '브랜드': filtered_brands,
                    'AS 비율(%)': [brand_as_ratio.get(brand, 0) for brand in filtered_brands],
                    '자산 비율(%)': [asset_brand_ratio.get(brand, 0) for brand in filtered_brands]
                })


                # AS/자산 비율 계산 (0으로 나누기 방지)
                brand_comparison['자산 비율(%)'] = brand_comparison['자산 비율(%)'].replace(0, 0.1)
                brand_comparison['AS/자산 비율'] = (brand_comparison['AS 비율(%)'] / brand_comparison['자산 비율(%)']).round(2)

                with col2:
                    # 간단한 비교 막대 그래프
                    st.subheader("주요 브랜드 비율 비교")

                    fig, ax = create_figure_with_korean(figsize=(6, 6), dpi=200)

                    x = np.arange(len(brand_comparison))
                    width = 0.4

                    # 막대 그래프 (두 값 나란히)
                    ax.bar(x - width/2, brand_comparison['자산 비율(%)'], width, 
                        label='자산 비율(%)', color='#8ECAE6')
                    ax.bar(x + width/2, brand_comparison['AS 비율(%)'], width, 
                        label='AS 비율(%)', color='#FB8500')

                    # 축 설정
                    ax.set_xticks(x)
                    ax.set_xticklabels(brand_comparison['브랜드'], fontsize=8)
                    ax.legend(fontsize=8)

                    # 값 표시
                    for i, v in enumerate(brand_comparison['자산 비율(%)']):
                        ax.text(i - width/2, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=7)

                    for i, v in enumerate(brand_comparison['AS 비율(%)']):
                        ax.text(i + width/2, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=7)

                    plt.tight_layout()
                    st.pyplot(fig)

                    # 다운로드 링크
                    st.markdown(get_image_download_link(fig, '주요_브랜드_비율_비교.png', '비교 그래프 다운로드'), unsafe_allow_html=True)

                with col3:
                    # AS/자산 비율 그래프
                    st.subheader("주요 브랜드 AS/자산 비율")

                    fig, ax = create_figure_with_korean(figsize=(6, 6), dpi=200)

                    # 기준선 추가 (1.0 = 자산 대비 AS 비율이 동일)
                    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7)

                    # 색상 구분 (1보다 크면 빨간색 계열, 작으면 파란색 계열)
                    colors = ['#FB8500' if x > 1 else '#8ECAE6' for x in brand_comparison['AS/자산 비율']]

                    # 막대 그래프
                    bars = ax.bar(brand_comparison['브랜드'], brand_comparison['AS/자산 비율'], color=colors)

                    # 값 표시
                    for i, v in enumerate(brand_comparison['AS/자산 비율']):
                        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

                    plt.xticks(fontsize=8)
                    ax.set_ylabel('AS/자산 비율', fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # 다운로드 링크
                    st.markdown(get_image_download_link(fig, '주요_브랜드_AS자산_비율.png', 'AS/자산 비율 다운로드'), unsafe_allow_html=True)

        # 간단한 설명 추가 (자산 데이터가 있는 경우)
        if has_asset_data:
            st.info("""
            **자산 대비 AS 비율 해석**
            - 1.0보다 큰 값: 자산 비율보다 AS 비율이 더 높음 (AS 발생 빈도가 상대적으로 높음)
            - 1.0과 같은 값: 자산 비율과 AS 비율이 동일함
            - 1.0보다 작은 값: 자산 비율보다 AS 비율이 더 낮음 (AS 발생 빈도가 상대적으로 낮음)
            """)

        # 섹션 2: 모델 분석
        st.header("모델 분석")

        # 브랜드 선택 목록 준비 (우선순위 브랜드 + 빈도순)
        priority_brands = ['도요타', '두산', '현대', '클라크']
        brand_list = ["전체"] + [brand for brand in priority_brands if brand in brand_counts.index]

        # 나머지 브랜드 추가 (상위 20개)
        other_top_brands = [b for b in brand_counts.nlargest(20).index 
                        if b not in priority_brands and b != '기타']
        brand_list.extend(other_top_brands)

        # '기타' 추가
        if '기타' in brand_counts.index:
            brand_list.append('기타')

        # 브랜드 선택
        selected_brand = st.selectbox("브랜드 선택", brand_list)

        # 선택된 브랜드에 따른 데이터 필터링
        if selected_brand != "전체":
            brand_df = df1[df1['브랜드'] == selected_brand]
            brand_total_as = len(brand_df)
            brand_title = f"{selected_brand} "
        else:
            brand_df = df1
            brand_total_as = total_as
            brand_title = "전체 "

        # 모델별 AS 건수 및 비율
        model_counts = brand_df['모델명'].value_counts().head(15)  # 상위 10개만 (가독성)
        model_as_ratio = (model_counts / brand_total_as * 100).round(1)

        # 모델별 분석 그래프 (2개 그래프 나란히 배치)
        col1, col2 = st.columns(2)

        with col1:
            # 모델별 AS 비율 그래프
            st.subheader(f"{brand_title}모델별 AS 비율")

            fig, ax = create_figure_with_korean(figsize=(7, 6), dpi=200)
            bars = sns.barplot(x=model_as_ratio.values, y=model_as_ratio.index, ax=ax, palette=f"{current_theme}_r")

            # 간결한 값 표시
            for i, v in enumerate(model_as_ratio.values):
                ax.text(v + 0.01, i, f"{v:.1f}%", va='center', fontsize=8)

            ax.set_xlabel('AS 비율 (%)')
            plt.tight_layout()
            st.pyplot(fig)

            # 다운로드 링크
            st.markdown(get_image_download_link(fig, f'{selected_brand}_모델별_AS_비율.png', 
                                            '모델별 AS 비율 다운로드'), unsafe_allow_html=True)

        with col2:
            # 고장 유형 분석 (있는 경우)
            if '고장유형' in brand_df.columns:
                st.subheader(f"{brand_title}고장 유형 분석")

                fault_counts = brand_df['고장유형'].value_counts().head(15)
                fault_ratio = (fault_counts / brand_total_as * 100).round(1)

                fig, ax = create_figure_with_korean(figsize=(7, 6), dpi=200)
                bars = sns.barplot(x=fault_ratio.values, y=fault_ratio.index, ax=ax, palette=f"{current_theme}_r")

                # 간결한 값 표시
                for i, v in enumerate(fault_ratio.values):
                    ax.text(v + 0.01, i, f"{v:.1f}%", va='center', fontsize=8)

                ax.set_xlabel('고장유형 비율 (%)')
                plt.tight_layout()
                st.pyplot(fig)

                # 다운로드 링크
                st.markdown(get_image_download_link(fig, f'{selected_brand}_고장유형_분석.png', 
                                                '고장유형 분석 다운로드'), unsafe_allow_html=True)

        # 자산 데이터가 있는 경우 모델별 자산 대비 AS 비율 분석
        if has_asset_data:
            st.subheader(f"{brand_title}모델별 자산 대비 AS 비율")

            # 자산 데이터 필터링
            if selected_brand != "전체":
                asset_brand_df = df2[df2['제조사명'] == selected_brand]
                brand_total_assets = len(asset_brand_df)
            else:
                asset_brand_df = df2
                brand_total_assets = total_assets

            if brand_total_assets > 0:
                # 모델별 자산 비율 계산
                asset_model_counts = asset_brand_df['제조사모델명'].value_counts()
                asset_model_ratio = (asset_model_counts / brand_total_assets * 100).round(1)

                # 모델별 비교 데이터 준비
                model_comparison = pd.DataFrame({
                    '모델명': model_as_ratio.index,
                    'AS 비율(%)': model_as_ratio.values,
                    '자산 비율(%)': [asset_model_ratio.get(model, 0) for model in model_as_ratio.index]
                })

                # AS/자산 비율 계산 (0으로 나누기 방지)
                model_comparison['자산 비율(%)'] = model_comparison['자산 비율(%)'].replace(0, 0.1)
                model_comparison['AS/자산 비율'] = (model_comparison['AS 비율(%)'] / model_comparison['자산 비율(%)']).round(2)

                # 2개 그래프 나란히 배치
                col1, col2 = st.columns(2)

                with col1:
                    # 모델별 AS 및 자산 비율 비교 (수평 막대)
                    fig, ax = create_figure_with_korean(figsize=(8, 7), dpi=200)

                    # 데이터 준비 
                    top_models = model_comparison.head(8)

                    # 수평 막대 그래프 (두 값 나란히)
                    y_pos = np.arange(len(top_models))
                    width = 0.4

                    # 자산 비율 막대
                    ax.barh(y_pos - width/2, top_models['자산 비율(%)'], width, 
                        label='자산 비율', color='#8ECAE6')

                    # AS 비율 막대
                    ax.barh(y_pos + width/2, top_models['AS 비율(%)'], width, 
                        label='AS 비율', color='#FB8500')

                    # 축 및 레이블 설정
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(top_models['모델명'], fontsize=9)
                    ax.legend(fontsize=10)

                    # 값 표시 (간결하게)
                    for i, v in enumerate(top_models['자산 비율(%)']):
                        ax.text(v + 0.02, i - width/2, f"{v:.1f}%", va='center', fontsize=9)

                    for i, v in enumerate(top_models['AS 비율(%)']):
                        ax.text(v + 0.02, i + width/2, f"{v:.1f}%", va='center', fontsize=9)

                    plt.tight_layout()
                    st.pyplot(fig)

                    # 다운로드 링크
                    st.markdown(get_image_download_link(fig, f'{selected_brand}_모델_비율_비교.png', 
                                                    '모델별 비율 비교 다운로드'), unsafe_allow_html=True)

                with col2:
                    # AS/자산 비율 그래프
                    fig, ax = create_figure_with_korean(figsize=(8, 7), dpi=200)

                    # 기준선 추가
                    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)

                    # 색상 구분 (1보다 크면 빨간색 계열, 작으면 파란색 계열)
                    colors = ['#FB8500' if x > 1 else '#8ECAE6' for x in top_models['AS/자산 비율']]

                    # 수평 막대 그래프
                    bars = ax.barh(top_models['모델명'], top_models['AS/자산 비율'], color=colors)

                    # 값 표시 (간결하게)
                    for i, v in enumerate(top_models['AS/자산 비율']):
                        ax.text(v + 0.02, i, f"{v:.1f}", va='center', fontsize=9)

                    ax.set_xlabel('AS/자산 비율 (1.0 = 동일 비율)')
                    plt.tight_layout()
                    st.pyplot(fig)

                    # 다운로드 링크
                    st.markdown(get_image_download_link(fig, f'{selected_brand}_모델_AS자산_비율.png', 
                                                    '모델별 AS/자산 비율 다운로드'), unsafe_allow_html=True)

        # 섹션 3: 제조년도별 분석
        if '제조년도' in df1.columns:
            st.header("제조년도별 분석")

            # 제조년도 데이터 필터링 및 정리
            year_df = brand_df.dropna(subset=['제조년도'])

            if len(year_df) > 0:
                # 제조년도를 정수로 변환하고 정렬
                year_df['제조년도'] = year_df['제조년도'].astype(int)
                year_counts = year_df['제조년도'].value_counts().sort_index()
                year_ratio = (year_counts / brand_total_as * 100).round(1)

                # 그래프 2개 나란히 배치
                col1, col2 = st.columns(2)

                with col1:
                    # 제조년도별 AS 비율
                    st.subheader(f"{brand_title}연식별 AS 비율")

                    fig, ax = create_figure_with_korean(figsize=(7, 6), dpi=200)
                    bars = sns.barplot(x=year_ratio.index.astype(str), y=year_ratio.values, 
                                    ax=ax, palette=f"{current_theme}")

                    # 간결한 값 표시
                    for i, v in enumerate(year_ratio.values):
                        ax.text(i, v + 0.2, f"{v:.1f}%", ha='center', fontsize=5)

                    plt.xticks(rotation=45)
                    ax.set_ylabel('AS 비율 (%)')
                    plt.tight_layout()
                    st.pyplot(fig)

                    # 다운로드 링크
                    st.markdown(get_image_download_link(fig, f'{selected_brand}_연식별_AS_비율.png', 
                                                    '연식별 AS 비율 다운로드'), unsafe_allow_html=True)

                with col2:
                    # 제조년도별 평균 처리일수 (있는 경우)
                    if 'AS처리일수' in df1.columns:
                        st.subheader(f"{brand_title}연식별 평균 처리일수")

                        # 필요한 데이터만 추출
                        days_df = year_df.dropna(subset=['AS처리일수'])

                        if len(days_df) > 0:
                            # 제조년도별 평균 처리일수 계산
                            year_avg_days = days_df.groupby('제조년도')['AS처리일수'].mean().round(1)

                            fig, ax = create_figure_with_korean(figsize=(7, 6), dpi=200)
                            bars = sns.barplot(x=year_avg_days.index.astype(str), y=year_avg_days.values, 
                                            ax=ax, palette=f"{current_theme}_r")

                            # 간결한 값 표시
                            for i, v in enumerate(year_avg_days.values):
                                ax.text(i, v + 0.02, f"{v:.1f}일", ha='center', fontsize=4)

                            plt.xticks(rotation=45)
                            ax.set_ylabel('평균 처리일수')
                            plt.tight_layout()
                            st.pyplot(fig)

                            # 다운로드 링크
                            st.markdown(get_image_download_link(fig, f'{selected_brand}_연식별_처리일수.png', 
                                                            '연식별 처리일수 다운로드'), unsafe_allow_html=True)

                # 자산 데이터가 있는 경우 제조년도별 자산 대비 AS 비율 분석
                if has_asset_data:
                    st.subheader(f"{brand_title}연식별 자산 대비 AS 비율")

                    # 자산 데이터 필터링
                    if selected_brand != "전체":
                        asset_year_df = df2[df2['제조사명'] == selected_brand]
                    else:
                        asset_year_df = df2

                    if len(asset_year_df) > 0:
                        # 제조년도별 자산 비율 계산
                        asset_year_counts = asset_year_df['제조년도'].value_counts().sort_index()
                        asset_year_ratio = (asset_year_counts / len(asset_year_df) * 100).round(1)

                        # 공통 연식만 추출
                        common_years = sorted(set(year_ratio.index) & set(asset_year_ratio.index))

                        if common_years:
                            # 연식별 비교 데이터 준비
                            year_comparison = pd.DataFrame({
                                '제조년도': common_years,
                                'AS 비율(%)': [year_ratio.get(year, 0) for year in common_years],
                                '자산 비율(%)': [asset_year_ratio.get(year, 0) for year in common_years]
                            })

                            # AS/자산 비율 계산
                            year_comparison['자산 비율(%)'] = year_comparison['자산 비율(%)'].replace(0, 0.1)
                            year_comparison['AS/자산 비율'] = (year_comparison['AS 비율(%)'] / year_comparison['자산 비율(%)']).round(2)

                            # 2개 그래프 나란히 배치
                            col1, col2 = st.columns(2)

                            with col1:
                                # 연식별 AS 및 자산 비율 비교
                                fig, ax = create_figure_with_korean(figsize=(8, 6), dpi=200)

                                x = np.arange(len(common_years))
                                width = 0.4

                                # 막대 그래프 (두 값 나란히)
                                ax.bar(x - width/2, year_comparison['자산 비율(%)'], width, 
                                    label='자산 비율', color='#8ECAE6')
                                ax.bar(x + width/2, year_comparison['AS 비율(%)'], width, 
                                    label='AS 비율', color='#FB8500')

                                # 축 설정
                                ax.set_xticks(x)
                                ax.set_xticklabels(year_comparison['제조년도'].astype(str), rotation=45)
                                ax.legend(fontsize=10)

                                # 간결한 값 표시
                                for i, v in enumerate(year_comparison['자산 비율(%)']):
                                    ax.text(i - width/2, v + 0.1, f"{v:.1f}%", ha='center', fontsize=4)

                                for i, v in enumerate(year_comparison['AS 비율(%)']):
                                    ax.text(i + width/2, v + 0.4, f"{v:.1f}%", ha='center', fontsize=4)

                                plt.tight_layout()
                                st.pyplot(fig)

                                # 다운로드 링크
                                st.markdown(get_image_download_link(fig, f'{selected_brand}_연식별_비율_비교.png', 
                                                                '연식별 비율 비교 다운로드'), unsafe_allow_html=True)

                            with col2:
                                # AS/자산 비율 그래프
                                fig, ax = create_figure_with_korean(figsize=(8, 6), dpi=200)

                                # 기준선 추가
                                plt.axhline(y=1, color='red', linestyle='--', alpha=0.7)

                                # 색상 구분
                                colors = ['#FB8500' if x > 1 else '#8ECAE6' for x in year_comparison['AS/자산 비율']]

                                # 막대 그래프
                                bars = ax.bar(year_comparison['제조년도'].astype(str), 
                                            year_comparison['AS/자산 비율'], color=colors)

                                # 간결한 값 표시
                                for i, v in enumerate(year_comparison['AS/자산 비율']):
                                    ax.text(i, v + 0.02, f"{v:.1f}", ha='center', fontsize=9)

                                plt.xticks(rotation=45)
                                ax.set_ylabel('AS/자산 비율 (1.0 = 동일 비율)')
                                plt.tight_layout()
                                st.pyplot(fig)

                                # 다운로드 링크
                                st.markdown(get_image_download_link(fig, f'{selected_brand}_연식별_AS자산_비율.png', 
                                                                '연식별 AS/자산 비율 다운로드'), unsafe_allow_html=True)

    elif menu == "정비내용 분석":
        st.title("정비내용 분석")

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
                st.header("전체 정비내용 분석")
                display_maintenance_text_analysis(df1, None)
                
            # 내부 탭
            with tabs[1]:
                st.header("내부 정비내용 분석")
                if has_internal:
                    display_maintenance_text_analysis(df1, "내부")
                else:
                    st.info("내부 정비 데이터가 없습니다.")
                    
            # 외부 탭
            with tabs[2]:
                st.header("외부 정비내용 분석")
                if has_external:
                    display_maintenance_text_analysis(df1, "외부")
                else:
                    st.info("외부 정비 데이터가 없습니다.")
        else:
            # 정비구분 컬럼이 없는 경우 전체 데이터만
            display_maintenance_text_analysis(df1, None)

    elif menu == "고장 예측":
            st.title("고장 예측")

            @st.cache_resource
            def prepare_prediction_model(df1):
                try:
                    # 필수 컬럼 체크 - AS처리일수 제거
                    required_cols = ['브랜드', '모델명', '작업유형', '정비대상', '정비작업', '제조년도']
                    if not all(col in df1.columns for col in required_cols):
                        return [None] * 10

                    model_df = df1.dropna(subset=required_cols[:-1]).copy()  # 제조년도는 후처리로

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

                    # 피처 - AS처리일수 제거
                    features = ['브랜드_인코딩', '모델_인코딩', '작업유형_인코딩', '정비대상_인코딩', '정비작업_인코딩', '제조년도_구간_인코딩']

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
        
            # 모델 준비 (백그라운드에서 실행) - df를 df1로 변경
            interval_model, category_model, subcategory_model, detail_model, le_brand, le_model, le_category, le_subcategory, le_detail, le_year_range = prepare_prediction_model(df1)
            
            # 모델이 준비되었는지 확인하는 명시적인 검사 추가
            models_ready = interval_model is not None
            
            if models_ready:
                st.info("다음 고장 시기 예측과 확률이 높은 고장 유형을 예측합니다.")
            else:
                st.warning("""
                예측 모델을 준비할 수 없습니다. 다음 사항을 확인해주세요:
                1. 충분한 데이터(최소 100개 이상의 기록)가 있는지 확인
                2. 필요한 컬럼(브랜드, 모델명, 작업유형, 정비대상, 정비작업)이 모두 있는지 확인
                3. 재정비 간격 정보가 있는지 확인
                """)
                
            col1, col2, col3, col4, col5 = st.columns(5)

            # 브랜드 목록 정의 - 특정 브랜드를 우선순위로 배치하고 나머지는 알파벳 순서로
            priority_brands = ['도요타', '두산', '현대', '클라크']
            
            # 우선순위 브랜드 목록 생성 - df를 df1로 변경
            brand_list = [brand for brand in priority_brands if brand in df1['브랜드'].unique()]
            
            # 나머지 브랜드 추가 (우선순위와 '기타'를 제외하고 정렬) - df를 df1로 변경
            other_brands = sorted([brand for brand in df1['브랜드'].unique() 
                                if brand not in priority_brands and brand != '기타'])
            brand_list.extend(other_brands)
            
            # '기타'가 있으면 마지막에 추가 - df를 df1로 변경
            if '기타' in df1['브랜드'].unique():
                brand_list.append('기타')
                
            with col1:
                selected_brand = st.selectbox("브랜드(필수)", brand_list)

            with col2:
                # df를 df1로 변경
                brand_models = df1[df1['브랜드'] == selected_brand]['모델명'].unique()
                selected_model = st.selectbox("모델(필수)", brand_models)

            # 브랜드/모델 선택 이후 필터링 - df를 df1로 변경
            filtered_df = df1[(df1['브랜드'] == selected_brand) & (df1['모델명'] == selected_model)]

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


                # 브랜드 + 모델 기준 1차 필터링 - df를 df1로 변경
                filtered_df = df1[(df1['브랜드'] == selected_brand) & (df1['모델명'] == selected_model)]

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
                    # 모델이 준비되지 않았으면 예측 시도하지 않음
                    if not models_ready:
                        st.error("예측 모델이 준비되지 않았습니다. 데이터를 확인하세요.")
                    else:
                        with st.spinner("예측 분석 중..."):
                            try:
                                # 선택한 값을 인코딩
                                brand_code = le_brand.transform([selected_brand])[0]
                                model_code = le_model.transform([selected_model])[0]

                                # 최근 정비 데이터 가져오기 - df를 df1로 변경
                                latest_data = df1[(df1['브랜드'] == selected_brand) & (df1['모델명'] == selected_model)] \
                                    .sort_values('정비일자', ascending=False).iloc[0]

                                category_code = le_category.transform([latest_data['작업유형']])[0]
                                subcat_code = le_subcategory.transform([latest_data['정비대상']])[0]
                                detail_code = le_detail.transform([latest_data['정비작업']])[0]

                                # 제조년도 구간 → 인코딩
                                if selected_year_range == "전체":
                                    # df를 df1로 변경
                                    mode_range = df1['제조년도'].dropna().astype(int).apply(year_to_range).mode().iloc[0]
                                else:
                                    mode_range = selected_year_range
                                year_range_encoded = le_year_range.transform([mode_range])[0]

                                # 예측할 데이터 준비 - AS처리일수 제거
                                pred_data = np.array([[ 
                                    brand_code, model_code, category_code, subcat_code, detail_code, 
                                    year_range_encoded
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
                2. 필요한 컬럼(브랜드, 모델명, 작업유형, 정비대상, 정비작업)이 모두 있는지 확인 
                3. 재정비 간격 정보가 있는지 확인
                """)

else:
    st.header("산업장비 AS 대시보드")
    st.info("좌측에 데이터 파일을 업로드해 주세요.")

    # 대시보드 설명 표시
    st.markdown("""
    ### 분석 메뉴
    
    1. **정비일지 대시보드**: 정비일지 데이터 기반의 AS 분석 (정비구분별 탭 제공)
    2. **고장 유형 분석**: 고장 유형 분포 및 브랜드-모델별 고장 패턴 히트맵
    3. **브랜드/모델 분석**: 브랜드 및 모델별 특성 분석
    4. **정비내용 분석**: 정비내용 워드클라우드 및 분류별 정비내용 분석
    5. **고장 예측**: 기계학습 모델을 활용한 재정비 기간 및 증상 예측
    
    ### 필요한 파일
    
    - **정비일지 데이터**: 장비 AS 정보
    - **수리비 데이터**: 수리비용 정보
    """)
