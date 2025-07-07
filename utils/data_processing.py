# 2. utils/data_processing.py

import pandas as pd
import numpy as np
import streamlit as st
from collections import Counter
import traceback
import datetime
import re

@st.cache_data
def load_data(file):
    """파일에서 데이터를 로드하는 함수"""
    try:
        # 모든 문자열 컬럼을 문자열로 로드하도록 설정
        df = pd.read_excel(file, dtype=str)

        # 컬럼명 정리 (줄바꿈 제거 및 공백 제거)
        df.columns = [str(col).strip().replace('\n', '') for col in df.columns]

        # 관리번호가 있으면 문자열로 강제 변환
        if '관리번호' in df.columns:
            df['관리번호'] = df['관리번호'].astype(str)

        # 숫자형 데이터 변환 (금액, 시간 등)
        numeric_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['금액', '시간', '비용', '단가']):
                numeric_cols.append(col)
        
        # 숫자형 컬럼 변환
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 날짜형 데이터 변환
        date_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['일자', '날짜', 'date']):
                date_cols.append(col)
        
        # 날짜형 컬럼 변환
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 컬럼명 매핑 (정비일지 데이터인 경우)
        try:
            # 대분류, 중분류, 소분류가 있는 경우 작업유형, 정비대상, 정비작업으로 변환
            if all(col in df.columns for col in ['대분류', '중분류', '소분류']):
                df.rename(columns={
                    '대분류': '작업유형',
                    '중분류': '정비대상',
                    '소분류': '정비작업'
                }, inplace=True)
        except Exception as e:
            st.warning(f"일부 데이터 전처리 중 오류가 발생했습니다: {e}")

        return df
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        return None

def extract_region_from_address(address):
    """주소에서 지역 정보를 정확하게 추출하는 함수"""
    if not isinstance(address, str):
        return None, None

    address = address.strip()

    region_prefixes = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
                       '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']

    tokens = address.split()
    if len(tokens) < 2:
        return None, None

    first, second = tokens[0], tokens[1]
    if first in region_prefixes and second.endswith(('시', '군', '구')):
        return first, address

    return None, None

# 현장 컬럼에서 지역과 주소 추출 적용
@st.cache_data
def extract_and_apply_region(df):
    """현장 컬럼에서 지역과 주소를 추출하여 적용하는 함수"""
    df_copy = df.copy()
    if '현장' in df_copy.columns:
        # 지역과 주소 추출
        results = df_copy['현장'].apply(extract_region_from_address)
        
        # 결과에서 지역과 주소 분리
        df_copy['지역'] = [r[0] for r in results]
        df_copy['주소'] = [r[1] for r in results]
        
        # 주소가 아닌 값은 현장명으로 설정
        df_copy['현장명'] = np.where(df_copy['주소'].notna(), None, df_copy['현장'])
        mask = df_copy['주소'].notna()
        df_copy.loc[~mask, '현장명'] = df_copy.loc[~mask, '현장']
    return df_copy

# 문자열 리스트 변환
def convert_to_str_list(arr):
    """NaN과 혼합 유형을 처리하여 문자열 리스트로 변환"""
    return [str(x) for x in arr if not pd.isna(x)]

# 작은 비율 항목을 '기타'로 그룹화
def group_small_categories(series, threshold=0.03):
    """작은 비율의 항목을 '기타'로 그룹화"""
    total = series.sum()
    mask = series / total < threshold
    if mask.any():
        others = pd.Series({'기타': series[mask].sum()})
        return pd.concat([series[~mask], others])
    return series

# 최근 정비일자 계산
@st.cache_data
def calculate_previous_maintenance_dates(df):
    """각 관리번호별 이전 정비일자 계산"""
    df_copy = df.copy()
    
    if '관리번호' not in df_copy.columns or '정비일자' not in df_copy.columns:
        return df_copy

    # 정비일자 정렬 및 그룹화
    df_copy = df_copy.sort_values(['관리번호', '정비일자'])

    # 각 관리번호별로 이전 정비일자 계산
    df_copy['최근정비일자'] = df_copy.groupby('관리번호')['정비일자'].shift(1)

    return df_copy

# 조직도 데이터와 정비자번호/출고자 매핑
@st.cache_data
def map_employee_data(df, org_df):
    """정비자번호 또는 출고자를 조직도 데이터와 매핑"""
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
        st.error(traceback.format_exc())
        return df

# 두 데이터프레임 병합 함수 - 브랜드 매핑 문제 해결
@st.cache_data
def merge_dataframes(df1, df2):
    """정비일지 데이터와 자산조회 데이터 병합"""
    if df1 is None or df2 is None:
        return df1

    try:
        # 데이터 복사
        df1_copy = df1.copy()
        df2_copy = df2.copy()
        
        # 데이터 타입 통일 - 관리번호를 문자열로 변환
        df1_copy['관리번호'] = df1_copy['관리번호'].astype(str)
        df2_copy['관리번호'] = df2_copy['관리번호'].astype(str)
        
        # 중복 관리번호 확인 및 제거 (자산 데이터에서)
        if df2_copy['관리번호'].duplicated().any():
            # 중복 제거 (첫 번째 값 유지)
            df2_copy = df2_copy.drop_duplicates(subset='관리번호')
            
        # 자산 데이터에서 필요한 컬럼만 선택
        df2_subset = df2_copy[['관리번호', '제조사명', '제조사모델명', '제조년도', '취득가', '자재내역']]
        
        # 컬럼명 표준화: 제조사명 -> 브랜드, 제조사모델명 -> 모델명
        df2_subset = df2_subset.rename(columns={
            '제조사명': '브랜드',
            '제조사모델명': '모델명'
        })
        
        # 관리번호 컬럼을 기준으로 왼쪽 조인으로 병합 (AS 데이터는 모두 유지)
        merged_df = pd.merge(df1_copy, df2_subset, on='관리번호', how='left')
            
        # 브랜드 컬럼 처리
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

        return merged_df
    except Exception as e:
        st.error(f"데이터 병합 중 오류 발생: {e}")
        st.error(traceback.format_exc())
        return df1

@st.cache_data
def merge_repair_costs(maintenance_df, parts_df):
    """
    정비일지와 소모품 데이터를 병합하여 수리비와 사용부품을 계산합니다.
    조건:
    - 관리번호 일치
    - 정비자번호 == 출고자
    - 정비일자와 출고일자 간 차이가 ±30일 이내
    """
    if maintenance_df is None or parts_df is None:
        return maintenance_df

    try:
        df1 = maintenance_df.copy()
        df3 = parts_df.copy()

        # 필수 컬럼 확인
        required_cols_df1 = ['관리번호', '정비일자', '정비자번호']
        required_cols_df3 = ['관리번호', '출고일자', '출고자', '출고금액', '자재명']
        for col in required_cols_df1 + required_cols_df3:
            if col not in (df1.columns if col in required_cols_df1 else df3.columns):
                st.warning(f"필수 컬럼 누락: '{col}'")
                return df1

        # 타입 정리
        df1['관리번호'] = df1['관리번호'].astype(str)
        df1['정비자번호'] = df1['정비자번호'].fillna("").astype(str)
        df1['정비일자'] = pd.to_datetime(df1['정비일자'], errors='coerce')

        df3['관리번호'] = df3['관리번호'].astype(str)
        df3['출고자'] = df3['출고자'].astype(str).fillna("")
        df3['출고일자'] = pd.to_datetime(df3['출고일자'], errors='coerce')
        df3['자재명'] = df3['자재명'].fillna("")
        df3['출고금액'] = pd.to_numeric(df3['출고금액'], errors='coerce').fillna(0)

        # 인덱스를 저장
        df1['원본인덱스'] = df1.index

        # 병합: 관리번호 + 정비자번호 매칭
        merged = pd.merge(
            df1[['관리번호', '정비일자', '정비자번호', '원본인덱스']],
            df3[['관리번호', '출고일자', '출고자', '출고금액', '자재명']],
            left_on=['관리번호', '정비자번호'],
            right_on=['관리번호', '출고자'],
            how='inner'
        )

        # 날짜 차이 필터
        merged['일자차이'] = (merged['출고일자'] - merged['정비일자']).dt.days
        merged = merged[merged['일자차이'].abs() <= 30]

        # 수리비 집계
        cost_summary = merged.groupby('원본인덱스')['출고금액'].sum()
        parts_summary = merged.groupby('원본인덱스')['자재명'].agg(
            lambda x: ', '.join(sorted(set(x.dropna())))
        )

        # 결과 반영
        df1['수리비'] = df1['원본인덱스'].map(cost_summary).fillna(0)
        df1['사용부품'] = df1['원본인덱스'].map(parts_summary).fillna("")

        df1.drop('원본인덱스', axis=1, inplace=True)

        # 로그 출력
        matched = (df1['수리비'] > 0).sum()
        st.info(f"총 {len(df1)}건 중 {matched}건 수리비 매칭됨 ({matched / len(df1) * 100:.1f}%)")

        return df1

    except Exception as e:
        st.error("병합 중 오류 발생: " + str(e))
        st.error(traceback.format_exc())
        df1['수리비'] = 0
        df1['사용부품'] = ""
        return df1

# 재정비 간격 계산을 위한 날짜 처리
@st.cache_data
def process_date_columns(df):
    """날짜 컬럼 처리 및 재정비 간격 계산"""
    df_copy = df.copy()
    
    try:
        date_columns = ['정비일자', '최근정비일자']
        for col in date_columns:
            if col in df_copy.columns:
                try:
                    # 기본 날짜 변환 시도
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                except Exception:
                    try:
                        # Excel 날짜 숫자 처리 시도
                        df_copy[col] = pd.to_datetime(df_copy[col], origin='1899-12-30', unit='D', errors='coerce')
                    except Exception:
                        pass

        # 재정비 간격 계산 (정비일자 - 최근정비일자)
        if '최근정비일자' in df_copy.columns and '정비일자' in df_copy.columns:
            df_copy['재정비간격'] = (df_copy['정비일자'] - df_copy['최근정비일자']).dt.days
            # 30일 내 재정비 여부
            df_copy['30일내재정비'] = (df_copy['재정비간격'] <= 30) & (df_copy['재정비간격'] > 0)

    except Exception as e:
        st.error(f"날짜 처리 중 오류 발생: {e}")
        st.error(traceback.format_exc())
    
    return df_copy

# 수리비 데이터 전처리
@st.cache_data
def preprocess_repair_costs(df):
    """수리비 데이터 전처리"""
    df_copy = df.copy()
    
    try:
        # 날짜 변환
        if '출고일자' in df_copy.columns:
            df_copy['출고일자'] = pd.to_datetime(df_copy['출고일자'], errors='coerce')

        # 금액 컬럼 숫자로 변환
        for col in df_copy.columns:
            if '금액' in col or '비용' in col or '단가' in col:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    except Exception as e:
        st.warning(f"수리비 데이터 전처리 중 오류가 발생했습니다: {e}")
    
    return df_copy

def generate_fault_type_column(df):
    if all(col in df.columns for col in ['작업유형', '정비대상', '정비작업']):
        mask = df['작업유형'].notna() & df['정비대상'].notna() & df['정비작업'].notna()
        df.loc[mask, '고장유형'] = df.loc[mask, ['작업유형', '정비대상', '정비작업']].astype(str).agg('_'.join, axis=1)
        df['고장유형'] = df['고장유형'].replace('nan_nan_nan', np.nan)
    return df
