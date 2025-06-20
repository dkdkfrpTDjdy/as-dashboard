## 3. Home.py (메인 페이지)

import streamlit as st
import pandas as pd
from utils.data_processing import load_data, merge_dataframes, extract_and_apply_region
from utils.data_processing import calculate_previous_maintenance_dates, map_employee_data, merge_repair_costs
from utils.data_processing import process_date_columns, preprocess_repair_costs
from utils.visualization import setup_korean_font
import os

# 페이지 설정
st.set_page_config(
    page_title="산업장비 AS 분석 대시보드",
    layout="wide"
)

# 한글 폰트 설정 (한 번만 실행)
setup_korean_font()

# 세션 상태 초기화
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# 사이드바 설정
st.sidebar.title("데이터 업로드")

# 파일 업로더
uploaded_file1 = st.sidebar.file_uploader("**정비일지 데이터 업로드**", type=["xlsx"])
uploaded_file3 = st.sidebar.file_uploader("**수리비 데이터 업로드**", type=["xlsx"])

# 내장 데이터 로드 (자산조회 및 조직도)
@st.cache_data
def load_static_data():
    try:
        # 자산조회 데이터 로드
        asset_data_path = "data/자산조회데이터.xlsx"
        if os.path.exists(asset_data_path):
            df2 = pd.read_excel(asset_data_path)
            df2.columns = [str(col).strip().replace('\n', '') for col in df2.columns]
        else:
            df2 = None
            st.sidebar.warning("자산조회 데이터 파일이 없습니다.")
        
        # 조직도 데이터 로드
        org_data_path = "data/조직도데이터.xlsx"
        if os.path.exists(org_data_path):
            df4 = pd.read_excel(org_data_path)
            df4.columns = [str(col).strip().replace('\n', '') for col in df4.columns]
        else:
            df4 = None
            st.sidebar.warning("조직도 데이터 파일이 없습니다.")
        
        return df2, df4
    except Exception as e:
        st.sidebar.error(f"내장 데이터 로드 중 오류 발생: {e}")
        return None, None

# 내장 데이터 로드
df2, df4 = load_static_data()

# 메인 제목
st.title("산업장비 AS 분석 대시보드")

# 사용자 업로드 파일 처리
if uploaded_file1 is not None:
    # 정비일지 데이터 로드
    df1 = load_data(uploaded_file1)
    
    if df1 is not None:
        st.session_state.df1 = df1
        st.session_state.file_name1 = uploaded_file1.name
        
        # 자산 데이터와 병합
        if df2 is not None:
            df1 = merge_dataframes(df1, df2)
            
        # 최근 정비일자 계산
        df1 = calculate_previous_maintenance_dates(df1)
        
        # 현장 컬럼에서 지역 정보 추출
        df1 = extract_and_apply_region(df1)
        
        # 날짜 처리 및 재정비 간격 계산
        df1 = process_date_columns(df1)
        
        # 조직도 데이터 매핑
        if df4 is not None:
            df1 = map_employee_data(df1, df4)
        
        st.session_state.df1_processed = df1
        st.success(f"정비일지 데이터가 성공적으로 로드되었습니다. (총 {len(df1)}개 레코드)")

if uploaded_file3 is not None:
    # 수리비 데이터 로드
    df3 = load_data(uploaded_file3)
    
    if df3 is not None:
        st.session_state.df3 = df3
        st.session_state.file_name3 = uploaded_file3.name
        
        # 수리비 데이터 전처리
        df3 = preprocess_repair_costs(df3)
        
        # 조직도 데이터 매핑
        if df4 is not None:
            df3 = map_employee_data(df3, df4)
        
        st.session_state.df3_processed = df3
        st.success(f"수리비 데이터가 성공적으로 로드되었습니다. (총 {len(df3)}개 레코드)")

# 정비일지와 수리비 데이터 병합
if 'df1_processed' in st.session_state and 'df3_processed' in st.session_state:
    df1 = st.session_state.df1_processed
    df3 = st.session_state.df3_processed
    
    # 병합 실행
    df1_with_costs = merge_repair_costs(df1, df3)
    st.session_state.df1_with_costs = df1_with_costs
    
    st.success("정비일지와 수리비 데이터가 성공적으로 병합되었습니다.")
    
    # 데이터 로드 상태 업데이트
    st.session_state.data_loaded = True

# 로드된 데이터 확인 및 미리보기
if st.session_state.data_loaded:
    st.header("데이터 미리보기")
    
    # 탭 생성
    data_tabs = st.tabs(["정비일지 데이터", "수리비 데이터", "처리 정보"])
    
    with data_tabs[0]:
        if 'df1_with_costs' in st.session_state:
            st.write(st.session_state.df1_with_costs.head())
            
            # 정비구분 정보 표시
            if '정비구분' in st.session_state.df1_with_costs.columns:
                maint_types = st.session_state.df1_with_costs['정비구분'].value_counts()
                st.write("정비구분별 건수:", maint_types)
    
    with data_tabs[1]:
        if 'df3_processed' in st.session_state:
            st.write(st.session_state.df3_processed.head())
    
    with data_tabs[2]:
        # 데이터 처리 정보 표시
        st.write("### 데이터 처리 정보")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'df1_with_costs' in st.session_state:
                df1 = st.session_state.df1_with_costs
                st.write(f"- 정비일지 레코드 수: {len(df1):,}개")
                st.write(f"- 정비일자 범위: {df1['정비일자'].min().strftime('%Y-%m-%d')} ~ {df1['정비일자'].max().strftime('%Y-%m-%d')}")
                st.write(f"- 브랜드 수: {df1['브랜드'].nunique()}개")
                st.write(f"- 모델 수: {df1['모델명'].nunique()}개")
        
        with col2:
            if 'df3_processed' in st.session_state:
                df3 = st.session_state.df3_processed
                st.write(f"- 수리비 레코드 수: {len(df3):,}개")
                if '출고금액' in df3.columns:
                    st.write(f"- 총 수리비: {df3['출고금액'].sum():,.0f}원")
                if '자재명' in df3.columns:
                    st.write(f"- 자재 종류 수: {df3['자재명'].nunique()}개")

else:
    # 데이터가 로드되지 않은 경우 안내 메시지 표시
    st.info("좌측 사이드바에서 정비일지 및 수리비 데이터를 업로드해 주세요.")
    
    # 대시보드 설명 표시
    st.markdown("""
    ## 산업장비 AS 분석 대시보드 사용 안내
    
    ### 분석 메뉴
    
    1. **대시보드**: 정비일지 및 수리비 통합 분석
        - 정비 건수, 가동시간, 수리시간, 수리비 등 주요 지표
        - 월별, 지역별, 소속별 분석
        - 수리비 상세 분석
    
    2. **고장 유형 분석**: 고장 유형 분포 및 브랜드-모델별 고장 패턴
        - 작업유형, 정비대상, 정비작업별 분석
        - 고장 패턴 히트맵
        - 연료, 운전방식 등 자재내역 분석
    
    3. **브랜드/모델 분석**: 브랜드 및 모델별 특성 분석
        - 브랜드별 AS 비율
        - 모델별 AS 비율
        - 제조년도별 분석
    
    4. **고장 예측**: 기계학습 모델을 활용한 미래 정비 예측
        - 다음 고장 시기 예측
        - 예상 고장 유형 분석
    
    ### 필요한 파일
    
    - **정비일지 데이터**: 장비 AS 정보 (필수)
    - **수리비 데이터**: 수리비용 정보 (선택)
    
    ### 주의 사항
    
    - 데이터 로드 후 자동으로 데이터 전처리가 수행됩니다.
    - 내부/외부 정비구분이 있는 경우 각각 분석할 수 있습니다.
    """)
    
    # 샘플 화면 표시
    with st.expander("분석 화면 예시", expanded=True):
        st.image("https://via.placeholder.com/800x400?text=AS+분석+대시보드+예시", 
                 caption="분석 화면 예시 (데이터 업로드 시 실제 데이터로 분석됩니다)")