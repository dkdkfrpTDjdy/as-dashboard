import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from utils.visualization import create_figure, get_image_download_link, get_color_theme

# 페이지 설정
st.set_page_config(
    page_title="AS 대시보드",
    layout="wide"
)

# 색상 테마 설정
current_theme = get_color_theme("대시보드")

# 페이지 제목
st.title("산업장비 AS 대시보드")

# 세션 상태 확인
if 'df1_with_costs' not in st.session_state:
    st.warning("정비일지 데이터가 로드되지 않았습니다. 홈 화면에서 데이터를 업로드해 주세요.")
    st.stop()

# 데이터 불러오기
df1 = st.session_state.df1_with_costs

# 조직도 데이터 확인 및 로드
import os  # 파일 경로 확인을 위해 필요

if 'df4' not in st.session_state or st.session_state.df4 is None:
    try:
        # 조직도 데이터 직접 로드
        org_data_path = "data/조직도데이터.xlsx"
        if os.path.exists(org_data_path):
            st.session_state.df4 = pd.read_excel(org_data_path)
            st.session_state.df4.columns = [str(col).strip().replace('\n', '') for col in st.session_state.df4.columns]
        else:
            # 빈 데이터프레임 생성
            st.session_state.df4 = pd.DataFrame(columns=['소속'])
    except Exception as e:
        # 빈 데이터프레임 생성
        st.session_state.df4 = pd.DataFrame(columns=['소속'])

# 정비구분 컬럼 전처리 (줄바꿈 제거 및 공백 정리)
if '정비구분' in df1.columns:
    df1['정비구분'] = df1['정비구분'].astype(str).apply(lambda x: x.strip().replace('\n', '') if not pd.isna(x) else x)
    # NaN 값을 문자열 'nan'으로 변환한 경우 다시 NaN으로 변경
    df1.loc[df1['정비구분'] == 'nan', '정비구분'] = np.nan

# 통합 대시보드 표시 함수 (정비일지 + 수리비 통합)
def display_integrated_dashboard(df, category_name, key_prefix):
    # 지표 카드용 컬럼 생성
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_cases = len(df)
        st.metric(f"{category_name} AS 건수", f"{total_cases:,}")

    with col2:
        # 가동시간 컬럼이 있는지 확인
        operation_col = None
        for col in df.columns:
            if '가동시간' in col:
                operation_col = col
                break

        if operation_col and df[operation_col].notna().any():
            avg_operation = df[operation_col].mean()
            st.metric("평균 가동시간", f"{avg_operation:.2f}시간")
        else:
            st.metric("평균 가동시간", "데이터 없음")

    with col3:
        # 수리시간 컬럼이 있는지 확인
        repair_col = None
        for col in df.columns:
            if '수리시간' in col:
                repair_col = col
                break

        if repair_col and df[repair_col].notna().any():
            avg_repair = df[repair_col].mean()
            st.metric("평균 수리시간", f"{avg_repair:.2f}시간")
        else:
            st.metric("평균 수리시간", "데이터 없음")

    with col4:
        # 수리비 평균 표시
        if '수리비' in df.columns and df['수리비'].notna().any():
            avg_cost = df['수리비'].mean()
            st.metric("평균 수리비용", f"{avg_cost:,.0f}원")
        else:
            st.metric("평균 수리비용", "데이터 없음")

    st.markdown("---")
    
# 분석 섹션 선택기 (확장 가능한 섹션으로 분리)
    sections = st.multiselect(
        "표시할 분석 섹션 선택",
        ["기본 분석", "소속별 분석", "수리비 상세 분석"],
        default=["기본 분석"],  # 기본 분석을 기본값으로 설정
        key=f"{key_prefix}_sections"
    )

    # 1. 기본 분석 (월별 + 지역별)
    if "기본 분석" in sections:
        with st.expander("기본 분석", expanded=True):
            st.subheader("월별 및 지역별 분석")
            
            # 첫 번째 줄: 월별 AS 건수와 지역별 AS 건수
            col1, col2 = st.columns(2)
            
            with col1:
                # 월별 AS 건수
                st.subheader("월별 AS 건수")
                if '정비일자' in df.columns:
                    # 데이터 준비
                    df_time = df.copy()
                    
                    # datetime 형식 확인 및 변환
                    if not pd.api.types.is_datetime64_any_dtype(df_time['정비일자']):
                        try:
                            df_time['정비일자'] = pd.to_datetime(df_time['정비일자'], errors='coerce')
                        except Exception as e:
                            st.error(f"'정비일자' 컬럼을 datetime으로 변환하는 데 실패했습니다: {e}")
                            st.stop()
                    
                    # 이전에 작동했던 방식으로 월 추출
                    df_time['월'] = df_time['정비일자'].dt.to_period('M')
                    monthly_counts = df_time.groupby('월').size().reset_index(name='건수')
                    monthly_counts['월'] = monthly_counts['월'].astype(str)
                    
                    # 그래프 생성
                    fig, ax = create_figure(figsize=(10, 6), dpi=150)
                    sns.barplot(x='월', y='건수', data=monthly_counts, ax=ax, palette='Blues')
                    
                    # 막대 위에 텍스트 표시
                    for i, v in enumerate(monthly_counts['건수']):
                        ax.text(i, v + max(monthly_counts['건수']) * 0.01, str(v), ha='center')
                        
                    plt.xticks(rotation=45)
                    ax.set_ylabel('건수')
                    plt.tight_layout()
                    
                    st.pyplot(fig, use_container_width=True)
                    st.markdown(get_image_download_link(fig, f'{category_name}_월별_AS_건수.png', '월별 AS 건수 다운로드'), unsafe_allow_html=True)
                else:
                    st.warning("정비일자 컬럼이 없어 월별 분석을 수행할 수 없습니다.")
            
            with col2:
                # 지역별 AS 건수
                st.subheader("지역별 AS 건수")
                if '지역' in df.columns:
                    df_clean = df.dropna(subset=['지역']).copy()
                    
                    # 없으면 건너뛰기
                    if len(df_clean) == 0:
                        st.warning("지역 정보가 없습니다.")
                    else:
                        region_counts = df_clean['지역'].value_counts()
                        
                        # 최소 빈도수 처리 및 상위 15개 표시
                        others_count = region_counts[region_counts < 3].sum()
                        region_counts = region_counts[region_counts >= 3]
                        if others_count > 0:
                            region_counts['기타'] = others_count
                        
                        region_counts = region_counts.sort_values(ascending=False).nlargest(15)
                        
                        # 시각화
                        fig, ax = create_figure(figsize=(10, 8), dpi=150)
                        blue_palette = sns.color_palette("Blues", n_colors=len(region_counts))
                        
                        sns.barplot(x=region_counts.index, y=region_counts.values, ax=ax, palette=blue_palette)
                        
                        # 막대 위에 텍스트 표시
                        for i, v in enumerate(region_counts.values):
                            ax.text(i, v + max(region_counts.values) * 0.02, str(v), ha='center', fontsize=10)
                        
                        plt.tight_layout()
                        plt.xticks(rotation=45)
                        st.pyplot(fig, use_container_width=True)
                        
                        # 다운로드 링크 추가
                        st.markdown(get_image_download_link(fig, f'{category_name}_지역별_AS_현황.png', '지역별 AS 현황 다운로드'), unsafe_allow_html=True)
                else:
                    st.warning("지역 정보가 없습니다.")
            
            # 두 번째 줄: 월별 평균 지표와 수리시간 분포
            col3, col4 = st.columns(2)
            
        with col3:
            # 월별 평균 지표
            st.subheader("월별 평균 지표")
            if '정비일자' in df.columns:
                # 기본 차트 옵션 설정 (초기 표시할 차트)
                default_chart = "월별 평균 가동시간"
                
                # 먼저 기본 차트 표시
                if default_chart == "월별 평균 가동시간" and operation_col in df.columns:
                    # 데이터 준비
                    df_op = df.copy()
                    df_op['월'] = df_op['정비일자'].dt.to_period('M')
                    monthly_avg = df_op.groupby('월')[operation_col].mean().reset_index()
                    monthly_avg['월'] = monthly_avg['월'].astype(str)
                    
                    # 그래프 생성
                    fig, ax = create_figure(figsize=(10, 6), dpi=150)
                    sns.barplot(data=monthly_avg, x='월', y=operation_col, ax=ax, palette="Blues")
                    
                    # 평균값 텍스트 표시
                    for index, row in monthly_avg.iterrows():
                        ax.text(index, row[operation_col] + 0.2, f"{row[operation_col]:.1f}시간", ha='center')
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig, use_container_width=True)
                    st.markdown(get_image_download_link(fig, f'{category_name}_월별_평균_가동시간.png', '월별 평균 가동시간 다운로드'), unsafe_allow_html=True)
                else:
                    if operation_col not in df.columns:
                        st.warning("가동시간 데이터가 없습니다.")
                
                # 차트 선택 라디오 버튼을 그래프 아래에 배치
                chart_option = st.radio(
                    "차트 선택", 
                    ["월별 평균 가동시간", "월별 평균 수리비"],
                    key=f"{key_prefix}_chart_option",
                    horizontal=True  # 수평으로 배치하여 공간 절약
                )
                
                # 선택된 차트가 기본 차트와 다를 경우에만 다시 그리기
                if chart_option != default_chart:
                    if chart_option == "월별 평균 수리비" and '수리비' in df.columns:
                        # 데이터 준비
                        df_cost = df.copy()
                        df_cost['월'] = df_cost['정비일자'].dt.to_period('M')
                        monthly_cost_avg = df_cost.groupby('월')['수리비'].mean().reset_index()
                        monthly_cost_avg['월'] = monthly_cost_avg['월'].astype(str)
                        
                        # 그래프 생성
                        fig, ax = create_figure(figsize=(10, 6), dpi=150)
                        sns.barplot(x='월', y='수리비', data=monthly_cost_avg, ax=ax, palette="Blues")
                        
                        # 평균값 텍스트 표시
                        for i, v in enumerate(monthly_cost_avg['수리비']):
                            ax.text(i, v + max(monthly_cost_avg['수리비']) * 0.01, f"{v:,.0f}원", ha='center', fontsize=8)
                        
                        plt.xticks(rotation=45)
                        ax.set_ylabel('평균 수리비 (원)')
                        plt.tight_layout()
                        
                        st.pyplot(fig, use_container_width=True)
                        st.markdown(get_image_download_link(fig, f'{category_name}_월별_평균_수리비.png', '월별 평균 수리비 다운로드'), unsafe_allow_html=True)
                    elif chart_option == "월별 평균 가동시간" and operation_col in df.columns:
                        # 이미 위에서 표시했으므로 여기서는 아무 작업도 하지 않음
                        pass
                    else:
                        if chart_option == "월별 평균 가동시간":
                            st.warning("가동시간 데이터가 없습니다.")
                        else:
                            st.warning("수리비 데이터가 없습니다.")
            else:
                st.warning("정비일자 컬럼이 없어 월별 분석을 수행할 수 없습니다.")
            
            with col4:
                # 수리시간 분포
                st.subheader("수리시간 분포")
                if repair_col in df.columns:
                    # 데이터 준비
                    bins = [0, 2, 4, 8, 12, 24, float('inf')]
                    labels = ['0-2시간', '2-4시간', '4-8시간', '8-12시간', '12-24시간', '24시간 이상']
                    df_repair = df.copy()
                    df_repair['수리시간_구간'] = pd.cut(df_repair[repair_col], bins=bins, labels=labels)
                    repair_time_counts = df_repair['수리시간_구간'].value_counts().sort_index()
                    
                    # 그래프 생성
                    fig, ax = create_figure(figsize=(10, 6), dpi=150)
                    sns.barplot(x=repair_time_counts.index, y=repair_time_counts.values, ax=ax, palette="Blues")
                    
                    # 막대 위에 텍스트 표시
                    for i, v in enumerate(repair_time_counts.values):
                        ax.text(i, v + max(repair_time_counts.values) * 0.02, str(v), ha='center', fontsize=10)
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    st.markdown(get_image_download_link(fig, f'{category_name}_수리시간_분포.png', '수리시간 분포 다운로드'), unsafe_allow_html=True)
                else:
                    st.warning("수리시간 데이터가 없습니다.")
    
    # 3. 소속별 분석
    if "소속별 분석" in sections:
        with st.expander("소속별 분석", expanded=True):
            # 미리 계산된 소속별 통계 사용
            if 'dept_repair_stats' in st.session_state and st.session_state.dept_repair_stats is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("정비자 소속별 건수")
                    
                    # 미리 계산된 통계 사용
                    dept_stats = st.session_state.dept_repair_stats
                    
                    # 상위 10개 소속 선택
                    top_depts_by_count = dept_stats.sort_values('건수', ascending=False).head(10)
                    
                    if not top_depts_by_count.empty:
                        # 그래프 생성
                        fig, ax = create_figure(figsize=(10, 8), dpi=150)
                        sns.barplot(x='건수', y='정비자소속', data=top_depts_by_count, ax=ax, palette="Blues_r")
                        
                        # 막대 위에 텍스트 표시
                        for i, row in enumerate(top_depts_by_count.itertuples()):
                            ax.text(row.건수 + 0.5, i, f"{row.건수}건", va='center', fontsize=8)
                        
                        ax.set_xlabel('정비 건수')
                        plt.tight_layout()
                        
                        st.pyplot(fig, use_container_width=True)
                        st.markdown(get_image_download_link(fig, f'{category_name}_소속별_정비건수.png', '소속별 정비건수 다운로드'), unsafe_allow_html=True)
                    else:
                        st.warning("소속별 정비 건수 데이터가 없습니다.")
                
                with col2:
                    st.subheader("정비자 소속별 수리비")
                    
                    # 상위 10개 소속 선택 (인원당 수리비 기준)
                    top_depts_by_cost = dept_stats.sort_values('인원당수리비', ascending=False).head(10)
                    
                    if not top_depts_by_cost.empty:
                        # 그래프 생성
                        fig, ax = create_figure(figsize=(10, 8), dpi=150)
                        sns.barplot(x='인원당수리비', y='정비자소속', data=top_depts_by_cost, ax=ax, palette="Blues_r")
                        
                        # 막대 위에 텍스트 표시
                        for i, row in enumerate(top_depts_by_cost.itertuples()):
                            ax.text(row.인원당수리비 + 100, i, f"{row.인원당수리비:,.0f}원/인", va='center', fontsize=8)
                        
                        ax.set_xlabel('인원당 수리비 (원)')
                        plt.tight_layout()
                        
                        st.pyplot(fig, use_container_width=True)
                        st.markdown(get_image_download_link(fig, f'{category_name}_소속별_인원당수리비.png', '소속별 인원당수리비 다운로드'), unsafe_allow_html=True)
                    else:
                        st.warning("소속별 수리비 데이터가 없습니다.")
            else:
                st.info("소속별 분석 데이터가 준비되지 않았습니다. 홈 화면에서 데이터를 다시 로드해 주세요.")
    
    # 4. 수리비 상세 분석
    if "수리비 상세 분석" in sections and '수리비' in df.columns:
        with st.expander("수리비 상세 분석", expanded=True):
            st.header("수리비 상세 분석")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("수리비 많은 현장 Top 15")
                
                # 현장별 수리비 합계
                if '현장명' in df.columns:
                    # 유효한 데이터만 필터링
                    valid_data = df.dropna(subset=['현장명', '수리비'])
                    
                    if not valid_data.empty:
                        site_costs = valid_data.groupby('현장명')['수리비'].sum().sort_values(ascending=False).head(15)
                        
                        # 그래프 생성
                        fig, ax = create_figure(figsize=(10, 8), dpi=150)
                        sns.barplot(x=site_costs.values, y=site_costs.index, ax=ax, palette="Blues_r")
                        
                        # 값 표시
                        for i, v in enumerate(site_costs.values):
                            ax.text(v + v*0.01, i, f"{v:,.0f}원", va='center', fontsize=8)
                        
                        ax.set_xlabel('총 수리비 (원)')
                        plt.tight_layout()
                        
                        st.pyplot(fig, use_container_width=True)
                        st.markdown(get_image_download_link(fig, f'{category_name}_현장별_수리비_Top15.png', '현장별 수리비 Top15 다운로드'), unsafe_allow_html=True)
                    else:
                        st.warning("유효한 현장 및 수리비 데이터가 없습니다.")
                else:
                    st.warning("현장 정보가 없습니다.")
            
            with col2:
                st.subheader("정비자별 수리비 Top 15")
                
                # 정비자별 수리비 합계
                if '정비자번호' in df.columns and '정비자' in df.columns:
                    # 유효한 데이터만 필터링
                    valid_data = df.dropna(subset=['정비자번호', '정비자', '수리비'])
                    
                    if not valid_data.empty:
                        # 정비자번호와 이름 조합
                        valid_data['정비자정보'] = valid_data['정비자'].astype(str) + " (" + valid_data['정비자번호'].astype(str) + ")"
                        worker_costs = valid_data.groupby('정비자정보')['수리비'].sum().sort_values(ascending=False).head(15)
                        
                        # 그래프 생성
                        fig, ax = create_figure(figsize=(10, 8), dpi=150)
                        sns.barplot(x=worker_costs.values, y=worker_costs.index, ax=ax, palette="Blues_r")
                        
                        # 값 표시
                        for i, v in enumerate(worker_costs.values):
                            ax.text(v + v*0.01, i, f"{v:,.0f}원", va='center', fontsize=8)
                        
                        ax.set_xlabel('총 수리비 (원)')
                        plt.tight_layout()
                        
                        st.pyplot(fig, use_container_width=True)
                        st.markdown(get_image_download_link(fig, f'{category_name}_정비자별_수리비_Top15.png', '정비자별 수리비 Top15 다운로드'), unsafe_allow_html=True)
                    else:
                        st.warning("유효한 정비자 및 수리비 데이터가 없습니다.")
                else:
                    st.warning("정비자 정보가 없습니다.")
            
            # 자주 사용되는 부품 분석
            if '사용부품' in df.columns:
                st.subheader("자주 사용되는 부품 분석")
                
                # 부품 데이터 추출 및 처리
                parts_data = []
                for parts_str in df['사용부품'].dropna():
                    if isinstance(parts_str, str) and parts_str.strip():
                        # 쉼표로 구분된 부품 목록 처리
                        parts = [p.strip() for p in parts_str.split(',')]
                        parts_data.extend(parts)
                
                # 부품별 빈도 계산
                parts_counter = Counter(parts_data)
                top_parts = parts_counter.most_common(15)
                
                if top_parts:
                    # 데이터프레임 생성
                    parts_df = pd.DataFrame(top_parts, columns=['부품명', '사용빈도'])
                    
                    # 그래프 생성
                    fig, ax = create_figure(figsize=(12, 8), dpi=150)
                    sns.barplot(x='사용빈도', y='부품명', data=parts_df, ax=ax, palette="Blues_r")
                    
                    # 값 표시
                    for i, v in enumerate(parts_df['사용빈도']):
                        ax.text(v + 0.1, i, str(v), va='center')
                    
                    ax.set_xlabel('사용 빈도')
                    plt.tight_layout()
                    
                    st.pyplot(fig, use_container_width=True)
                    st.markdown(get_image_download_link(fig, f'{category_name}_자주사용부품_Top15.png', '자주 사용되는 부품 Top15 다운로드'), unsafe_allow_html=True)
                else:
                    st.warning("부품 사용 데이터가 없습니다.")

# 정비구분 컬럼 확인 및 값 검증
if '정비구분' in df1.columns and df1['정비구분'].notna().any():
    # 실제 존재하는 정비구분 값 확인
    maintenance_types = df1['정비구분'].dropna().unique()
    
    # 내부, 외부 값이 있는지 확인 (대소문자 구분 없이)
    has_internal = any('내부' in str(val).lower() for val in maintenance_types)
    has_external = any('외부' in str(val).lower() for val in maintenance_types)
    
    # 정확한 '내부'/'외부' 값 찾기
    internal_value = next((val for val in maintenance_types if '내부' in str(val).lower()), None)
    external_value = next((val for val in maintenance_types if '외부' in str(val).lower()), None)
    
    # 탭 생성
    tabs = st.tabs(["전체", "내부", "외부"])

    # 전체 탭
    with tabs[0]:
        st.header("전체 정비 현황")
        display_integrated_dashboard(df1, "전체", "all")

    # 내부 탭
    with tabs[1]:
        st.header("내부 정비 현황")
        if has_internal and internal_value is not None:
            df_internal = df1[df1['정비구분'] == internal_value]
            display_integrated_dashboard(df_internal, "내부", "internal")
        else:
            st.info("내부 정비 데이터가 없습니다.")

    # 외부 탭
    with tabs[2]:
        st.header("외부 정비 현황")
        if has_external and external_value is not None:
            df_external = df1[df1['정비구분'] == external_value]
            display_integrated_dashboard(df_external, "외부", "external")
        else:
            st.info("외부 정비 데이터가 없습니다.")
else:
    # 정비구분 컬럼이 없는 경우 전체 데이터만 표시
    st.header("정비 현황")
    display_integrated_dashboard(df1, "전체", "all")
