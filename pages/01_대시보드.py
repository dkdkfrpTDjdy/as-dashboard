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
                        fig, ax = create_figure(figsize=(10, 6), dpi=150)
                        blue_palette = sns.color_palette("Blues", n_colors=len(region_counts))
                        
                        sns.barplot(x=region_counts.index, y=region_counts.values, ax=ax, palette=blue_palette)
                        
                        # 막대 위에 텍스트 표시
                        for i, v in enumerate(region_counts.values):
                            ax.text(i, v + max(region_counts.values) * 0.02, str(v), ha='center', fontsize=10)
                        
                        plt.tight_layout()
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
                            ax.text(index, row[operation_col] + 0.2, f"{row[operation_col]:.0f}", ha='center')
                        
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        st.pyplot(fig, use_container_width=True)
                        st.markdown(get_image_download_link(fig, f'{category_name}_월별_평균_가동시간.png', '월별 평균 가동시간 다운로드'), unsafe_allow_html=True)
                    else:
                        if operation_col not in df.columns:
                            st.warning("가동시간 데이터가 없습니다.")
                    
                    chart_option = st.radio(
                        "차트 선택", 
                        ["월별 평균 가동시간", "월별 평균 수리비"],
                        key=f"{key_prefix}_chart_option",
                        horizontal=True
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
                        ax.text(i, v + max(repair_time_counts.values) * 0.01, str(v), ha='center', fontsize=10)
                    
                    plt.xticks()
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    st.markdown(get_image_download_link(fig, f'{category_name}_수리시간_분포.png', '수리시간 분포 다운로드'), unsafe_allow_html=True)
                else:
                    st.warning("수리시간 데이터가 없습니다.")
    
    # 소속별 분석 부분만 수정
    if "소속별 분석" in sections:
        with st.expander("파트별 분석", expanded=True):
            # 미리 계산된 소속별 통계 사용
            if 'dept_repair_stats' in st.session_state and st.session_state.dept_repair_stats is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("정비자 파트별 건수")
                    
                    # 미리 계산된 통계 사용
                    dept_stats = st.session_state.dept_repair_stats
                    
                    # 3건 초과인 소속만 필터링
                    filtered_depts = dept_stats[dept_stats['건수'] > 30]
                    
                    # 필터링된 데이터에서 상위 10개 소속 선택
                    top_depts_by_count = filtered_depts.sort_values('건수', ascending=False).head(10)
                    
                    if not top_depts_by_count.empty:
                        # 그래프 생성
                        fig, ax = create_figure(figsize=(12, 8), dpi=150)
                        sns.barplot(x='건수', y='정비자소속', data=top_depts_by_count, ax=ax, palette="Blues_r")
                        
                        # 막대 위에 텍스트 표시
                        for i, row in enumerate(top_depts_by_count.itertuples()):
                            ax.text(row.건수 + 0.5, i, f"{row.건수}건", va='center', fontsize=8)
                        
                        ax.set_xlabel('정비 건수')
                        plt.tight_layout()
                        
                        st.pyplot(fig, use_container_width=True)
                        st.markdown(get_image_download_link(fig, f'{category_name}_파트별_정비건수.png', '파트별 정비건수 다운로드'), unsafe_allow_html=True)
                    else:
                        st.warning("3건 초과인 소속별 정비 건수 데이터가 없습니다.")
                
                # 파트별 건수 대비 수리비 부분 수정
                with col2:
                    st.subheader("파트별 건수 대비 수리비")
                    
                    if not dept_stats.empty and '총수리비' in dept_stats.columns and '건수' in dept_stats.columns:
                        # 3건 초과인 소속만 필터링
                        filtered_depts = dept_stats[dept_stats['건수'] > 3]
                        
                        # 건수 대비 수리비 효율성 지수 계산
                        # 건수 비율 대비 수리비 비율 (1보다 크면 비용이 많이 소요, 1보다 작으면 비용 효율적)
                        total_repairs = dept_stats['건수'].sum()
                        total_costs = dept_stats['총수리비'].sum()
                        
                        filtered_depts['건수비율'] = filtered_depts['건수'] / total_repairs * 100
                        filtered_depts['수리비비율'] = filtered_depts['총수리비'] / total_costs * 100
                        filtered_depts['효율성지수'] = filtered_depts['수리비비율'] / filtered_depts['건수비율']
                        
                        # 효율성 지수 기준으로 정렬
                        sorted_depts = filtered_depts.sort_values('효율성지수', ascending=False).head(10)
                        
                        # 그래프 생성
                        fig, ax = create_figure(figsize=(12, 8), dpi=150)
                        
                        # 효율성 지수에 따라 색상 결정 (1보다 크면 빨간색, 작으면 파란색)
                        colors = ["#234fa0" if x > 1 else '#bdc3c7' for x in sorted_depts['효율성지수']]
                        
                        # 효율성 지수 막대 그래프
                        bars = ax.bar(sorted_depts['정비자소속'], sorted_depts['효율성지수'], color=colors)
                        
                        # 기준선 (1.0) 추가
                        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)

                        
                        # 막대 위에 텍스트 표시
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
                        
                        plt.tight_layout()
                        
                        st.pyplot(fig, use_container_width=True)
                        st.markdown(get_image_download_link(fig, f'{category_name}_파트별_수리비효율성.png', '파트별 수리비 효율성 다운로드'), unsafe_allow_html=True)
                        
                        # 효율성 지수에 대한 설명 추가
                        st.info("1 초과: 건수 대비 수리비가 많이 소요됨 / 1 미만: 효율적인 비용으로 수리함")
                        
                        # 데이터 테이블로도 표시
                        display_cols = ['정비자소속', '건수', '총수리비', '건수비율', '수리비비율', '효율성지수']
                        st.dataframe(sorted_depts[display_cols].style.format({
                            '건수비율': '{:.2f}%',
                            '수리비비율': '{:.2f}%',
                            '효율성지수': '{:.2f}',
                            '총수리비': '{:,.0f}원'
                        }))
                    else:
                        st.warning("소속별 정비 건수 또는 수리비 데이터가 없습니다.")

    # 수리비 상세 분석 부분에서 정비자별 수리비 부분만 수정
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
                            ax.text(v + v*0.005, i, f"{v:,.0f}원", va='center', fontsize=6)
                        
                        ax.set_xlabel('총 수리비 (원)')
                        plt.tight_layout()
                        
                        st.pyplot(fig, use_container_width=True)
                        st.markdown(get_image_download_link(fig, f'{category_name}_현장별_수리비_Top15.png', '현장별 수리비 Top15 다운로드'), unsafe_allow_html=True)
                    else:
                        st.warning("유효한 현장 및 수리비 데이터가 없습니다.")
                else:
                    st.warning("현장 정보가 없습니다.")
            
                with col2:
                    st.subheader("정비자별 건수 대비 수리비")
                    
                    # 정비자별 수리비 분석
                    if '정비자' in df.columns:
                        # 유효한 데이터만 필터링
                        valid_data = df.dropna(subset=['정비자', '수리비'])
                        
                        if not valid_data.empty:
                            # 정비자소속 필터링 - TM센터와 nan 제외
                            if '정비자소속' in valid_data.columns:
                                # nan 값과 TM센터 제외
                                valid_data = valid_data[~valid_data['정비자소속'].isin(['TM센터', 'nan', np.nan])]
                                valid_data = valid_data[~valid_data['정비자소속'].isna()]
                                valid_data = valid_data[~valid_data['정비자소속'].astype(str).str.lower().str.contains('nan')]
                                
                                # 정비자번호와 이름 조합
                                valid_data['정비자정보'] = valid_data['정비자'].astype(str) + " (" + valid_data['정비자소속'].astype(str) + ")"
                            else:
                                valid_data['정비자정보'] = valid_data['정비자'].astype(str)
                            
                            # 정비자별 수리비 합계와 건수 계산
                            worker_stats = valid_data.groupby('정비자정보').agg({
                                '수리비': ['sum', 'count']
                            })
                            worker_stats.columns = ['총수리비', '건수']
                            worker_stats.reset_index(inplace=True)
                            
                            # 30건 이상 처리한 정비자만 포함
                            worker_stats = worker_stats[worker_stats['건수'] >= 30]
                            
                            if not worker_stats.empty:
                                # 효율성 지수 계산
                                total_repairs = valid_data.shape[0]  # 전체 정비 건수
                                total_costs = valid_data['수리비'].sum()  # 전체 수리비
                                
                                worker_stats['건수비율'] = worker_stats['건수'] / total_repairs * 100
                                worker_stats['수리비비율'] = worker_stats['총수리비'] / total_costs * 100
                                worker_stats['효율성지수'] = worker_stats['수리비비율'] / worker_stats['건수비율']
                                
                                # 효율성 지수 기준으로 정렬
                                sorted_workers = worker_stats.sort_values('효율성지수', ascending=False).head(20)
                                
                                # 그래프 생성
                                bar_count = len(sorted_workers)
                                fig_height = bar_count * 0.4 + 2  # 막대 하나당 0.4 + 상하 여백
                                fig, ax = create_figure(figsize=(12, fig_height), dpi=150)

                                efficiency_mean = worker_stats['효율성지수'].mean()
                                # 효율성 지수에 따라 색상 결정 (1보다 크면 초록색, 작으면 회색)
                                colors = ['#234fa0' if x > efficiency_mean else '#bdc3c7' for x in sorted_workers['효율성지수']]
                                
                                # 효율성 지수 막대 그래프
                                bars = ax.barh(sorted_workers['정비자정보'], sorted_workers['효율성지수'], color=colors)
                                
                                # 기준선 (1.0) 추가
                                ax.axvline(x=efficiency_mean, color='red', linestyle='--', alpha=0.7)
                                
                                # 막대 오른쪽에 텍스트 표시
                                for i, bar in enumerate(bars):
                                    width = bar.get_width()
                                    ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center', fontsize=9)

                                # 시각적 정리: 정렬 및 레이아웃
                                ax.invert_yaxis()  # 상위 효율 정비자가 위로
                                plt.tight_layout()
                                
                                st.pyplot(fig, use_container_width=True)
                                st.markdown(get_image_download_link(fig, f'{category_name}_정비자별_수리비효율성.png', '정비자별 수리비 효율성 다운로드'), unsafe_allow_html=True)
                                
                                # 효율성 지수에 대한 설명 추가
                                st.markdown(
                                    f"""
                                    <div style="background-color:#e1f5fe; padding: 10px; border-radius: 5px;">
                                        <b>평균 효율성 지수 기준:</b> {efficiency_mean:.2f}<br>
                                        <b>{efficiency_mean:.2f} 초과</b> → 평균보다 높은 수리비 사용 <span style="color:red;">(비효율)</span><br>
                                        <b>{efficiency_mean:.2f} 이하</b> → 평균보다 적은 수리비로 처리 <span style="color:green;">(고효율)</span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                            else:
                                st.warning("30건 이상 처리한 정비자 데이터가 없습니다.")
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
