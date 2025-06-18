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

# 정비일지 대시보드 표시 함수
def display_maintenance_dashboard(df, category_name):
    # 지표 카드용 컬럼 생성
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cases = len(df)
        st.metric(f"{category_name} AS 건수", f"{total_cases:,}")
    
    with col2:
        if '가동시간' in df.columns:
            avg_operation = df['가동시간'].mean()
            st.metric("평균 가동시간", f"{avg_operation:.2f}시간")
        else:
            st.metric("평균 가동시간", "데이터 없음")
    
    with col3:
        if '수리시간' in df.columns:
            avg_repair = df['수리시간'].mean()
            st.metric("평균 수리시간", f"{avg_repair:.2f}시간")
        else:
            st.metric("평균 수리시간", "데이터 없음")
    
    with col4:
        if '정비일자' in df.columns:
            last_month = df['정비일자'].max().strftime('%Y-%m')
            last_month_count = df[df['정비일자'].dt.strftime('%Y-%m') == last_month].shape[0]
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
        valid_operation = df.dropna(subset=['가동시간', '수리시간']).copy()
        
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
    "정비일지 대시보드": "Blues",
    "수리비 대시보드": "Purples",
    "고장 유형 분석": "Greens",
    "브랜드/모델 분석": "Oranges",
    "정비내용 분석": "YlOrRd",
    "고장 예측": "viridis"
}

# 사이드바 설정
st.sidebar.title("데이터 업로드 및 메뉴 클릭")

# 데이터 로드 함수 (한 번만 정의)
@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file)
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

# 자동 로드되는 파일 안내
st.sidebar.info("자산조회 데이터와 조직도 데이터는 자동으로 로드됩니다.")

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
    st.sidebar.success("자산조회 데이터가 성공적으로 로드되었습니다.")
except Exception as e:
    st.sidebar.warning(f"자산조회 데이터를 로드할 수 없습니다: {e}")
    df2 = None

try:
    # 조직도 데이터 로드 (같은 레포지토리 내 파일)
    org_data_path = "data/조직도데이터.xlsx"
    df4 = pd.read_excel(org_data_path)
    st.sidebar.success("조직도 데이터가 성공적으로 로드되었습니다.")
except Exception as e:
    st.sidebar.warning(f"조직도 데이터를 로드할 수 없습니다: {e}")
    df4 = None

# 주소에서 지역 추출 함수
def extract_region_from_address(address):
    if not isinstance(address, str):
        return None
    
    # 주소 형태인 경우만 처리 (시/도로 시작하는 경우)
    if len(address) >= 2:
        first_two = address[:2]
        if first_two in ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', 
                         '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']:
            return first_two
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

# 수리비 데이터 전처리
if df3 is not None:
    # 조직도 데이터 매핑
    if df4 is not None:
        df3 = map_employee_data(df3, df4)
        st.sidebar.success("수리비 데이터에 조직도 정보가 매핑되었습니다.")

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
    
    # 정비일지 데이터 전처리
    if df1 is not None:
        try:
            # 날짜 변환 - 오류 수정
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
    
    # 메뉴별 콘텐츠 표시
    if menu == "정비일지 대시보드":
        st.title("정비일지 대시보드")
        
        # 정비 구분에 따른 탭 생성
        if '정비구분' in df1.columns:
            tabs = st.tabs(["전체", "내부", "외부"])
            
            # 탭별 데이터 필터링
            df_all = df1.copy()
            df_internal = df1[df1['정비구분'] == '내부']
            df_external = df1[df1['정비구분'] == '외부']
            
            # 전체 탭
            with tabs[0]:
                st.header("전체 정비 현황")
                
                # 데이터 분석 및 시각화 - 전체
                display_maintenance_dashboard(df_all, "전체")
            
            # 내부 탭
            with tabs[1]:
                st.header("내부 정비 현황")
                
                # 데이터 분석 및 시각화 - 내부
                display_maintenance_dashboard(df_internal, "내부")
            
            # 외부 탭
            with tabs[2]:
                st.header("외부 정비 현황")
                
                # 데이터 분석 및 시각화 - 외부
                display_maintenance_dashboard(df_external, "외부")
        else:
            # 정비구분 컬럼이 없는 경우 전체 데이터로 표시
            display_maintenance_dashboard(df1, "전체")
    
    elif menu == "수리비 대시보드":
        st.title("수리비 대시보드")
        display_repair_cost_dashboard(df3)
    
    elif menu == "고장 유형 분석":
        st.title("고장 유형 분석")

        # 필요한 컬럼 존재 여부 확인
        if all(col in df1.columns for col in ['작업유형', '정비대상', '정비작업', '고장유형']):
            
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
                    category_counts = df1[colname].value_counts().head(15)
                    category_values = convert_to_str_list(df1[colname].unique())
                    selected_category = st.selectbox(f"{colname} 선택", ["전체"] + sorted(category_values), key=f"sel_{colname}")

                    if selected_category != "전체":
                        filtered_df = df1[df1[colname].astype(str) == selected_category]
                    else:
                        filtered_df = df1

                    top_faults = filtered_df['고장유형'].value_counts().nlargest(15).index
                    df_filtered = filtered_df[filtered_df['고장유형'].isin(top_faults)]
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
                if '연료' in df1.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 연료별 AS 건수
                        st.subheader("연료별 AS 건수")
                        fuel_type_counts = df1['연료'].value_counts().dropna()
                        
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
                        fuel_types = ["전체"] + df1['연료'].value_counts().index.tolist()
                        selected_fuel = st.selectbox("연료", fuel_types)
                        
                        if selected_fuel != "전체":
                            filtered_df_fuel = df1[df1['연료'] == selected_fuel]
                        else:
                            filtered_df_fuel = df1
                            
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
                if '운전방식' in df1.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 운전방식별 AS 건수
                        st.subheader("운전방식별 AS 건수")
                        driving_type_counts = df1['운전방식'].value_counts().head(15)
                        
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
                        driving_types = ["전체"] + df1['운전방식'].value_counts().index.tolist()
                        selected_driving = st.selectbox("운전방식", driving_types)
                        
                        if selected_driving != "전체":
                            filtered_df_driving = df1[df1['운전방식'] == selected_driving]
                        else:
                            filtered_df_driving = df1
                            
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
                if '적재용량' in df1.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 적재용량별 AS 건수
                        st.subheader("적재용량별 AS 건수")
                        load_capacity_counts = df1['적재용량'].value_counts().head(15)
                        
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
                        load_capacities = ["전체"] + df1['적재용량'].value_counts().index.tolist()
                        selected_capacity = st.selectbox("적재용량", load_capacities)
                        
                        if selected_capacity != "전체":
                            filtered_df_capacity = df1[df1['적재용량'] == selected_capacity]
                        else:
                            filtered_df_capacity = df1
                            
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
                if '마스트형' in df1.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 마스트별 AS 건수
                        st.subheader("마스트별 AS 건수")
                        mast_type_counts = df1['마스트형'].value_counts().head(15)
                        
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
                        mast_types = ["전체"] + df1['마스트형'].value_counts().index.tolist()
                        selected_mast = st.selectbox("마스트", mast_types)
                        
                        if selected_mast != "전체":
                            filtered_df_mast = df1[df1['마스트형'] == selected_mast]
                        else:
                            filtered_df_mast = df1
                            
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
        
        # 텍스트 데이터 확인
        if '정비내용' in df1.columns:
            # 정비내용 데이터 준비

            from kiwipiepy import Kiwi
            kiwi = Kiwi()

            text_data_raw = ' '.join(df1['정비내용'].dropna().astype(str))

            # 형태소 분석 + 명사 추출
            tokens = kiwi.tokenize(text_data_raw)
            nouns = [token.form for token in tokens if token.tag.startswith('N')]

            stopwords = ["및", "있음", "없음", "함", "을", "후", "함", "접수", "취소", "확인", "위해", "통해", "오류", "완료", "작업", "실시", "진행", "수리", '정상작동', '정상작동확인', '조치완료']

            # 불용어 제거
            filtered_nouns = [word for word in nouns if word not in stopwords and len(word) > 1]

            # 워드클라우드용 문자열 생성
            text_data = ' '.join(filtered_nouns)
            
            if not text_data:
                st.warning("정비내용 데이터가 없습니다.")
            else:
                # 그래프 행 1: 전체 워드클라우드와 분류별 워드클라우드
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("전체 정비내용 워드클라우드")
                    
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
                
                if all(col in df1.columns for col in ['작업유형', '정비대상', '정비작업', '정비내용']):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # 문자열 변환으로 정렬 오류 방지
                        categories = ["전체"] + sorted(convert_to_str_list(df1['작업유형'].dropna().unique()))
                        selected_category = st.selectbox("작업유형", categories)
                    
                    # 선택된 작업유형에 따라 데이터 필터링
                    if selected_category != "전체":
                        filtered_df = df1[df1['작업유형'].astype(str) == selected_category]
                        
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
                        filtered_df = df1
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
        interval_model, category_model, subcategory_model, detail_model, le_brand, le_model, le_category, le_subcategory, le_detail, le_year_range = prepare_prediction_model(df1)
        
        if interval_model is not None:
            st.info("다음 고장 시기 예측과 확률이 높은 고장 유형을 예측합니다.")
            
        col1, col2, col3, col4, col5 = st.columns(5)

        # 브랜드 목록 정의 - 특정 브랜드를 우선순위로 배치하고 나머지는 알파벳 순서로
        priority_brands = ['도요타', '두산', '현대', '클라크']
        
        # 우선순위 브랜드 목록 생성
        brand_list = [brand for brand in priority_brands if brand in df1['브랜드'].unique()]
        
        # 나머지 브랜드 추가 (우선순위와 '기타'를 제외하고 정렬)
        other_brands = sorted([brand for brand in df1['브랜드'].unique() 
                              if brand not in priority_brands and brand != '기타'])
        brand_list.extend(other_brands)
        
        # '기타'가 있으면 마지막에 추가
        if '기타' in df1['브랜드'].unique():
            brand_list.append('기타')
            
        with col1:
            selected_brand = st.selectbox("브랜드(필수)", brand_list)

        with col2:
            brand_models = df1[df1['브랜드'] == selected_brand]['모델명'].unique()
            selected_model = st.selectbox("모델(필수)", brand_models)

        # 브랜드/모델 선택 이후 필터링
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


            # 브랜드 + 모델 기준 1차 필터링
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
                    st.write(f"**현장명:** {latest_record.get('현장', '정보 없음')}")
        
            # 예측 실행
            if st.button("고장 예측 실행"):
                with st.spinner("예측 분석 중..."):
                    try:
                        # 선택한 값을 인코딩
                        brand_code = le_brand.transform([selected_brand])[0]
                        model_code = le_model.transform([selected_model])[0]

                        # 최근 정비 데이터 가져오기
                        latest_data = df1[(df1['브랜드'] == selected_brand) & (df1['모델명'] == selected_model)] \
                            .sort_values('정비일자', ascending=False).iloc[0]

                        category_code = le_category.transform([latest_data['작업유형']])[0]
                        subcat_code = le_subcategory.transform([latest_data['정비대상']])[0]
                        detail_code = le_detail.transform([latest_data['정비작업']])[0]

                        # AS 처리일수 평균
                        avg_repair_days = df1[(df1['브랜드'] == selected_brand) & (df1['모델명'] == selected_model)]['AS처리일수'].mean()
                        if pd.isna(avg_repair_days):
                            avg_repair_days = df1['AS처리일수'].mean()

                        # 제조년도 구간 → 인코딩
                        if selected_year_range == "전체":
                            mode_range = df1['제조년도'].dropna().astype(int).apply(year_to_range).mode().iloc[0]
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
