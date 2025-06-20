## 6. pages/03_브랜드_모델_분석.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import create_figure, get_image_download_link, get_color_theme
from utils.data_processing import group_small_categories

# 페이지 설정
st.set_page_config(
    page_title="브랜드/모델 분석",
    layout="wide"
)

# 색상 테마 설정
current_theme = get_color_theme("브랜드/모델 분석")

# 페이지 제목
st.title("브랜드 및 모델 분석")

# 세션 상태 확인
if 'df1_with_costs' not in st.session_state:
    st.warning("정비일지 데이터가 로드되지 않았습니다. 홈 화면에서 데이터를 업로드해 주세요.")
    st.stop()

# 데이터 불러오기
df1 = st.session_state.df1_with_costs

# 자산 데이터 확인
has_asset_data = 'df2' in st.session_state and st.session_state.df2 is not None

if has_asset_data:
    df2 = st.session_state.df2
    # 자산 데이터 기본 정보
    total_assets = len(df2)
    
    if '제조사명' in df2.columns:
        asset_brand_counts = df2['제조사명'].value_counts()
        # 비율 계산
        asset_brand_ratio = asset_brand_counts / total_assets * 100
    
    if '제조사모델명' in df2.columns:
        asset_model_counts = df2['제조사모델명'].value_counts()
        # 비율 계산
        asset_model_ratio = asset_model_counts / total_assets * 100
    
    if '제조년도' in df2.columns:
        asset_year_counts = df2['제조년도'].value_counts()
        # 비율 계산
        asset_year_ratio = asset_year_counts / total_assets * 100
else:
    st.info("자산조회 데이터가 없습니다. AS 데이터만 분석합니다.")

# AS 데이터 전체 건수
total_as = len(df1)

# 섹션 1: 브랜드 분석
st.header("브랜드 분석")

# 브랜드별 AS 건수 및 비율
if '브랜드' in df1.columns:
    brand_counts = df1['브랜드'].value_counts()
    brand_as_ratio = brand_counts / total_as * 100
    # 작은 비율 항목은 기타로 그룹화
    brand_as_ratio = group_small_categories(brand_as_ratio, threshold=0.03)
    brand_as_ratio = brand_as_ratio.nlargest(15)  # 상위 15개만 표시
    
    # 브랜드 분석 그래프 - 3개의 그래프를 나란히 배치
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 브랜드별 AS 비율 파이 차트
        st.subheader("브랜드별 AS 비율")
        
        fig, ax = create_figure(figsize=(6, 6), dpi=150)
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
        st.markdown(get_image_download_link(fig, '브랜드_AS_비율_파이.png', '파이차트 다운로드'), unsafe_allow_html=True)
    
    # 자산 데이터가 있는 경우 특정 브랜드에 대한 비교 그래프 표시
    if has_asset_data and '제조사명' in df2.columns:
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
                
                fig, ax = create_figure(figsize=(6, 6), dpi=150)
                
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
                st.markdown(get_image_download_link(fig, '주요_브랜드_비율_비교.png', '비교 그래프 다운로드'), unsafe_allow_html=True)
            
            with col3:
                # AS/자산 비율 그래프
                st.subheader("주요 브랜드 AS/자산 비율")
                
                fig, ax = create_figure(figsize=(6, 6), dpi=150)
                
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

if '브랜드' in df1.columns:
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
    if selected_brand != "전체" and '브랜드' in df1.columns:
        brand_df = df1[df1['브랜드'] == selected_brand]
        brand_total_as = len(brand_df)
        brand_title = f"{selected_brand} "
    else:
        brand_df = df1
        brand_total_as = total_as
        brand_title = "전체 "
        
    if '모델명' in brand_df.columns:
        # 모델별 AS 건수 및 비율
        model_counts = brand_df['모델명'].value_counts().head(15)  # 상위 10개만 (가독성)
        model_as_ratio = (model_counts / brand_total_as * 100).round(1)
        
        # 모델별 분석 그래프 (2개 그래프 나란히 배치)
        col1, col2 = st.columns(2)
        
        with col1:
            # 모델별 AS 비율 그래프
            st.subheader(f"{brand_title}모델별 AS 비율")
            
            fig, ax = create_figure(figsize=(7, 6), dpi=150)
            bars = sns.barplot(x=model_as_ratio.values, y=model_as_ratio.index, ax=ax, palette=f"{current_theme}_r")
            
            # 간결한 값 표시
            for i, v in enumerate(model_as_ratio.values):
                ax.text(v + 0.01, i, f"{v:.1f}%", va='center', fontsize=8)
                
            ax.set_xlabel('AS 비율 (%)')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown(get_image_download_link(fig, f'{selected_brand}_모델별_AS_비율.png', '모델별 AS 비율 다운로드'), unsafe_allow_html=True)
            
        with col2:
            # 고장 유형 분석 (있는 경우)
            if '고장유형' in brand_df.columns:
                st.subheader(f"{brand_title}고장 유형 분석")
                
                fault_counts = brand_df['고장유형'].value_counts().head(15)
                fault_ratio = (fault_counts / brand_total_as * 100).round(1)
                
                fig, ax = create_figure(figsize=(7, 6), dpi=150)
                bars = sns.barplot(x=fault_ratio.values, y=fault_ratio.index, ax=ax, palette=f"{current_theme}_r")
                
                # 간결한 값 표시
                for i, v in enumerate(fault_ratio.values):
                    ax.text(v + 0.01, i, f"{v:.1f}%", va='center', fontsize=8)
                    
                ax.set_xlabel('고장유형 비율 (%)')
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown(get_image_download_link(fig, f'{selected_brand}_고장유형_분석.png', '고장유형 분석 다운로드'), unsafe_allow_html=True)
        
        # 자산 데이터가 있는 경우 모델별 자산 대비 AS 비율 분석
        if has_asset_data and '제조사명' in df2.columns and '제조사모델명' in df2.columns:
            with st.expander("모델별 자산 대비 AS 비율 분석", expanded=True):
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
                        fig, ax = create_figure(figsize=(8, 7), dpi=150)
                        
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
                        st.markdown(get_image_download_link(fig, f'{selected_brand}_모델_비율_비교.png', '모델별 비율 비교 다운로드'), unsafe_allow_html=True)
                        
                    with col2:
                        # AS/자산 비율 그래프
                        fig, ax = create_figure(figsize=(8, 7), dpi=150)
                        
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
                        st.markdown(get_image_download_link(fig, f'{selected_brand}_모델_AS자산_비율.png', '모델별 AS/자산 비율 다운로드'), unsafe_allow_html=True)
    
    # 섹션 3: 제조년도별 분석
    if '제조년도' in df1.columns:
        with st.expander("제조년도별 분석", expanded=True):
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
                    
                    fig, ax = create_figure(figsize=(7, 6), dpi=150)
                    bars = sns.barplot(x=year_ratio.index.astype(str), y=year_ratio.values, 
                                     ax=ax, palette=f"{current_theme}")
                                     
                    # 간결한 값 표시
                    for i, v in enumerate(year_ratio.values):
                        ax.text(i, v + 0.2, f"{v:.1f}%", ha='center', fontsize=7)
                        
                    plt.xticks(rotation=45)
                    ax.set_ylabel('AS 비율 (%)')
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.markdown(get_image_download_link(fig, f'{selected_brand}_연식별_AS_비율.png', '연식별 AS 비율 다운로드'), unsafe_allow_html=True)
                    
                with col2:
                    # 제조년도별 평균 처리일수 또는 재정비간격
                    if 'AS처리일수' in df1.columns:
                        st.subheader(f"{brand_title}연식별 평균 처리일수")
                        
                        # 필요한 데이터만 추출
                        days_df = year_df.dropna(subset=['AS처리일수'])
                        
                        if len(days_df) > 0:
                            # 제조년도별 평균 처리일수 계산
                            year_avg_days = days_df.groupby('제조년도')['AS처리일수'].mean().round(1)
                            
                            fig, ax = create_figure(figsize=(7, 6), dpi=150)
                            bars = sns.barplot(x=year_avg_days.index.astype(str), y=year_avg_days.values, 
                                             ax=ax, palette=f"{current_theme}_r")
                                             
                            # 간결한 값 표시
                            for i, v in enumerate(year_avg_days.values):
                                ax.text(i, v + 0.02, f"{v:.1f}일", ha='center', fontsize=7)
                                
                            plt.xticks(rotation=45)
                            ax.set_ylabel('평균 처리일수')
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.markdown(get_image_download_link(fig, f'{selected_brand}_연식별_처리일수.png', '연식별 처리일수 다운로드'), unsafe_allow_html=True)
                    elif '재정비간격' in df1.columns:
                        st.subheader(f"{brand_title}연식별 평균 재정비간격")
                        
                        # 필요한 데이터만 추출
                        interval_df = year_df.dropna(subset=['재정비간격'])
                        
                        if len(interval_df) > 0:
                            # 제조년도별 평균 재정비간격 계산
                            year_avg_interval = interval_df.groupby('제조년도')['재정비간격'].mean().round(1)
                            
                            fig, ax = create_figure(figsize=(7, 6), dpi=150)
                            bars = sns.barplot(x=year_avg_interval.index.astype(str), y=year_avg_interval.values, 
                                             ax=ax, palette=f"{current_theme}_r")
                                             
                            # 간결한 값 표시
                            for i, v in enumerate(year_avg_interval.values):
                                ax.text(i, v + 0.02, f"{v:.1f}일", ha='center', fontsize=7)
                                
                            plt.xticks(rotation=45)
                            ax.set_ylabel('평균 재정비간격 (일)')
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.markdown(get_image_download_link(fig, f'{selected_brand}_연식별_재정비간격.png', '연식별 재정비간격 다운로드'), unsafe_allow_html=True)
                
                # 자산 데이터가 있는 경우 제조년도별 자산 대비 AS 비율 분석
                if has_asset_data and '제조년도' in df2.columns:
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
                                fig, ax = create_figure(figsize=(8, 6), dpi=150)
                                
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
                                    ax.text(i - width/2, v + 0.1, f"{v:.1f}%", ha='center', fontsize=7)
                                    
                                for i, v in enumerate(year_comparison['AS 비율(%)']):
                                    ax.text(i + width/2, v + 0.4, f"{v:.1f}%", ha='center', fontsize=7)
                                    
                                plt.tight_layout()
                                st.pyplot(fig)
                                st.markdown(get_image_download_link(fig, f'{selected_brand}_연식별_비율_비교.png', '연식별 비율 비교 다운로드'), unsafe_allow_html=True)
                                
                            with col2:
                                # AS/자산 비율 그래프
                                fig, ax = create_figure(figsize=(8, 6), dpi=150)
                                
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
                                st.markdown(get_image_download_link(fig, f'{selected_brand}_연식별_AS자산_비율.png', '연식별 AS/자산 비율 다운로드'), unsafe_allow_html=True)