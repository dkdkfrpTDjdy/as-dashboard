## 5. pages/02_고장_유형_분석.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import create_figure, get_image_download_link, get_color_theme
from utils.data_processing import convert_to_str_list

# 페이지 설정
st.set_page_config(
    page_title="고장 유형 분석",
    layout="wide"
)

# 색상 테마 설정
current_theme = get_color_theme("고장 유형 분석")

# 페이지 제목
st.title("고장 유형 분석")

# 세션 상태 확인
if 'df1_with_costs' not in st.session_state:
    st.warning("정비일지 데이터가 로드되지 않았습니다. 홈 화면에서 데이터를 업로드해 주세요.")
    st.stop()

# 데이터 불러오기
df1 = st.session_state.df1_with_costs

def display_fault_analysis(df, maintenance_type=None):
    if maintenance_type and maintenance_type != "전체":
        filtered_df = df[df['정비구분'] == maintenance_type].copy()
        title_prefix = f"{maintenance_type} "
    else:
        filtered_df = df.copy()
        title_prefix = ""

    # 고장유형 컬럼 생성 및 NaN 제거
    if '고장유형' not in filtered_df.columns and all(col in filtered_df.columns for col in ['작업유형', '정비대상', '정비작업']):
        filtered_df['고장유형'] = filtered_df[['작업유형', '정비대상', '정비작업']].astype(str).agg('_'.join, axis=1)
    
    filtered_df['고장유형'].replace('nan_nan_nan', np.nan, inplace=True)
    filtered_df.dropna(subset=['고장유형'], inplace=True)

    if not all(col in filtered_df.columns for col in ['작업유형', '정비대상', '정비작업']):
        st.warning("고장 유형 분석에 필요한 컬럼(작업유형, 정비대상, 정비작업)이 누락되었습니다.")
        return

    category_tabs = {
        "작업유형": "작업유형",
        "정비대상": "정비대상",
        "정비작업": "정비작업"
    }

    tabs = st.tabs(list(category_tabs.keys()))

    for tab_idx, (tab_name, colname) in enumerate(category_tabs.items()):
        with tabs[tab_idx]:
            st.subheader(f"{title_prefix}{colname}")

            category_counts = filtered_df[colname].value_counts().head(15)
            category_values = sorted([str(v) for v in filtered_df[colname].dropna().unique()])
            selected_category = st.selectbox(
                f"{colname} 선택",
                ["전체"] + category_values,
                key=f"sel_{colname}_{maintenance_type}"
            )

            tab_filtered_df = filtered_df.copy()
            if selected_category != "전체":
                tab_filtered_df = tab_filtered_df[tab_filtered_df[colname].astype(str) == selected_category]

            if '고장유형' not in tab_filtered_df.columns or '브랜드_모델' not in tab_filtered_df.columns:
                st.warning("고장유형 또는 브랜드_모델 컬럼이 누락되어 있습니다.")
                continue

            # 히트맵용 필터
            top_faults = tab_filtered_df['고장유형'].value_counts().nlargest(15).index
            top_combos = tab_filtered_df['브랜드_모델'].value_counts().nlargest(15).index
            df_filtered = tab_filtered_df[
                tab_filtered_df['고장유형'].isin(top_faults) &
                tab_filtered_df['브랜드_모델'].isin(top_combos)
            ]

            col1, col2, col3 = st.columns(3)

            # 분포
            with col1:
                st.markdown(f"**{colname} 분포**")
                if not category_counts.empty:
                    fig1, ax1 = create_figure(figsize=(8, 8), dpi=150)
                    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax1, palette=f"{current_theme}_r")
                    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
                    for i, v in enumerate(category_counts.values):
                        ax1.text(i, v + max(category_counts.values) * 0.01, str(v), ha='center', fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig1, use_container_width=True)
                    st.markdown(get_image_download_link(fig1, f'{title_prefix}고장유형_{colname}_분포.png', f'{colname} 분포 다운로드'), unsafe_allow_html=True)
                else:
                    st.info("분포 데이터가 없습니다.")

            # 비율
            with col2:
                st.markdown(f"**{colname} 비율**")
                if not category_counts.empty:
                    category_counts_ratio = category_counts / category_counts.sum()
                    small = category_counts_ratio[category_counts_ratio < 0.05]
                    grouped = category_counts[category_counts_ratio >= 0.05].copy()
                    if not small.empty:
                        grouped['기타'] = int(small.sum() * category_counts.sum())
                    fig2, ax2 = create_figure(figsize=(8, 8), dpi=150)
                    grouped.plot.pie(
                        y=None, labels=grouped.index, autopct='%1.1f%%', ax=ax2,
                        colors=sns.color_palette(current_theme, n_colors=len(grouped))
                    )
                    ax2.set_ylabel('')
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    st.markdown(get_image_download_link(fig2, f'{title_prefix}고장유형_{colname}_비율.png', f'{colname} 비율 다운로드'), unsafe_allow_html=True)
                else:
                    st.info("비율 데이터가 없습니다.")

            # 히트맵
            with col3:
                st.markdown(f"**{colname}에 따른 고장 증상**")
                if len(df_filtered) > 0:
                    pivot_df = df_filtered.pivot_table(index='고장유형', columns='브랜드_모델', aggfunc='size', fill_value=0)
                    if pivot_df.shape[0] > 10 or pivot_df.shape[1] > 10:
                        row_sums = pivot_df.sum(axis=1).nlargest(10).index
                        col_sums = pivot_df.sum(axis=0).nlargest(10).index
                        pivot_df = pivot_df.loc[row_sums, col_sums]

                    fig3, ax3 = create_figure(figsize=(8, 8), dpi=150)
                    sns.heatmap(pivot_df, cmap=current_theme, annot=True, fmt='d', linewidths=0.5, ax=ax3, cbar=False)
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig3, use_container_width=True)
                    st.markdown(get_image_download_link(fig3, f'{title_prefix}고장유형_{colname}_히트맵.png', f'{colname} 히트맵 다운로드'), unsafe_allow_html=True)
                else:
                    st.info("히트맵을 생성할 데이터가 충분하지 않습니다.")
        
        # 자재내역 분석 섹션 추가
        with st.expander("장비 특성별 분석", expanded=True):
            st.subheader(f"{title_prefix}모델 타입 분석")
            
            # 분석 항목 선택
            model_type_options = []
            for col_name in ['연료', '운전방식', '적재용량', '마스트']:
                if col_name in filtered_df.columns and filtered_df[col_name].notna().any():
                    model_type_options.append(col_name)
            
            if model_type_options:
                selected_type = st.selectbox(
                    "분석 항목 선택",
                    model_type_options,
                    key=f"model_type_selector_{maintenance_type or '전체'}"
                )
                
                if selected_type:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 선택된 항목별 AS 건수
                        st.subheader(f"{title_prefix}{selected_type}별 AS 건수")
                        type_counts = filtered_df[selected_type].value_counts().dropna()
                        
                        if len(type_counts) > 0:
                            # 상위 15개만 시각화
                            if len(type_counts) > 15:
                                type_counts = type_counts.head(15)
                            
                            fig, ax = create_figure(figsize=(10, 8), dpi=150)
                            sns.barplot(x=type_counts.index, y=type_counts.values, ax=ax, palette=f"{current_theme}_r")
                            
                            # 막대 위에 텍스트 표시
                            for i, v in enumerate(type_counts.values):
                                ax.text(i, v + max(type_counts.values) * 0.01, str(v), ha='center', fontsize=10)
                                
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            st.markdown(get_image_download_link(fig, f'{title_prefix}{selected_type}별_AS_건수.png', f'{selected_type}별 AS 건수 다운로드'), unsafe_allow_html=True)
                        else:
                            st.warning(f"{selected_type} 데이터가 없습니다.")
                    
                    with col2:
                        # 선택된 항목별 고장유형 분석
                        st.subheader(f"{title_prefix}{selected_type}별 고장유형")
                        
                        # 해당 항목 값 선택
                        type_values = ["전체"] + filtered_df[selected_type].value_counts().index.tolist()
                        selected_value = st.selectbox(selected_type, type_values, key=f"{selected_type}_{maintenance_type}")
                        
                        if selected_value != "전체":
                            filtered_df_type = filtered_df[filtered_df[selected_type] == selected_value]
                        else:
                            filtered_df_type = filtered_df
                            
                        if '고장유형' in filtered_df_type.columns:
                            top_faults_by_type = filtered_df_type['고장유형'].value_counts().head(10)
                            
                            if not top_faults_by_type.empty:
                                fig, ax = create_figure(figsize=(10, 8), dpi=150)
                                sns.barplot(x=top_faults_by_type.values, y=top_faults_by_type.index, ax=ax, palette=f"{current_theme}_r")
                                
                                # 막대 위에 텍스트 표시
                                for i, v in enumerate(top_faults_by_type.values):
                                    ax.text(v + max(top_faults_by_type.values) * 0.002, i, str(v), va='center', fontsize=10)
                                    
                                plt.tight_layout()
                                st.pyplot(fig, use_container_width=True)
                                st.markdown(get_image_download_link(fig, f'{title_prefix}{selected_value}_고장유형_TOP10.png', f'{selected_value} 고장유형 TOP10 다운로드'), unsafe_allow_html=True)
                            else:
                                st.info(f"선택한 {selected_type}({selected_value})에 대한 고장유형 데이터가 없습니다.")
                        else:
                            st.warning("고장유형 데이터가 없습니다.")
            else:
                st.warning("장비 특성 데이터(연료, 운전방식, 적재용량, 마스트)가 없습니다.")
        
        # 상위 고장 유형 데이터 테이블
        with st.expander("상위 고장 유형 목록", expanded=False):
            st.subheader(f"{title_prefix}상위 고장 유형")
            if '고장유형' in filtered_df.columns:
                top_40_faults = filtered_df['고장유형'].value_counts().nlargest(40)
                fault_df = pd.DataFrame({
                    '고장유형': top_40_faults.index,
                    '건수': top_40_faults.values
                })
                st.dataframe(fault_df, use_container_width=True)
            else:
                st.warning("고장유형 데이터가 없습니다.")
    else:
        st.warning("고장 유형 분석에 필요한 컬럼(작업유형, 정비대상, 정비작업)이 데이터에 없습니다.")

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
