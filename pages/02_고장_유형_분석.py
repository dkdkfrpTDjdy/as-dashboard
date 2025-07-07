import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import create_figure, get_image_download_link, get_color_theme
from utils.data_processing import convert_to_str_list

st.set_page_config(page_title="고장 유형 분석", layout="wide")
current_theme = get_color_theme("고장 유형 분석")
st.title("고장 유형 분석")

if 'df1_with_costs' not in st.session_state:
    st.warning("정비일지 데이터가 로드되지 않았습니다. 홈 화면에서 데이터를 업로드해 주세요.")
    st.stop()

df1 = st.session_state.df1_with_costs


def display_fault_analysis(df, maintenance_type=None):
    unique_prefix = f"fault_{maintenance_type or 'all'}"
    title_prefix = f"{maintenance_type} " if maintenance_type and maintenance_type != "전체" else ""
    filtered_df = df[df['정비구분'] == maintenance_type].copy() if maintenance_type and maintenance_type != "전체" else df.copy()

    required_cols = ['작업유형', '정비대상', '정비작업']
    if not all(col in filtered_df.columns for col in required_cols):
        st.warning("고장 유형 분석에 필요한 컬럼(작업유형, 정비대상, 정비작업)이 누락되었습니다.")
        return

    if '고장유형' not in filtered_df.columns:
        filtered_df['고장유형'] = filtered_df[required_cols].astype(str).agg('_'.join, axis=1)

    filtered_df['고장유형'].replace('nan_nan_nan', np.nan, inplace=True)
    filtered_df.dropna(subset=['고장유형'], inplace=True)

    tabs = st.tabs(["작업유형", "정비대상", "정비작업"])

    for idx, colname in enumerate(tabs):
        with tabs[idx]:
            st.subheader(f"{title_prefix}{colname}")
            category_counts = filtered_df[colname].value_counts().head(15)
            category_values = sorted([str(v) for v in filtered_df[colname].dropna().unique()])
            selected = st.selectbox(
                f"{colname} 선택", ["전체"] + category_values, key=f"{unique_prefix}_{colname}_select"
            )

            tab_df = filtered_df if selected == "전체" else filtered_df[filtered_df[colname].astype(str) == selected]

            if '브랜드_모델' not in tab_df.columns:
                st.warning("브랜드_모델 컬럼이 누락되어 있습니다.")
                continue

            top_faults = tab_df['고장유형'].value_counts().nlargest(15).index
            top_models = tab_df['브랜드_모델'].value_counts().nlargest(15).index
            heatmap_df = tab_df[tab_df['고장유형'].isin(top_faults) & tab_df['브랜드_모델'].isin(top_models)]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"**{colname} 분포**")
                if not category_counts.empty:
                    fig, ax = create_figure(figsize=(8, 8), dpi=150)
                    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax, palette=f"{current_theme}_r")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                    for i, v in enumerate(category_counts.values):
                        ax.text(i, v + v * 0.01, str(v), ha='center', fontsize=12)
                    st.pyplot(fig, use_container_width=True)
                    st.markdown(get_image_download_link(fig, f"{title_prefix}{colname}_분포.png", f"{colname} 분포 다운로드"), unsafe_allow_html=True)
                else:
                    st.info("분포 데이터가 없습니다.")

            with col2:
                st.markdown(f"**{colname} 비율**")
                if not category_counts.empty:
                    ratio = category_counts / category_counts.sum()
                    grouped = category_counts[ratio >= 0.05].copy()
                    if (small := ratio[ratio < 0.05]).any():
                        grouped['기타'] = int(small.sum() * category_counts.sum())
                    fig, ax = create_figure(figsize=(8, 8), dpi=150)
                    grouped.plot.pie(labels=grouped.index, autopct='%1.1f%%', ax=ax, colors=sns.color_palette(current_theme, n_colors=len(grouped)))
                    ax.set_ylabel('')
                    st.pyplot(fig, use_container_width=True)
                    st.markdown(get_image_download_link(fig, f"{title_prefix}{colname}_비율.png", f"{colname} 비율 다운로드"), unsafe_allow_html=True)
                else:
                    st.info("비율 데이터가 없습니다.")

            with col3:
                st.markdown(f"**{colname}에 따른 고장 증상**")
                if not heatmap_df.empty:
                    pivot = heatmap_df.pivot_table(index='고장유형', columns='브랜드_모델', aggfunc='size', fill_value=0)
                    if pivot.shape[0] > 10 or pivot.shape[1] > 10:
                        pivot = pivot.loc[pivot.sum(axis=1).nlargest(10).index, pivot.sum(axis=0).nlargest(10).index]
                    fig, ax = create_figure(figsize=(8, 8), dpi=150)
                    sns.heatmap(pivot, cmap=current_theme, annot=True, fmt='d', linewidths=0.5, ax=ax, cbar=False)
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=0)
                    st.pyplot(fig, use_container_width=True)
                    st.markdown(get_image_download_link(fig, f"{title_prefix}{colname}_히트맵.png", f"{colname} 히트맵 다운로드"), unsafe_allow_html=True)
                else:
                    st.info("히트맵을 생성할 데이터가 충분하지 않습니다.")

    with st.expander("장비 특성별 분석", expanded=True):
        st.subheader(f"{title_prefix}모델 타입 분석")
        candidates = ['연료', '운전방식', '적재용량', '마스트']
        options = [c for c in candidates if c in filtered_df.columns and filtered_df[c].notna().any()]

        if options:
            selected_type = st.selectbox("분석 항목 선택", options, key=f"{unique_prefix}_model_type_selector")
            if selected_type:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"{selected_type}별 AS 건수")
                    counts = filtered_df[selected_type].value_counts().dropna()
                    if not counts.empty:
                        if len(counts) > 15:
                            counts = counts.head(15)
                        fig, ax = create_figure(figsize=(10, 8), dpi=150)
                        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=f"{current_theme}_r")
                        for i, v in enumerate(counts.values):
                            ax.text(i, v + v * 0.01, str(v), ha='center', fontsize=10)
                        plt.xticks(rotation=45, ha='right')
                        st.pyplot(fig, use_container_width=True)
                        st.markdown(get_image_download_link(fig, f"{title_prefix}{selected_type}_AS_건수.png", f"{selected_type}별 AS 건수 다운로드"), unsafe_allow_html=True)
                    else:
                        st.warning(f"{selected_type} 데이터가 없습니다.")

                with col2:
                    st.subheader(f"{selected_type}별 고장유형")
                    values = ["전체"] + filtered_df[selected_type].value_counts().index.tolist()
                    selected_value = st.selectbox(selected_type, values, key=f"{unique_prefix}_{selected_type}_value")
                    df_type = filtered_df if selected_value == "전체" else filtered_df[filtered_df[selected_type] == selected_value]
                    if '고장유형' in df_type.columns:
                        top_faults = df_type['고장유형'].value_counts().head(10)
                        if not top_faults.empty:
                            fig, ax = create_figure(figsize=(10, 8), dpi=150)
                            sns.barplot(x=top_faults.values, y=top_faults.index, ax=ax, palette=f"{current_theme}_r")
                            for i, v in enumerate(top_faults.values):
                                ax.text(v + v * 0.002, i, str(v), va='center', fontsize=10)
                            st.pyplot(fig, use_container_width=True)
                            st.markdown(get_image_download_link(fig, f"{title_prefix}{selected_value}_고장유형_TOP10.png", f"{selected_value} 고장유형 TOP10 다운로드"), unsafe_allow_html=True)
                        else:
                            st.info(f"{selected_type}({selected_value})에 대한 고장유형 데이터가 없습니다.")
                    else:
                        st.warning("고장유형 데이터가 없습니다.")
        else:
            st.warning("장비 특성 데이터(연료, 운전방식, 적재용량, 마스트)가 없습니다.")

    with st.expander("상위 고장 유형 목록", expanded=False):
        st.subheader(f"{title_prefix}상위 고장 유형")
        if '고장유형' in filtered_df.columns:
            top_faults = filtered_df['고장유형'].value_counts().head(40)
            st.dataframe(pd.DataFrame({'고장유형': top_faults.index, '건수': top_faults.values}), use_container_width=True)
        else:
            st.warning("고장유형 데이터가 없습니다.")


# 메인 탭 구성
if '정비구분' in df1.columns:
    if df1['정비구분'].notna().any():
        maintenance_types = df1['정비구분'].dropna().unique()
        has_internal = '내부' in maintenance_types
        has_external = '외부' in maintenance_types

        tabs = st.tabs(["전체", "내부", "외부"])

        with tabs[0]:
            st.header("전체 고장 유형 분석")
            display_fault_analysis(df1, None)

        with tabs[1]:
            st.header("내부 고장 유형 분석")
            if has_internal:
                display_fault_analysis(df1, "내부")
            else:
                st.info("내부 정비 데이터가 없습니다.")

        with tabs[2]:
            st.header("외부 고장 유형 분석")
            if has_external:
                display_fault_analysis(df1, "외부")
            else:
                st.info("외부 정비 데이터가 없습니다.")
    else:
        st.info("정비구분 값이 없어서 전체 데이터를 분석합니다.")
        display_fault_analysis(df1, None)
else:
    st.info("정비구분 컬럼이 없어 전체 데이터를 분석합니다.")
    display_fault_analysis(df1, None)
