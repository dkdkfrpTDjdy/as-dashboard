## 7. pages/04_고장_예측.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.visualization import create_figure, get_image_download_link, get_color_theme

# 페이지 설정
st.set_page_config(
    page_title="고장 예측",
    layout="wide"
)

# 색상 테마 설정
current_theme = get_color_theme("고장 예측")

# 페이지 제목
st.title("고장 예측")

# 세션 상태 확인
if 'df1_with_costs' not in st.session_state:
    st.warning("정비일지 데이터가 로드되지 않았습니다. 홈 화면에서 데이터를 업로드해 주세요.")
    st.stop()

# 데이터 불러오기
df1 = st.session_state.df1_with_costs

# 고장 예측 모델 준비 함수
@st.cache_resource
def prepare_prediction_model(df):
    try:
        # 필수 컬럼 체크
        required_cols = ['브랜드', '모델명', '작업유형', '정비대상', '정비작업', '제조년도']
        if not all(col in df.columns for col in required_cols):
            st.warning(f"필수 컬럼이 누락되었습니다. 필요한 컬럼: {', '.join(required_cols)}")
            return [None] * 10

        # 필요한 컬럼만 선택하고 결측치 제거
        model_df = df.dropna(subset=required_cols[:-1]).copy()  # 제조년도는 후처리로

        # 충분한 데이터가 있는지 확인
        if len(model_df) < 100:
            st.warning(f"모델 학습에 필요한 데이터가 부족합니다. (현재 {len(model_df)}개, 최소 100개 필요)")
            return [None] * 10

        # 범주형 인코딩
        le_brand = LabelEncoder()
        le_model = LabelEncoder()
        le_category = LabelEncoder()
        le_subcategory = LabelEncoder()
        le_detail = LabelEncoder()

        # 각 특성 인코딩
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

        # 제조년도를 구간으로 변환
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

        # 타겟 - 재정비간격
        model_df['재정비간격_타겟'] = model_df['재정비간격'].fillna(365).clip(0, 365)
        X = model_df[features]
        y_interval = model_df['재정비간격_타겟']

        # 회귀 모델 학습 (재정비 간격 예측)
        X_train, X_test, y_train, y_test = train_test_split(X, y_interval, test_size=0.2, random_state=42)
        rf_interval_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_interval_model.fit(X_train, y_train)

        # 분류 모델들 학습 (고장 유형 예측)
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
        import traceback
        st.error(traceback.format_exc())
        return [None] * 10

# 모델 준비
with st.spinner("예측 모델을 준비 중입니다. 잠시만 기다려주세요..."):
    models = prepare_prediction_model(df1)
    interval_model, category_model, subcategory_model, detail_model, le_brand, le_model, le_category, le_subcategory, le_detail, le_year_range = models

# 모델이 준비되었는지 확인
models_ready = interval_model is not None

# 모델 상태에 따른 안내 메시지
if models_ready:
    st.info("다음 고장 시기 예측과 확률이 높은 고장 유형을 예측합니다.")
else:
    st.warning("""
    예측 모델을 준비할 수 없습니다. 다음 사항을 확인해주세요:
    1. 충분한 데이터(최소 100개 이상의 기록)가 있는지 확인
    2. 필요한 컬럼(브랜드, 모델명, 작업유형, 정비대상, 정비작업)이 모두 있는지 확인
    3. 재정비 간격 정보가 있는지 확인
    """)

# 장비 선택 UI
st.subheader("장비 선택")

# 5개 컬럼으로 레이아웃 구성
col1, col2, col3, col4, col5 = st.columns(5)

# 브랜드 목록 정의 - 특정 브랜드를 우선순위로 배치
with col1:
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
        
    # 브랜드 선택
    selected_brand = st.selectbox("브랜드(필수)", brand_list)

with col2:
    # 선택된 브랜드의 모델 목록
    if selected_brand:
        brand_models = sorted(df1[df1['브랜드'] == selected_brand]['모델명'].unique())
        selected_model = st.selectbox("모델(필수)", brand_models)
    else:
        selected_model = None

# 브랜드/모델 선택 이후 필터링
if selected_brand and selected_model:
    filtered_df = df1[(df1['브랜드'] == selected_brand) & (df1['모델명'] == selected_model)]
    
    if not filtered_df.empty:
        with col3:
            # 관리번호 선택
            existing_ids = filtered_df['관리번호'].dropna().unique()
            selected_id = st.selectbox("관리번호(선택)", ["전체"] + list(existing_ids), index=0)

            # 관리번호 기준 추가 필터링
            if selected_id != "전체":
                filtered_df = filtered_df[filtered_df['관리번호'] == selected_id]
        
        with col4:
            # 관리번호 직접 입력
            id_placeholder = f"예: {existing_ids[0]}" if len(existing_ids) > 0 else ""
            input_id = st.text_input("관리번호(직접 입력)", placeholder=id_placeholder).strip()
        
        with col5:
            # 제조년도 선택
            if '제조년도' in filtered_df.columns and filtered_df['제조년도'].notna().any():
                years = filtered_df['제조년도'].dropna().astype(int)

                def year_to_range(year):
                    if year <= 2005: return "2005년 이하"
                    elif year <= 2010: return "2006-2010"
                    elif year <= 2015: return "2011-2015"
                    elif year <= 2020: return "2016-2020"
                    else: return "2021-2025"

                year_ranges = sorted(set(year_to_range(y) for y in years))
                year_ranges = ["전체"] + year_ranges
                selected_year_range = st.selectbox("제조년도(선택)", year_ranges, index=0)
            else:
                selected_year_range = "전체"

        # 제조년도 범위로 필터링
        if selected_year_range != "전체":
            def year_in_range(year):
                if selected_year_range == "2005년 이하": return year <= 2005
                elif selected_year_range == "2006-2010": return 2006 <= year <= 2010
                elif selected_year_range == "2011-2015": return 2011 <= year <= 2015
                elif selected_year_range == "2016-2020": return 2016 <= year <= 2020
                elif selected_year_range == "2021-2025": return 2021 <= year <= 2025
                return False

            filtered_df = filtered_df[filtered_df['제조년도'].dropna().astype(int).apply(year_in_range)]

        # 최근 정비 정보 표시
        if len(filtered_df) > 0:
            # 정비일자로 정렬하여 최신 정보 가져오기
            latest_record = filtered_df.sort_values('정비일자', ascending=False).iloc[0]

            st.markdown("---")
            st.subheader("장비 최근 정비 정보")
            
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**최근 정비일:** {latest_record['정비일자'].strftime('%Y-%m-%d')}")
                st.write(f"**고장 유형:** {latest_record['작업유형']} > {latest_record['정비대상']} > {latest_record['정비작업']}")
                st.write(f"**종류:** {latest_record.get('자재내역', '정보 없음')}")
                st.write(f"**정비사:** {latest_record.get('정비자', '정보 없음')}")

            with col2:
                # 최근 정비일자 표시 (있는 경우만)
                if pd.notna(latest_record.get('최근정비일자')):
                    st.write(f"**이전 정비일:** {latest_record['최근정비일자'].strftime('%Y-%m-%d')}")
                else:
                    st.write("**이전 정비일:** 정보 없음")
                
                # 정비 내용 표시
                if '정비내용' in latest_record and pd.notna(latest_record['정비내용']):
                    st.write(f"**정비 내용:** {latest_record['정비내용']}")
                else:
                    st.write("**정비 내용:** 정보 없음")
                
                # 현장명 표시
                if '현장명' in latest_record and pd.notna(latest_record['현장명']):
                    st.write(f"**현장명:** {latest_record['현장명']}")
                else:
                    st.write("**현장명:** 정보 없음")
            
            # 예측 실행 버튼
            predict_button = st.button("고장 예측 실행", type="primary")
            
            if predict_button:
                # 모델이 준비되지 않았으면 예측 시도하지 않음
                if not models_ready:
                    st.error("예측 모델이 준비되지 않았습니다. 데이터를 확인하세요.")
                else:
                    with st.spinner("예측 분석 중..."):
                        try:
                            # 선택한 값을 인코딩
                            brand_code = le_brand.transform([selected_brand])[0]
                            model_code = le_model.transform([selected_model])[0]

                            # 최근 정비 데이터 가져오기
                            latest_data = filtered_df.sort_values('정비일자', ascending=False).iloc[0]

                            category_code = le_category.transform([latest_data['작업유형']])[0]
                            subcat_code = le_subcategory.transform([latest_data['정비대상']])[0]
                            detail_code = le_detail.transform([latest_data['정비작업']])[0]

                            # 제조년도 구간 → 인코딩
                            if selected_year_range == "전체":
                                mode_range = df1['제조년도'].dropna().astype(int).apply(lambda y: 
                                    "2005이하" if y <= 2005 
                                    else "2006-2010" if y <= 2010 
                                    else "2011-2015" if y <= 2015 
                                    else "2016-2020" if y <= 2020 
                                    else "2021-2025"
                                ).mode().iloc[0]
                            else:
                                # selected_year_range 형식을 모델에서 사용하는 형식으로 변환
                                mode_range = selected_year_range.replace("년 이하", "이하")
                            
                            year_range_encoded = le_year_range.transform([mode_range])[0]

                            # 예측할 데이터 준비
                            pred_data = np.array([[ 
                                brand_code, model_code, category_code, subcat_code, detail_code, 
                                year_range_encoded
                            ]])

                            # 예측 수행
                            predicted_days = interval_model.predict(pred_data)[0]
                            predicted_category_code = category_model.predict(pred_data)[0]
                            predicted_subcategory_code = subcategory_model.predict(pred_data)[0]
                            predicted_detail_code = detail_model.predict(pred_data)[0]

                            # 인코딩 값을 원래 값으로 변환
                            predicted_category = le_category.inverse_transform([predicted_category_code])[0]
                            predicted_subcategory = le_subcategory.inverse_transform([predicted_subcategory_code])[0]
                            predicted_detail = le_detail.inverse_transform([predicted_detail_code])[0]

                            # 결과 출력
                            st.success("**예측 분석 완료** ※ 해당 모델에 따른 예측모델 구현")
                            
                            st.markdown("---")
                            col1, col2 = st.columns(2)

                            with col1:
                                st.subheader("다음 정비 시기")
                                prediction_date = datetime.datetime.now() + datetime.timedelta(days=int(predicted_days))
                                st.markdown(f"""
                                **장비 정보**: {selected_brand} {selected_model}  
                                **예상 재정비 기간**: 약 **{int(predicted_days)}일** 후  
                                **예상 고장 날짜**: {prediction_date.strftime('%Y-%m-%d')}
                                """)

                                # 위험도 표시
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

                                st.markdown(f"<h3 style='color: {risk_color};'>재정비 위험도: {risk_level}</h3>", 
                                           unsafe_allow_html=True)

                            with col2:
                                st.subheader("고장 유형 예측")
                                st.markdown(f"""
                                **작업유형**: {predicted_category}  
                                **정비대상**: {predicted_subcategory}  
                                **정비작업**: {predicted_detail}
                                """)

                        except Exception as e:
                            st.error(f"예측 중 오류가 발생했습니다: {e}")
                            import traceback
                            st.error(traceback.format_exc())
                            st.info("선택한 데이터에 대한 학습 정보가 부족할 수 있습니다.")
    else:
        st.warning(f"선택한 브랜드 ({selected_brand})와 모델 ({selected_model})에 대한 데이터가 없습니다.")
else:
    st.info("브랜드와 모델을 선택하면 예측 기능을 사용할 수 있습니다.")
