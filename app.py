"""
5X Finder - 5년 내 5배 성장 종목 예측 시스템
Streamlit 시연용 앱 (02~04 노트북 결과 확인용)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# 페이지 설정
st.set_page_config(
    page_title="5X Finder",
    page_icon="📈",
    layout="wide"
)

# =============================================================================
# 경로 설정
# =============================================================================
# 현재 파일 기준 상대 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# =============================================================================
# 데이터 로드 함수
# =============================================================================
@st.cache_data
def load_data():
    dataset = pd.read_parquet(os.path.join(DATA_DIR, 'ml_dataset.parquet'))
    return dataset

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, 'final_model.joblib'))
    return model

@st.cache_data
def load_features():
    with open(os.path.join(DATA_DIR, 'feature_columns.txt'), 'r') as f:
        features = [line.strip() for line in f.readlines()]
    return features

# =============================================================================
# 사이드바
# =============================================================================
st.sidebar.title("5X Finder 📈")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "메뉴 선택",
    ["🏠 프로젝트 소개", "📊 데이터 수집", "🔧 Feature Engineering", "🤖 모델 학습"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**프로젝트:** 5X Finder  
**모델:** Logistic Regression  
**데이터:** S&P 500 (2010-2019)  
**샘플:** 4,652개  
**ROC-AUC:** 0.872  
**Recall:** 0.806
""")

# =============================================================================
# 1. 프로젝트 소개
# =============================================================================
if menu == "🏠 프로젝트 소개":
    st.title("5X Finder 📈")
    st.markdown("### 5년 내 5배 성장할 종목을 찾아주는 ML 시스템")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROC-AUC", "0.872")
    with col2:
        st.metric("Recall", "0.806", "31개 중 25개 발굴")
    with col3:
        st.metric("Precision", "0.111")
    with col4:
        st.metric("샘플 수", "4,652개")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 🎯 프로젝트 목표
        
        Baillie Gifford의 장기 성장주 투자 철학에서 영감을 받아,  
        **5년 내 5배(500%) 이상 성장할 가능성이 높은 종목**을 머신러닝으로 예측합니다.
        
        ## 📊 데이터 개요
        
        | 항목 | 내용 |
        |------|------|
        | 데이터 소스 | S&P 500 구성 종목 |
        | 수집 종목 | 502/503개 (99.8%) |
        | 기간 | 2010년 ~ 2019년 (10년) |
        | 샘플 수 | 4,652개 (종목 × 연도) |
        | Feature 수 | 26개 |
        | Target | 5년 후 5배 달성 여부 |
        """)
    
    with col2:
        st.markdown("""
        ## 🤖 모델 정보
        
        | 항목 | 내용 |
        |------|------|
        | 알고리즘 | Logistic Regression |
        | 선택 이유 | Recall 기준 최고 성능 |
        | ROC-AUC | 0.872 |
        | Recall | 0.806 (5배 종목 25/31개 발굴) |
        | Precision | 0.111 |
        
        ## 📈 핵심 인사이트
        
        | 발견 | 설명 |
        |------|------|
        | 복잡한 모델 ≠ 좋은 모델 | Logistic Regression이 XGBoost보다 Recall 높음 |
        | ROE 낮을수록 5배 확률 ↑ | 이미 성숙한 기업보다 성장 여력 있는 기업 |
        | 변동성 높을수록 5배 확률 ↑ | 고위험 고수익 |
        """)
    
    st.markdown("---")
    
    st.markdown("## 🔄 ML 파이프라인")
    
    pipeline_data = pd.DataFrame({
        '단계': ['02. 데이터 수집', '03. Feature Engineering', '04. 모델 학습'],
        '내용': ['S&P 500 502개 종목\nyfinance API', '26개 재무 지표\nSMOTE 클래스 균형', '5개 모델 비교\nLogistic Regression 선택'],
        '결과': ['99.8% 커버리지', '4,652 샘플', 'ROC-AUC 0.872']
    })
    
    st.dataframe(pipeline_data, use_container_width=True, hide_index=True)

# =============================================================================
# 2. 데이터 수집 (02)
# =============================================================================
elif menu == "📊 데이터 수집 (02)":
    st.title("📊 데이터 수집 (02_data_collection)")
    
    try:
        dataset = load_data()
        features = load_features()
        data_loaded = True
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        data_loaded = False
    
    if data_loaded:
        tab1, tab2, tab3 = st.tabs(["📥 수집 현황", "📋 데이터 미리보기", "📊 Target 분포"])
        
        with tab1:
            st.markdown("### 데이터 수집 현황")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("S&P 500 종목", "503개", "전체")
            with col2:
                st.metric("수집 완료", "502개", "99.8%")
            with col3:
                st.metric("수집 실패", "1개", "WBA")
            
            st.markdown("---")
            
            st.markdown("### 연도별 사용 가능 종목 수 (동적 필터링)")
            
            # 실제 데이터에서 연도별 종목 수 계산
            year_counts = dataset.groupby('start_year')['ticker'].nunique().reset_index()
            year_counts.columns = ['연도', '종목 수']
            
            fig = px.bar(year_counts, x='연도', y='종목 수', 
                        title='연도별 사용 가능 종목 수',
                        text='종목 수',
                        color='종목 수',
                        color_continuous_scale='Blues')
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **📌 동적 필터링이란?**
            - 각 연도에 데이터가 존재하는 종목만 사용
            - 2010년 상장 TSLA → 2010년부터 포함
            - 2012년 상장 META → 2012년부터 포함
            - 미래 정보 누출 방지 (Data Leakage 방지)
            """)
        
        with tab2:
            st.markdown("### 데이터셋 개요")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 샘플 수", f"{len(dataset):,}개")
            with col2:
                st.metric("Feature 수", f"{len(features)}개")
            with col3:
                st.metric("5배 달성 종목", f"{dataset['target_5x'].sum()}개")
            with col4:
                st.metric("5배 달성률", f"{dataset['target_5x'].mean()*100:.1f}%")
            
            st.markdown("---")
            st.markdown("### 데이터 미리보기")
            st.dataframe(dataset.head(20), use_container_width=True)
        
        with tab3:
            st.markdown("### Target 분포 (5배 달성 여부)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                target_counts = dataset['target_5x'].value_counts()
                fig = px.pie(
                    values=target_counts.values,
                    names=[f'미달성 ({target_counts[0]:,}개)', f'5배 달성 ({target_counts[1]:,}개)'],
                    title='전체 Target 분포',
                    color_discrete_sequence=['#3498db', '#e74c3c']
                )
                fig.update_traces(textinfo='percent+value')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                yearly = dataset.groupby('start_year').agg({
                    'ticker': 'count',
                    'target_5x': 'sum'
                }).reset_index()
                yearly.columns = ['연도', '총 종목', '5배 달성']
                
                fig = px.bar(yearly, x='연도', y=['총 종목', '5배 달성'], barmode='group',
                            title='연도별 종목 수 및 5배 달성',
                            color_discrete_sequence=['#3498db', '#e74c3c'])
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 3. Feature Engineering (03)
# =============================================================================
elif menu == "🔧 Feature Engineering (03)":
    st.title("🔧 Feature Engineering (03_feature_engineering)")
    
    try:
        dataset = load_data()
        features = load_features()
        data_loaded = True
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        data_loaded = False
    
    if data_loaded:
        tab1, tab2, tab3 = st.tabs(["📊 데이터셋 요약", "📋 Feature 목록", "🔥 상관관계"])
        
        with tab1:
            st.markdown("### 📊 Feature Engineering 요약")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rolling 기간", "2010-2019", "10년")
            with col2:
                st.metric("샘플 수", f"{len(dataset):,}개")
            with col3:
                st.metric("Feature 수", f"{len(features)}개")
            with col4:
                st.metric("5배 달성 비율", f"{dataset['target_5x'].mean()*100:.1f}%")
            
            st.markdown("---")
            
            st.markdown("""
            ### 💡 핵심 포인트
            
            | 항목 | 설명 |
            |------|------|
            | **동적 필터링** | 각 연도에 데이터가 존재하는 종목만 사용 |
            | **SMOTE** | 클래스 불균형 해소 (4.6% → 50%) |
            | **시간 기반 분할** | Train 2010-2017, Test 2018-2019 |
            """)
        
        with tab2:
            st.markdown("### 📋 Feature 목록 (26개)")
            
            # 카테고리별 분류
            feature_categories = {
                '📈 성장성': ['revenue_cagr_3y'],
                '💰 수익성': ['gross_margin', 'operating_margin', 'fcf_margin', 'operating_margin_trend'],
                '📊 효율성': ['roe', 'roa', 'roic'],
                '🏭 투자': ['capex_to_revenue', 'capex_to_depreciation', 'reinvestment_rate'],
                '🏦 재무 안정성': ['debt_to_equity', 'interest_coverage', 'current_ratio'],
                '✅ 품질': ['fcf_positive_years', 'earnings_quality'],
                '💵 밸류에이션': ['ps_ratio', 'pe_ratio', 'pb_ratio', 'peg_ratio', 'fcf_yield'],
                '📉 가격 모멘텀': ['price_momentum_12m', 'volatility_1y', 'volatility_3m', 'price_to_sma_50', 'price_to_sma_200']
            }
            
            for category, feats in feature_categories.items():
                with st.expander(f"{category} ({len(feats)}개)"):
                    for feat in feats:
                        if feat in features:
                            st.markdown(f"- `{feat}` ✅")
                        else:
                            st.markdown(f"- `{feat}` ❌ (제외됨)")
            
            st.markdown("---")
            
            # Feature 통계
            st.markdown("#### Feature 통계")
            available_features = [f for f in features if f in dataset.columns]
            if available_features:
                feature_stats = dataset[available_features].describe().T[['mean', 'std', 'min', '50%', 'max']]
                feature_stats.columns = ['평균', '표준편차', '최소', '중앙값', '최대']
                st.dataframe(feature_stats.round(3), use_container_width=True)
        
        with tab3:
            st.markdown("### 🔥 Feature 상관관계 히트맵")
            
            available_features = [f for f in features if f in dataset.columns]
            if available_features:
                corr_matrix = dataset[available_features].corr()
                
                fig = px.imshow(corr_matrix,
                               labels=dict(color="상관계수"),
                               x=available_features, y=available_features,
                               color_continuous_scale='RdBu_r',
                               aspect='auto')
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 4. 모델 학습 (04)
# =============================================================================
elif menu == "🤖 모델 학습 (04)":
    st.title("🤖 모델 학습 (04_model_training)")
    
    try:
        model = load_model()
        dataset = load_data()
        features = load_features()
        data_loaded = True
    except Exception as e:
        st.error(f"데이터/모델 로드 실패: {e}")
        data_loaded = False
    
    if data_loaded:
        tab1, tab2, tab3 = st.tabs(["📊 모델 비교", "📈 성능 지표", "🎯 Feature Importance"])
        
        with tab1:
            st.markdown("### 5개 모델 점진적 성능 비교")
            
            st.markdown("""
            **학습 순서:** 단순한 모델 → 복잡한 모델
            - 베이스라인(Logistic Regression)부터 시작
            - 점진적으로 복잡한 모델로 성능 개선 시도
            """)
            
            st.markdown("---")
            
            # 실제 04 노트북 결과
            model_comparison = pd.DataFrame({
                'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
                'ROC-AUC': [0.872, 0.873, 0.882, 0.887, 0.878],
                'Recall': [0.806, 0.774, 0.677, 0.645, 0.645],
                'Precision': [0.111, 0.171, 0.169, 0.180, 0.196]
            })
            
            st.dataframe(model_comparison, use_container_width=True, hide_index=True)
            
            # 점진적 성능 변화 시각화
            st.markdown("### 점진적 모델 성능 변화")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=model_comparison['Model'],
                    y=model_comparison['ROC-AUC'],
                    mode='lines+markers+text',
                    marker=dict(size=12, color='#3498db'),
                    line=dict(width=2, color='#3498db'),
                    text=model_comparison['ROC-AUC'].round(3),
                    textposition='top center',
                    name='ROC-AUC'
                ))
                fig.update_layout(title='ROC-AUC 변화', height=400)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=model_comparison['Model'],
                    y=model_comparison['Recall'],
                    mode='lines+markers+text',
                    marker=dict(size=12, color='#e74c3c'),
                    line=dict(width=2, color='#e74c3c'),
                    text=model_comparison['Recall'].round(3),
                    textposition='top center',
                    name='Recall'
                ))
                fig.update_layout(title='Recall 변화', height=400)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### 💡 핵심 발견
            
            | 발견 | 설명 |
            |------|------|
            | **ROC-AUC** | 복잡한 모델일수록 약간 상승 (0.872 → 0.887) |
            | **Recall** | 복잡한 모델일수록 **하락** (0.806 → 0.645) |
            | **결론** | 복잡한 모델 ≠ 더 좋은 모델! |
            """)
        
        with tab2:
            st.markdown("### 최종 모델 성능")
            
            st.markdown("""
            **✅ 최종 선택: Logistic Regression**
            - 선택 이유: Recall 기준 (5배 종목을 놓치지 않는 것이 중요)
            """)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ROC-AUC", "0.872")
            with col2:
                st.metric("Recall", "0.806", "31개 중 25개")
            with col3:
                st.metric("Precision", "0.111")
            with col4:
                st.metric("F1 Score", "0.195")
            
            st.markdown("---")
            
            # Confusion Matrix
            st.markdown("### Confusion Matrix")
            
            cm_data = [[739, 201], [6, 25]]
            
            fig = go.Figure(data=go.Heatmap(
                z=cm_data,
                x=['예측: 미달성', '예측: 5배 달성'],
                y=['실제: 미달성', '실제: 5배 달성'],
                text=cm_data,
                texttemplate='%{text}',
                textfont={'size': 20},
                colorscale='Blues'
            ))
            fig.update_layout(height=400, width=500)
            st.plotly_chart(fig)
            
            st.markdown("""
            | 항목 | 숫자 | 의미 |
            |------|------|------|
            | **25** | True Positive | 5배 종목 25개 정확히 찾음 ✅ |
            | **6** | False Negative | 5배 종목 6개 놓침 ❌ |
            | **201** | False Positive | 헛발질 (5배 아닌데 5배라고 예측) |
            | **739** | True Negative | 정확히 걸러냄 |
            """)
        
        with tab3:
            st.markdown("### Feature Importance (Logistic Regression 계수)")
            
            st.markdown("""
            **📌 해석 방법:**
            - **양수(+)**: 값이 높을수록 5배 달성 확률 ↑
            - **음수(-)**: 값이 높을수록 5배 달성 확률 ↓
            """)
            
            # Logistic Regression 계수 (04 노트북 결과)
            coef_data = pd.DataFrame({
                'feature': ['roe', 'volatility_1y', 'pb_ratio', 'roa', 'reinvestment_rate', 
                           'operating_margin_trend', 'fcf_yield', 'earnings_quality', 'pe_ratio', 'operating_margin'],
                'coefficient': [-2.037, 0.947, 0.870, 0.746, 0.507, -0.505, 0.427, -0.417, 0.371, 0.367]
            })
            
            # 절대값 기준 정렬
            coef_data['abs_coef'] = coef_data['coefficient'].abs()
            coef_data = coef_data.sort_values('abs_coef', ascending=True)
            
            # 색상: 양수=초록, 음수=빨강
            colors = ['#27ae60' if x > 0 else '#e74c3c' for x in coef_data['coefficient']]
            
            fig = go.Figure(go.Bar(
                x=coef_data['coefficient'],
                y=coef_data['feature'],
                orientation='h',
                marker_color=colors,
                text=[f'{v:+.3f}' for v in coef_data['coefficient']],
                textposition='outside'
            ))
            fig.update_layout(
                title='Logistic Regression Feature Coefficients (Top 10)',
                xaxis_title='Coefficient',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### 💡 주요 인사이트
            
            | Feature | 계수 | 해석 |
            |---------|------|------|
            | **roe** | -2.04 | ROE 낮을수록 5배 확률 ↑ (성장 여력) |
            | **volatility_1y** | +0.95 | 변동성 높을수록 5배 확률 ↑ (고위험 고수익) |
            | **pb_ratio** | +0.87 | PBR 높을수록 5배 확률 ↑ (시장 기대) |
            | **roa** | +0.75 | ROA 높을수록 5배 확률 ↑ |
            """)

# =============================================================================
# 푸터
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("© 2024 5X Finder")