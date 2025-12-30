"""
5X Finder - 5년 내 5배 성장 종목 예측 시스템
Streamlit 시연용 앱 (전체 인터랙티브 버전)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
# Feature 설명 딕셔너리
# =============================================================================
FEATURE_DESC = {
    'revenue_cagr_3y': ('성장성', '3년 매출 성장률 (CAGR)'),
    'gross_margin': ('수익성', '매출총이익률'),
    'operating_margin': ('수익성', '영업이익률'),
    'fcf_margin': ('수익성', '잉여현금흐름 마진'),
    'operating_margin_trend': ('수익성', '영업이익률 추세'),
    'roe': ('효율성', '자기자본이익률 (Return on Equity)'),
    'roa': ('효율성', '총자산이익률 (Return on Assets)'),
    'roic': ('효율성', '투하자본수익률'),
    'capex_to_revenue': ('투자', '설비투자/매출 비율'),
    'capex_to_depreciation': ('투자', '설비투자/감가상각 비율'),
    'reinvestment_rate': ('투자', '재투자율'),
    'debt_to_equity': ('재무안정성', '부채비율'),
    'interest_coverage': ('재무안정성', '이자보상배율'),
    'current_ratio': ('재무안정성', '유동비율'),
    'fcf_positive_years': ('품질', 'FCF 양수 연도 수'),
    'earnings_quality': ('품질', '이익의 질'),
    'ps_ratio': ('밸류에이션', '주가매출비율 (PSR)'),
    'pe_ratio': ('밸류에이션', '주가수익비율 (PER)'),
    'pb_ratio': ('밸류에이션', '주가순자산비율 (PBR)'),
    'peg_ratio': ('밸류에이션', 'PEG 비율'),
    'fcf_yield': ('밸류에이션', 'FCF 수익률'),
    'price_momentum_12m': ('모멘텀', '12개월 가격 모멘텀'),
    'volatility_1y': ('모멘텀', '1년 변동성'),
    'volatility_3m': ('모멘텀', '3개월 변동성'),
    'price_to_sma_50': ('모멘텀', '50일 이동평균 대비'),
    'price_to_sma_200': ('모멘텀', '200일 이동평균 대비')
}

# =============================================================================
# 사이드바
# =============================================================================
st.sidebar.title("5X Finder 📈")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "메뉴 선택",
    ["🏠 프로젝트 소개", "📊 데이터 수집", "🔧 Feature Engineering", "🤖 모델 학습", "🔍 종목 분석"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**모델:** Logistic Regression  
**데이터:** S&P 500 (2010-2019)  
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
        | 기간 | 2010년 ~ 2019년 |
        | 샘플 수 | 4,652개 |
        | Feature 수 | 26개 |
        """)
    
    with col2:
        st.markdown("""
        ## 🤖 모델 정보
        
        | 항목 | 내용 |
        |------|------|
        | 알고리즘 | Logistic Regression |
        | 선택 이유 | Recall 기준 최고 성능 |
        | ROC-AUC | 0.872 |
        | Recall | 0.806 |
        
        ## 📈 핵심 인사이트
        
        | 발견 | 설명 |
        |------|------|
        | 복잡한 모델 ≠ 좋은 모델 | LR이 XGBoost보다 Recall 높음 |
        | ROE 낮을수록 5배 확률 ↑ | 성장 여력 있는 기업 |
        | 변동성 높을수록 5배 확률 ↑ | 고위험 고수익 |
        """)
    
    st.markdown("---")
    
    st.markdown("## 🔄 ML 파이프라인")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**02. 데이터 수집**\n\nS&P 500 502개 종목\n\nyfinance API")
    with col2:
        st.info("**03. Feature Engineering**\n\n42→26개 Feature\n\nSMOTE 클래스 균형")
    with col3:
        st.info("**04. 모델 학습**\n\n5개 모델 비교\n\nLogistic Regression 선택")

# =============================================================================
# 2. 데이터 수집 (인터랙티브)
# =============================================================================
elif menu == "📊 데이터 수집":
    st.title("📊 데이터 수집")
    
    tab1, tab2 = st.tabs(["📥 수집 현황", "🔄 동적 필터링"])
    
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📈 가격 데이터
            - **기간:** 20년치 OHLCV + Adj Close
            - **형식:** .parquet (빠르고 용량 작음)
            - **소스:** yfinance API
            """)
        
        with col2:
            st.markdown("""
            #### 📋 재무제표
            - **종류:** 손익계산서, 재무상태표, 현금흐름표
            - **기간:** 연간 + 분기
            - **소스:** yfinance API
            """)
        
        st.warning("⚠️ **WBA (Walgreens Boots Alliance)**: 상장폐지로 인해 데이터 수집 실패")
    
    with tab2:
        st.markdown("### 🔄 동적 필터링이란?")
        
        st.markdown("""
        종목마다 **상장 시점이 다름** → 각 연도에 실제 데이터가 있는 종목만 사용
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.error("""
            **고정 필터링 - 문제점**
            
            2010년: META 포함 (2012년 상장인데!)
            
            → **Data Leakage** (미래 정보 누출)
            """)
        
        with col2:
            st.success("""
            **동적 필터링 - 해결**
            
            2010년: META 제외 (아직 상장 전)
            
            → 각 연도에 실제 존재한 종목만!
            """)
        
        st.markdown("---")
        
        # 인터랙티브: 연도 선택
        year_data = {
            2010: {'count': 435, 'new': ['TSLA', 'GM']},
            2011: {'count': 442, 'new': []},
            2012: {'count': 451, 'new': ['META']},
            2013: {'count': 460, 'new': ['ABBV', 'ZTS']},
            2014: {'count': 466, 'new': []},
            2015: {'count': 471, 'new': ['PYPL']},
            2016: {'count': 477, 'new': []},
            2017: {'count': 479, 'new': []},
            2018: {'count': 482, 'new': []},
            2019: {'count': 489, 'new': ['UBER', 'CRWD', 'DDOG']}
        }
        
        selected_year = st.slider("📅 연도 선택", 2010, 2019, 2010)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(
                f"{selected_year}년 사용 가능 종목",
                f"{year_data[selected_year]['count']}개"
            )
            
            if year_data[selected_year]['new']:
                st.markdown(f"**{selected_year}년 신규 상장:**")
                for ticker in year_data[selected_year]['new']:
                    st.markdown(f"- {ticker}")
        
        with col2:
            chart_data = pd.DataFrame({
                '연도': list(year_data.keys()),
                '종목 수': [v['count'] for v in year_data.values()]
            })
            chart_data['선택'] = chart_data['연도'].apply(
                lambda x: '선택' if x == selected_year else '기타'
            )
            
            fig = px.bar(chart_data, x='연도', y='종목 수', 
                        color='선택',
                        color_discrete_map={'선택': '#e74c3c', '기타': '#3498db'},
                        text='종목 수')
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 3. Feature Engineering (인터랙티브)
# =============================================================================
elif menu == "🔧 Feature Engineering":
    st.title("🔧 Feature Engineering")
    
    try:
        dataset = load_data()
        features = load_features()
        data_loaded = True
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        data_loaded = False
    
    if data_loaded:
        tab1, tab2, tab3 = st.tabs(["🎯 Target 분포", "📋 Feature 탐색", "🔥 Feature Selection"])
        
        with tab1:
            st.markdown("### Target: 5년 후 5배(500%) 이상 상승 여부")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("전체 샘플", f"{len(dataset):,}개")
            with col2:
                st.metric("5배 달성", f"{dataset['target_5x'].sum():,}개", f"{dataset['target_5x'].mean()*100:.1f}%")
            with col3:
                st.metric("미달성", f"{(dataset['target_5x']==0).sum():,}개")
            
            st.markdown("---")
            
            selected_year = st.selectbox(
                "📅 연도별 Target 분포 보기",
                ['전체'] + list(range(2010, 2020))
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if selected_year == '전체':
                    data = dataset
                else:
                    data = dataset[dataset['start_year'] == selected_year]
                
                target_counts = data['target_5x'].value_counts()
                fig = px.pie(
                    values=target_counts.values,
                    names=['미달성', '5배 달성'],
                    title=f'{selected_year} Target 분포',
                    color_discrete_sequence=['#3498db', '#e74c3c']
                )
                fig.update_traces(textinfo='percent+value')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if selected_year != '전체':
                    achieved = data[data['target_5x'] == 1]['ticker'].tolist()
                    if achieved:
                        st.markdown(f"**{selected_year}년 시작 → 5배 달성 종목:**")
                        for t in achieved[:10]:
                            st.markdown(f"- {t}")
                        if len(achieved) > 10:
                            st.markdown(f"...외 {len(achieved)-10}개")
                    else:
                        st.info("5배 달성 종목 없음")
                else:
                    st.markdown("""
                    **클래스 불균형 문제**
                    
                    - 5배 달성: **4.6%** (213개)
                    - 미달성: **95.4%** (4,439개)
                    
                    **해결: SMOTE**
                    - 소수 클래스 합성하여 균형
                    - Train 데이터: 50% vs 50%
                    """)
        
        with tab2:
            st.markdown("### Feature 탐색")
            
            categories = {}
            for feat, (cat, desc) in FEATURE_DESC.items():
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append((feat, desc))
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_cat = st.selectbox("카테고리 선택", list(categories.keys()))
                
                feature_options = [f"{feat}" for feat, desc in categories[selected_cat]]
                selected_feat = st.selectbox("Feature 선택", feature_options)
            
            with col2:
                if selected_feat in dataset.columns:
                    cat, desc = FEATURE_DESC[selected_feat]
                    
                    st.markdown(f"**{selected_feat}**")
                    st.markdown(f"- 카테고리: {cat}")
                    st.markdown(f"- 설명: {desc}")
                    
                    fig = px.histogram(
                        dataset, x=selected_feat, 
                        color='target_5x',
                        barmode='overlay',
                        labels={'target_5x': 'Target'},
                        color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                        title=f'{selected_feat} 분포'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Feature Selection: 42개 → 26개")
            
            removal_reason = st.radio(
                "제거 사유 선택",
                ["결측치 50% 이상", "상관관계 0.8 이상"],
                horizontal=True
            )
            
            if removal_reason == "결측치 50% 이상":
                removed = pd.DataFrame({
                    'Feature': ['revenue_cagr_5y', 'rnd_growth_rate', 'rnd_to_revenue'],
                    '결측치': ['100%', '69.6%', '68.8%'],
                    '이유': ['Yahoo Finance가 4~5년치만 제공', 'R&D 비용 미공시 기업 많음', '위와 동일']
                })
                st.dataframe(removed, use_container_width=True, hide_index=True)
            else:
                removed = pd.DataFrame({
                    'Feature 1': ['ps_ratio', 'operating_margin', 'price_momentum_6m'],
                    'Feature 2': ['ev_to_revenue', 'net_margin', 'price_to_sma_200'],
                    '상관계수': [0.974, 0.893, 0.917],
                    '제거 대상': ['ev_to_revenue', 'net_margin', 'price_momentum_6m']
                })
                st.dataframe(removed, use_container_width=True, hide_index=True)

# =============================================================================
# 4. 모델 학습 (인터랙티브)
# =============================================================================
elif menu == "🤖 모델 학습":
    st.title("🤖 모델 학습")
    
    try:
        model = load_model()
        dataset = load_data()
        features = load_features()
        data_loaded = True
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        data_loaded = False
    
    if data_loaded:
        tab1, tab2, tab3 = st.tabs(["📊 모델 비교", "📈 Confusion Matrix", "🎯 Feature Importance"])
        
        with tab1:
            st.markdown("### 5개 모델 점진적 비교")
            
            model_data = {
                'Logistic Regression': {'ROC-AUC': 0.872, 'Recall': 0.806, 'Precision': 0.111, 'desc': '가장 간단한 모델, 베이스라인'},
                'Decision Tree': {'ROC-AUC': 0.873, 'Recall': 0.774, 'Precision': 0.171, 'desc': '비선형 관계 학습 가능'},
                'Random Forest': {'ROC-AUC': 0.882, 'Recall': 0.677, 'Precision': 0.169, 'desc': '트리 여러 개 병렬 학습'},
                'Gradient Boosting': {'ROC-AUC': 0.887, 'Recall': 0.645, 'Precision': 0.180, 'desc': '이전 모델 오차 순차 학습'},
                'XGBoost': {'ROC-AUC': 0.878, 'Recall': 0.645, 'Precision': 0.196, 'desc': 'Gradient Boosting 개선 버전'}
            }
            
            selected_model = st.selectbox("🤖 모델 선택", list(model_data.keys()))
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                data = model_data[selected_model]
                st.metric("ROC-AUC", f"{data['ROC-AUC']:.3f}")
                st.metric("Recall", f"{data['Recall']:.3f}")
                st.metric("Precision", f"{data['Precision']:.3f}")
                st.markdown(f"**설명:** {data['desc']}")
                
                if selected_model == 'Logistic Regression':
                    st.success("✅ **최종 선택** (Recall 기준)")
            
            with col2:
                compare_df = pd.DataFrame([
                    {'Model': k, 'ROC-AUC': v['ROC-AUC'], 'Recall': v['Recall']}
                    for k, v in model_data.items()
                ])
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='ROC-AUC', x=compare_df['Model'], y=compare_df['ROC-AUC'],
                    marker_color=['#e74c3c' if m == selected_model else '#3498db' for m in compare_df['Model']]
                ))
                fig.add_trace(go.Bar(
                    name='Recall', x=compare_df['Model'], y=compare_df['Recall'],
                    marker_color=['#e74c3c' if m == selected_model else '#2ecc71' for m in compare_df['Model']]
                ))
                fig.update_layout(barmode='group', height=350)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            st.info("**핵심 발견:** ROC-AUC는 복잡한 모델일수록 상승, Recall은 하락 → 복잡한 모델 ≠ 더 좋은 모델!")
        
        with tab2:
            st.markdown("### Confusion Matrix (Test: 971개)")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                cm_data = [[739, 201], [6, 25]]
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm_data,
                    x=['예측: 미달성', '예측: 5배'],
                    y=['실제: 미달성', '실제: 5배'],
                    text=[['TN: 739', 'FP: 201'], ['FN: 6', 'TP: 25']],
                    texttemplate='%{text}',
                    textfont={'size': 16},
                    colorscale='Blues'
                ))
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cm_item = st.radio(
                    "항목 선택",
                    ["TP (25)", "FN (6)", "FP (201)", "TN (739)"]
                )
                
                explanations = {
                    "TP (25)": "**정확히 찾음!** 5배 달성 종목 31개 중 25개 발굴 성공",
                    "FN (6)": "**놓침** 5배 달성 종목 6개를 미달성으로 잘못 예측",
                    "FP (201)": "**헛발질** 5배 아닌데 5배라고 예측",
                    "TN (739)": "**걸러냄** 5배 안 갈 종목 정확히 걸러냄"
                }
                
                st.info(explanations[cm_item])
                
                st.markdown("""
                **Precision** = 25 / (25+201) = **11%**
                
                **Recall** = 25 / (25+6) = **81%**
                """)
        
        with tab3:
            st.markdown("### Feature Importance")
            
            coef_data = pd.DataFrame({
                'feature': ['roe', 'volatility_1y', 'pb_ratio', 'roa', 'reinvestment_rate', 
                           'operating_margin_trend', 'fcf_yield', 'earnings_quality', 'pe_ratio', 'operating_margin'],
                'coefficient': [-2.037, 0.947, 0.870, 0.746, 0.507, -0.505, 0.427, -0.417, 0.371, 0.367]
            })
            
            selected_feat = st.selectbox("Feature 선택", coef_data['feature'].tolist())
            
            selected_coef = coef_data[coef_data['feature'] == selected_feat]['coefficient'].values[0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(selected_feat, f"{selected_coef:+.3f}", delta="5배 확률 ↑" if selected_coef > 0 else "5배 확률 ↓")
                
                interpretations = {
                    'roe': 'ROE 낮을수록 5배 확률 ↑ (성장 여력)',
                    'volatility_1y': '변동성 높을수록 5배 확률 ↑ (고위험 고수익)',
                    'pb_ratio': 'PBR 높을수록 5배 확률 ↑ (시장 기대)',
                    'roa': 'ROA 높을수록 5배 확률 ↑',
                    'reinvestment_rate': '재투자율 높을수록 5배 확률 ↑',
                    'operating_margin_trend': '영업이익률 추세 하락 시 5배 확률 ↑',
                    'fcf_yield': 'FCF 수익률 높을수록 5배 확률 ↑',
                    'earnings_quality': '이익의 질 낮을수록 5배 확률 ↑',
                    'pe_ratio': 'PER 높을수록 5배 확률 ↑',
                    'operating_margin': '영업이익률 높을수록 5배 확률 ↑'
                }
                
                st.markdown(f"**해석:** {interpretations.get(selected_feat, '')}")
            
            with col2:
                coef_sorted = coef_data.sort_values('coefficient', ascending=True)
                colors = ['#e74c3c' if f == selected_feat else ('#27ae60' if c > 0 else '#3498db') 
                         for f, c in zip(coef_sorted['feature'], coef_sorted['coefficient'])]
                
                fig = go.Figure(go.Bar(
                    x=coef_sorted['coefficient'],
                    y=coef_sorted['feature'],
                    orientation='h',
                    marker_color=colors,
                    text=[f'{v:+.3f}' for v in coef_sorted['coefficient']],
                    textposition='outside'
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 5. 종목 분석 (인터랙티브)
# =============================================================================
elif menu == "🔍 종목 분석":
    st.title("🔍 종목 분석")
    st.markdown("### 종목을 선택하면 5배 성장 가능성을 예측합니다")
    
    try:
        dataset = load_data()
        model = load_model()
        features = load_features()
        data_loaded = True
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        data_loaded = False
    
    if data_loaded:
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tickers = sorted(dataset['ticker'].unique())
            popular = ['TSLA', 'AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'NFLX']
            popular_available = [t for t in popular if t in tickers]
            other_tickers = [t for t in tickers if t not in popular]
            sorted_tickers = popular_available + other_tickers
            
            selected_ticker = st.selectbox("📌 종목 선택", sorted_tickers, index=0)
        
        with col2:
            available_years = sorted(dataset[dataset['ticker'] == selected_ticker]['start_year'].unique())
            selected_year = st.selectbox("📅 시작 연도 선택", available_years, index=0)
        
        row = dataset[(dataset['ticker'] == selected_ticker) & (dataset['start_year'] == selected_year)]
        
        if len(row) > 0:
            row = row.iloc[0]
            X = row[features].values.reshape(1, -1)
            
            prob = model.predict_proba(X)[0][1]
            prediction = model.predict(X)[0]
            actual = int(row['target_5x'])
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("5배 달성 확률", f"{prob*100:.1f}%", delta="높음" if prob > 0.5 else "낮음")
            with col2:
                st.metric("모델 예측", "5배 달성" if prediction == 1 else "미달성")
            with col3:
                st.metric(f"실제 결과 ({selected_year}→{selected_year+5})", "5배 달성" if actual == 1 else "미달성")
            
            st.markdown("---")
            
            if prediction == actual:
                if actual == 1:
                    st.success(f"**정확한 예측!** {selected_ticker}는 5년간 5배 이상 성장, 모델도 예측 성공")
                else:
                    st.success(f"**정확한 예측!** {selected_ticker}는 5배 미달성, 모델도 정확히 예측")
            else:
                if actual == 1:
                    st.warning(f"**False Negative** {selected_ticker}는 실제 5배 달성, 모델은 미달성 예측")
                else:
                    st.warning(f"**False Positive** {selected_ticker}는 실제 미달성, 모델은 5배 달성 예측")
            
            st.markdown("---")
            st.markdown("### 🎯 예측 근거 (Feature 기여도)")
            
            contributions = X[0] * model.coef_[0]
            
            contrib_df = pd.DataFrame({'Feature': features, '기여도': contributions})
            contrib_df['abs_contrib'] = contrib_df['기여도'].abs()
            contrib_df = contrib_df.sort_values('abs_contrib', ascending=False).head(10)
            contrib_df = contrib_df.sort_values('기여도', ascending=True)
            
            colors = ['#27ae60' if x > 0 else '#e74c3c' for x in contrib_df['기여도']]
            
            fig = go.Figure(go.Bar(
                x=contrib_df['기여도'], y=contrib_df['Feature'],
                orientation='h', marker_color=colors,
                text=[f'{v:+.3f}' for v in contrib_df['기여도']],
                textposition='outside'
            ))
            fig.update_layout(title=f'{selected_ticker} ({selected_year}) Feature 기여도', height=400)
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 푸터
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("© 2024 5X Finder")