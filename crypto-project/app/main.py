import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import (
    load_data, load_models, get_feature_importance,
    get_model_metrics, get_confusion_matrix_data, create_export_data,
    get_shap_global_values, get_shap_single_values,
    predict_with_threshold, get_pr_curve, get_precision_at_k, get_metrics_by_time_step,
    load_transaction_graph, build_neighbor_graph_html
)
import streamlit.components.v1 as components

# Page Config
st.set_page_config(
    page_title="Crypto Fraud Detector",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - Dark Fintech/Crypto Theme + Material Icons
# ============================================
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Outlined" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Material Icon Helper */
    .material-icons, .material-icons-outlined, .material-icons-round {
        font-size: inherit;
        vertical-align: middle;
        margin-right: 8px;
    }
    
    .icon-sm { font-size: 18px; }
    .icon-md { font-size: 24px; }
    .icon-lg { font-size: 32px; }
    .icon-xl { font-size: 48px; }
    
    .icon-cyan { color: #00D4FF; }
    .icon-green { color: #00FF88; }
    .icon-pink { color: #FF2D7C; }
    .icon-purple { color: #7B2DFF; }
    .icon-white { color: #FAFAFA; }
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Title Styling */
    .main-title {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        margin-bottom: 0.5rem;
    }
    
    .main-title-text {
        background: linear-gradient(90deg, #00D4FF 0%, #7B2DFF 50%, #FF2D7C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-title .material-icons-round {
        font-size: 3rem;
        background: linear-gradient(90deg, #00D4FF 0%, #7B2DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphism Card */
    .glass-card {
        background: rgba(30, 30, 46, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
    }
    
    /* Metric Cards */
    .metric-container {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        flex-wrap: wrap;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 30, 46, 0.9) 0%, rgba(20, 20, 30, 0.9) 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        min-width: 200px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
    }
    
    .metric-card.total {
        border-color: rgba(0, 212, 255, 0.5);
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.2);
    }
    
    .metric-card.safe {
        border-color: rgba(0, 255, 136, 0.5);
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.2);
    }
    
    .metric-card.fraud {
        border-color: rgba(255, 45, 124, 0.5);
        box-shadow: 0 4px 20px rgba(255, 45, 124, 0.2);
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-value.total { color: #00D4FF; }
    .metric-value.safe { color: #00FF88; }
    .metric-value.fraud { color: #FF2D7C; }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3);
    }
    
    .section-header h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
        color: #FAFAFA;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .section-header .material-icons-round {
        color: #00D4FF;
        font-size: 1.75rem;
    }
    
    /* Alert Boxes */
    .alert-safe {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.15) 0%, rgba(0, 200, 100, 0.1) 100%);
        border: 1px solid rgba(0, 255, 136, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .alert-fraud {
        background: linear-gradient(135deg, rgba(255, 45, 124, 0.15) 0%, rgba(200, 30, 100, 0.1) 100%);
        border: 1px solid rgba(255, 45, 124, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(255, 45, 124, 0.4); }
        50% { box-shadow: 0 0 20px 5px rgba(255, 45, 124, 0.2); }
    }
    
    .alert-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .alert-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .alert-safe .alert-icon { color: #00FF88; }
    .alert-safe .alert-title { color: #00FF88; }
    .alert-fraud .alert-icon { color: #FF2D7C; }
    .alert-fraud .alert-title { color: #FF2D7C; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1a1a2e 100%);
    }
    
    .sidebar-header {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }
    
    .sidebar-logo {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #00D4FF 0%, #7B2DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        background: linear-gradient(90deg, #00D4FF, #7B2DFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Table Styling */
    .comparison-table {
        background: rgba(30, 30, 46, 0.5);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #00D4FF 0%, #7B2DFF 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #00FF88 0%, #00D4FF 100%);
        color: #0E1117;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: #666;
        font-size: 0.85rem;
    }
    
    .footer .material-icons-round {
        font-size: 1rem;
        vertical-align: middle;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00D4FF 0%, #FF2D7C 100%);
    }
    
    /* Sidebar section icons */
    .sidebar-section {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 1rem;
        color: #FAFAFA;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-section .material-icons-round {
        font-size: 1.25rem;
        color: #00D4FF;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">
            <span class="material-icons-round">security</span>
        </div>
        <div class="sidebar-title">Crypto Fraud Detector</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-section">
        <span class="material-icons-round">settings</span>
        <span>Configuration</span>
    </div>
    """, unsafe_allow_html=True)
    
# Load Data & Models
with st.spinner('Loading Dataset & Models...'):
    df = load_data()
    models = load_models()

if df is not None and models:
    
    # Model Selection in Sidebar
    with st.sidebar:
        selected_model_name = st.selectbox(
            "Select Model",
            list(models.keys()),
            help="Choose the ML model for fraud detection"
        )
        model = models[selected_model_name]
        
        threshold = st.slider(
            "Alert Threshold",
            min_value=0.05, max_value=0.95, value=0.50, step=0.05,
            help="Flag a transaction as fraud when P(fraud) ≥ threshold. "
                 "Lower = catch more fraud (higher recall) but more false alerts for the review team."
        )
        
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-section">
            <span class="material-icons-round">analytics</span>
            <span>Quick Stats</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"**Dataset Size:** {len(df):,} transactions")
        st.markdown(f"**Models Loaded:** {len(models)}")
        st.markdown(f"**Features:** {len(df.columns) - 3}")
    
    # ============================================
    # MAIN TITLE
    # ============================================
    st.markdown("""
    <div class="main-title">
        <span class="material-icons-round">security</span>
        <span class="main-title-text">Bitcoin Transaction Fraud Detection</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Detection using Machine Learning on Elliptic Dataset</p>', unsafe_allow_html=True)
    
    # ============================================
    # 1. DATASET OVERVIEW
    # ============================================
    st.markdown("""
    <div class="section-header">
        <h2><span class="material-icons-round">bar_chart</span>Dataset Overview</h2>
    </div>
    """, unsafe_allow_html=True)
    
    total_tx = len(df)
    illicit_tx = len(df[df['label'] == 1])
    licit_tx = len(df[df['label'] == 0])
    
    # Custom Metric Cards with Material Icons
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card total">
            <div class="metric-icon icon-cyan"><span class="material-icons-round">receipt_long</span></div>
            <div class="metric-label">Total Transactions</div>
            <div class="metric-value total">{total_tx:,}</div>
        </div>
        <div class="metric-card safe">
            <div class="metric-icon icon-green"><span class="material-icons-round">verified_user</span></div>
            <div class="metric-label">Licit (Safe)</div>
            <div class="metric-value safe">{licit_tx:,}</div>
        </div>
        <div class="metric-card fraud">
            <div class="metric-icon icon-pink"><span class="material-icons-round">gpp_bad</span></div>
            <div class="metric-label">Illicit (Fraud)</div>
            <div class="metric-value fraud">{illicit_tx:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Distribution Chart
    col_chart1, col_chart2 = st.columns([1, 1])
    
    with col_chart1:
        fig_pie = px.pie(
            names=['Licit (Safe)', 'Illicit (Fraud)'],
            values=[licit_tx, illicit_tx],
            color_discrete_sequence=['#00FF88', '#FF2D7C'],
            hole=0.6
        )
        fig_pie.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FAFAFA'),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            title=dict(text="Transaction Distribution", font=dict(size=16))
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#0E1117', width=2))
        )
        st.plotly_chart(fig_pie, width='stretch')
    
    with col_chart2:
        # Time-based distribution
        time_dist = df.groupby(['time_step', 'label']).size().unstack(fill_value=0)
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=time_dist.index, y=time_dist[0] if 0 in time_dist.columns else [],
            mode='lines+markers', name='Licit',
            line=dict(color='#00FF88', width=2),
            fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.1)'
        ))
        fig_time.add_trace(go.Scatter(
            x=time_dist.index, y=time_dist[1] if 1 in time_dist.columns else [],
            mode='lines+markers', name='Illicit',
            line=dict(color='#FF2D7C', width=2),
            fill='tozeroy', fillcolor='rgba(255, 45, 124, 0.1)'
        ))
        fig_time.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(text="Transactions Over Time", font=dict(size=16)),
            xaxis_title="Time Step",
            yaxis_title="Count",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_time, width='stretch')
    
    # ============================================
    # 2. MODEL COMPARISON TABLE
    # ============================================
    st.markdown("""
    <div class="section-header">
        <h2><span class="material-icons-round">leaderboard</span>Model Performance Comparison</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare test data
    test_df = df[df['time_step'] > 34]
    X_test = test_df.drop(columns=['label', 'txId', 'time_step'], errors='ignore')
    if 'class' in X_test.columns:
        X_test = X_test.drop(columns=['class'])
    y_test = test_df['label']
    
    # Get metrics for all models
    metrics_df = get_model_metrics(models, X_test, y_test, threshold)
    
    # Display as styled table
    col_table, col_bar = st.columns([1, 1])
    
    with col_table:
        st.markdown(f"#### Performance Metrics (threshold = {threshold:.2f})")
        st.caption("Test set = time steps 35–49 (unseen future data). PR-AUC does not depend on the threshold.")
        styled_df = metrics_df.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'PR-AUC']:
            styled_df[col] = styled_df[col].apply(lambda x: f"{x:.2%}")
        st.dataframe(styled_df, width='stretch', hide_index=True)
    
    with col_bar:
        st.markdown("#### F1 Score Comparison")
        fig_bar = px.bar(
            metrics_df,
            x='Model',
            y='F1 Score',
            color='F1 Score',
            color_continuous_scale=['#FF2D7C', '#7B2DFF', '#00D4FF', '#00FF88']
        )
        fig_bar.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(tickformat='.0%')
        )
        st.plotly_chart(fig_bar, width='stretch')
    
    # ============================================
    # 3. CONFUSION MATRIX
    # ============================================
    st.markdown("""
    <div class="section-header">
        <h2><span class="material-icons-round">grid_on</span>Confusion Matrix</h2>
    </div>
    """, unsafe_allow_html=True)
    
    cm = get_confusion_matrix_data(model, X_test, y_test, threshold)
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: Safe', 'Predicted: Fraud'],
        y=['Actual: Safe', 'Actual: Fraud'],
        colorscale=[[0, '#1E1E2E'], [0.5, '#7B2DFF'], [1, '#00D4FF']],
        text=cm,
        texttemplate="%{text:,}",
        textfont={"size": 16, "color": "white"},
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>"
    ))
    fig_cm.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text=f"Confusion Matrix - {selected_model_name}", font=dict(size=16)),
        xaxis=dict(side='bottom'),
        height=400
    )
    st.plotly_chart(fig_cm, width='stretch')
    
    # ============================================
    # 3.1 ALERT THRESHOLD & REVIEW BUDGET
    # ============================================
    st.markdown("""
    <div class="section-header">
        <h2><span class="material-icons-round">tune</span>Alert Threshold & Review Budget</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='color: #888; margin-bottom: 1rem;'>ในงาน AML การพลาดธุรกรรมผิดกฎหมาย (false negative) มักแพงกว่าการแจ้งเตือนผิด แต่ทีม Compliance ตรวจได้จำกัด — ส่วนนี้ช่วยเลือกจุดสมดุล</p>", unsafe_allow_html=True)
    
    col_pr, col_budget = st.columns([3, 2])
    
    with col_pr:
        pr_df = get_pr_curve(model, X_test, y_test)
        y_pred_thr = predict_with_threshold(model, X_test, threshold)
        cur_p = float(((y_pred_thr == 1) & (y_test.values == 1)).sum() / max(y_pred_thr.sum(), 1))
        cur_r = float(((y_pred_thr == 1) & (y_test.values == 1)).sum() / max(y_test.sum(), 1))
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=pr_df['Recall'], y=pr_df['Precision'], mode='lines', name='PR curve',
            line=dict(color='#00D4FF', width=2),
            customdata=pr_df['Threshold'],
            hovertemplate='Recall %{x:.2f}<br>Precision %{y:.2f}<br>Threshold %{customdata:.2f}<extra></extra>'
        ))
        fig_pr.add_trace(go.Scatter(
            x=[cur_r], y=[cur_p], mode='markers', name=f'Current threshold ({threshold:.2f})',
            marker=dict(color='#FF2D7C', size=14, line=dict(color='white', width=2))
        ))
        fig_pr.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            title=dict(text=f"Precision–Recall Curve — {selected_model_name}", font=dict(size=14)),
            xaxis=dict(title='Recall (share of fraud caught)', range=[0, 1.02], tickformat='.0%'),
            yaxis=dict(title='Precision (share of alerts that are fraud)', range=[0, 1.02], tickformat='.0%'),
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pr, width='stretch')
    
    with col_budget:
        st.markdown("#### If the team can review only K alerts…")
        pk_df = get_precision_at_k(model, X_test, y_test)
        pk_show = pk_df.copy()
        pk_show['Precision@K'] = pk_show['Precision@K'].apply(lambda x: f"{x:.1%}")
        pk_show['Share of all fraud caught'] = pk_show['Share of all fraud caught'].apply(lambda x: f"{x:.1%}")
        st.dataframe(pk_show, width='stretch', hide_index=True)
        st.caption(f"Ranks the {len(X_test):,} test transactions by fraud probability and checks the top K. "
                   f"Test period contains {int(y_test.sum()):,} illicit transactions.")
    
    # ============================================
    # 3.2 PERFORMANCE OVER TIME (DRIFT)
    # ============================================
    st.markdown("""
    <div class="section-header">
        <h2><span class="material-icons-round">timeline</span>Performance Over Time</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='color: #888; margin-bottom: 1rem;'>โมเดลยังแม่นอยู่ไหมเมื่อเวลาผ่านไป? ถ้า recall ตกในช่วงหลัง แปลว่าพฤติกรรมการฟอกเงินเปลี่ยนไป (concept drift) และควร retrain</p>", unsafe_allow_html=True)
    
    ts_df = get_metrics_by_time_step(model, test_df, X_test.columns.tolist(), threshold)
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ts.add_trace(go.Bar(
        x=ts_df['time_step'], y=ts_df['Fraud count'], name='Illicit tx count',
        marker_color='rgba(255, 45, 124, 0.25)'
    ), secondary_y=True)
    for metric, color in [('Recall', '#FF2D7C'), ('Precision', '#00D4FF'), ('F1 Score', '#00FF88')]:
        fig_ts.add_trace(go.Scatter(
            x=ts_df['time_step'], y=ts_df[metric], mode='lines+markers', name=metric,
            line=dict(color=color, width=2)
        ), secondary_y=False)
    fig_ts.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text=f"Test-period metrics by time step — {selected_model_name} (threshold {threshold:.2f})", font=dict(size=14)),
        xaxis_title='Time Step', height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5)
    )
    fig_ts.update_yaxes(title_text='Score', range=[0, 1.05], tickformat='.0%', secondary_y=False)
    fig_ts.update_yaxes(title_text='Illicit tx count', showgrid=False, secondary_y=True)
    st.plotly_chart(fig_ts, width='stretch')
    
    # ============================================
    # 3.5 SHAP GLOBAL FEATURE IMPORTANCE
    # ============================================
    st.markdown("""
    <div class="section-header">
        <h2><span class="material-icons-round">auto_awesome</span>SHAP Feature Importance</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='color: #888; margin-bottom: 1rem;'>SHAP (SHapley Additive exPlanations) แสดงว่า features ไหนมีอิทธิพลต่อการทำนายของโมเดลมากที่สุด</p>", unsafe_allow_html=True)
    
    with st.spinner('Calculating SHAP values...'):
        shap_importance, shap_explainer = get_shap_global_values(
            model, X_test, selected_model_name
        )
    
    if shap_importance is not None:
        col_shap_bar, col_shap_compare = st.columns([1, 1])
        
        with col_shap_bar:
            top_n_shap = 15
            shap_top = shap_importance.head(top_n_shap)
            
            fig_shap_bar = go.Figure(go.Bar(
                x=shap_top['SHAP_Importance'].values[::-1],
                y=shap_top['Feature'].values[::-1],
                orientation='h',
                marker=dict(
                    color=shap_top['SHAP_Importance'].values[::-1],
                    colorscale=[[0, '#1E1E2E'], [0.3, '#7B2DFF'], [0.7, '#00D4FF'], [1, '#00FF88']],
                    line=dict(width=0)
                )
            ))
            fig_shap_bar.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title=dict(text=f"Top {top_n_shap} SHAP Features — {selected_model_name}", font=dict(size=14)),
                xaxis_title="Mean |SHAP Value|",
                height=500,
                margin=dict(l=10, r=10, t=40, b=40)
            )
            st.plotly_chart(fig_shap_bar, width='stretch')
        
        with col_shap_compare:
            # Compare SHAP vs Traditional Feature Importance
            traditional_imp = get_feature_importance(model, X_test.columns)
            if traditional_imp is not None:
                top_trad = traditional_imp.head(top_n_shap)
                top_shap_names = set(shap_top['Feature'].values)
                top_trad_names = set(top_trad['Feature'].values)
                overlap = top_shap_names & top_trad_names
                
                # Merge for comparison
                merged = pd.merge(
                    shap_top.rename(columns={'SHAP_Importance': 'SHAP'}),
                    top_trad.rename(columns={'Importance': 'Traditional'}),
                    on='Feature', how='outer'
                ).fillna(0).head(10)
                
                # Normalize for comparison
                if merged['SHAP'].max() > 0:
                    merged['SHAP_norm'] = merged['SHAP'] / merged['SHAP'].max()
                else:
                    merged['SHAP_norm'] = 0
                if merged['Traditional'].max() > 0:
                    merged['Trad_norm'] = merged['Traditional'] / merged['Traditional'].max()
                else:
                    merged['Trad_norm'] = 0
                
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(
                    name='SHAP',
                    x=merged['Feature'],
                    y=merged['SHAP_norm'],
                    marker_color='#00D4FF'
                ))
                fig_compare.add_trace(go.Bar(
                    name='Traditional',
                    x=merged['Feature'],
                    y=merged['Trad_norm'],
                    marker_color='#7B2DFF'
                ))
                fig_compare.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    title=dict(text="SHAP vs Traditional Feature Importance", font=dict(size=14)),
                    barmode='group',
                    yaxis_title='Normalized Importance',
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    xaxis=dict(tickangle=-45),
                    margin=dict(l=10, r=10, t=40, b=80)
                )
                st.plotly_chart(fig_compare, width='stretch')
    else:
        st.info("SHAP analysis is not available for this model.")
    
    # ============================================
    # 4. FRAUD PREDICTION INTERFACE
    # ============================================
    st.markdown("""
    <div class="section-header">
        <h2><span class="material-icons-round">search</span>Fraud Prediction</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Test Sample Selection
    test_txs = df[df['time_step'] > 34]
    
    col_method, col_input = st.columns([1, 2])
    
    with col_method:
        sample_method = st.radio(
            "Input Method",
            ["Random Sample", "Enter Transaction ID"],
            label_visibility="visible"
        )
    
    selected_tx = None
    
    with col_input:
        if sample_method == "Random Sample":
            if st.button("Get Random Transaction", width='stretch'):
                selected_tx = test_txs.sample(1)
                st.session_state['selected_tx'] = selected_tx
            elif 'selected_tx' in st.session_state:
                selected_tx = st.session_state['selected_tx']
        else:
            tx_id_input = st.number_input("Enter Transaction ID (txId)", min_value=0, key="tx_id_input")
            if tx_id_input in test_txs['txId'].values:
                selected_tx = test_txs[test_txs['txId'] == tx_id_input]
            elif tx_id_input != 0:
                st.error("Transaction ID not found in Test Set.")
    
    if selected_tx is not None:
        with st.expander("Transaction Details", expanded=True):
            st.dataframe(selected_tx.head(), width='stretch')
        
        # Prepare features
        features = selected_tx.drop(columns=['label', 'txId', 'time_step'], errors='ignore')
        if 'class' in features.columns:
            features = features.drop(columns=['class'])
        
        # Keep the result on screen across reruns (e.g. moving the threshold slider)
        current_tx_id = int(selected_tx['txId'].values[0])
        if st.button("Analyze Transaction", width='stretch', type="primary"):
            st.session_state['analyzed_tx'] = current_tx_id
        
        if st.session_state.get('analyzed_tx') == current_tx_id:
            probability = model.predict_proba(features)[0][1]
            prediction = int(probability >= threshold)
            
            st.markdown("---")
            
            # Result Display
            col_result, col_prob = st.columns([2, 1])
            
            with col_result:
                if prediction == 1:
                    st.markdown("""
                    <div class="alert-fraud">
                        <div class="alert-icon"><span class="material-icons-round">warning</span></div>
                        <div class="alert-title">FRAUD DETECTED</div>
                        <div>This transaction shows suspicious patterns</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="alert-safe">
                        <div class="alert-icon"><span class="material-icons-round">check_circle</span></div>
                        <div class="alert-title">SAFE TRANSACTION</div>
                        <div>No fraudulent activity detected</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_prob:
                st.markdown("#### Fraud Probability")
                st.metric(label="Fraud probability", value=f"{probability:.1%}", label_visibility="collapsed")
                st.progress(float(probability))
            
            # Ground Truth Comparison
            actual_label = selected_tx['label'].values[0]
            
            col_actual, col_accuracy = st.columns(2)
            
            with col_actual:
                if actual_label == 1:
                    st.warning("**Actual Status:** FRAUD")
                else:
                    st.info("**Actual Status:** SAFE")
            
            with col_accuracy:
                if prediction == actual_label:
                    st.success("### Prediction is CORRECT")
                else:
                    st.error("### Prediction is WRONG")
            
            # ============================================
            # 5. MODEL EXPLANATION
            # ============================================
            st.markdown("""
            <div class="section-header">
                <h2><span class="material-icons-round">psychology</span>Model Explanation</h2>
            </div>
            """, unsafe_allow_html=True)
            
            feature_imp = get_feature_importance(model, features.columns)
            
            if feature_imp is not None:
                col_imp, col_radar = st.columns([1, 1])
                
                with col_imp:
                    top_n = 10
                    fig_imp = px.bar(
                        feature_imp.head(top_n),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        color='Importance',
                        color_continuous_scale=['#1E1E2E', '#7B2DFF', '#00D4FF']
                    )
                    fig_imp.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title=dict(text=f"Top {top_n} Important Features", font=dict(size=14)),
                        yaxis={'categoryorder': 'total ascending'},
                        coloraxis_showscale=False,
                        height=400
                    )
                    st.plotly_chart(fig_imp, width='stretch')
                
                with col_radar:
                    # Radar Chart
                    top_features = feature_imp['Feature'].head(5).tolist()
                    subset_df = df[top_features]
                    normalized_df = subset_df.rank(pct=True)
                    
                    fraud_indices = df[df['label'] == 1].index
                    avg_fraud = normalized_df.loc[fraud_indices].mean()
                    
                    tx_index = selected_tx.index[0]
                    current_tx_norm = normalized_df.loc[tx_index]
                    
                    fig_radar = go.Figure()
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=avg_fraud.values,
                        theta=top_features,
                        fill='toself',
                        name='Avg Fraud Profile',
                        line=dict(color='#FF2D7C', width=2),
                        fillcolor='rgba(255, 45, 124, 0.2)'
                    ))
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=current_tx_norm.values,
                        theta=top_features,
                        fill='toself',
                        name='This Transaction',
                        line=dict(color='#00D4FF', width=2),
                        fillcolor='rgba(0, 212, 255, 0.2)'
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.1)'),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title=dict(text="Feature Profile Comparison", font=dict(size=14)),
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_radar, width='stretch')
            
            # ============================================
            # 5.5 SHAP WATERFALL - Per Transaction
            # ============================================
            st.markdown("""
            <div class="section-header">
                <h2><span class="material-icons-round">account_tree</span>SHAP Explanation — This Transaction</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<p style='color: #888;'>แสดงว่า features ไหนดันผลการทำนายไปทาง <span style='color:#FF2D7C'>Fraud</span> หรือ <span style='color:#00FF88'>Safe</span></p>", unsafe_allow_html=True)
            
            with st.spinner('Calculating SHAP for this transaction...'):
                shap_vals, base_val = get_shap_single_values(
                    model, features, X_test, selected_model_name
                )
            
            if shap_vals is not None:
                # Create waterfall-style horizontal bar chart
                feat_names = features.columns.tolist()
                shap_df = pd.DataFrame({
                    'Feature': feat_names,
                    'SHAP': shap_vals
                })
                
                # Sort by absolute SHAP value, take top 15
                shap_df['abs_SHAP'] = shap_df['SHAP'].abs()
                shap_df = shap_df.sort_values('abs_SHAP', ascending=False).head(15)
                shap_df = shap_df.sort_values('SHAP', ascending=True)
                
                # Color: positive = fraud (red), negative = safe (green)
                colors = ['#FF2D7C' if v > 0 else '#00FF88' for v in shap_df['SHAP']]
                
                fig_waterfall = go.Figure(go.Bar(
                    x=shap_df['SHAP'].values,
                    y=shap_df['Feature'].values,
                    orientation='h',
                    marker=dict(color=colors, line=dict(width=0)),
                    hovertemplate='%{y}: %{x:.4f}<extra></extra>'
                ))
                
                fig_waterfall.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    title=dict(
                        text=f"SHAP Values — Top 15 Features ({selected_model_name})",
                        font=dict(size=14)
                    ),
                    xaxis_title='SHAP Value (→ Fraud | ← Safe)',
                    height=500,
                    margin=dict(l=10, r=10, t=40, b=40),
                    shapes=[dict(
                        type='line', x0=0, x1=0,
                        y0=-0.5, y1=len(shap_df) - 0.5,
                        line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash')
                    )]
                )
                st.plotly_chart(fig_waterfall, width='stretch')
                
                # Summary info
                col_base, col_output = st.columns(2)
                with col_base:
                    shap_unit = "probability" if selected_model_name == "Random Forest" else "log-odds"
                    st.markdown(f"""
                    <div class="glass-card" style="text-align:center;">
                        <div style="color:#888;font-size:0.85rem;">BASE VALUE</div>
                        <div style="font-size:1.5rem;font-weight:700;color:#00D4FF;">{base_val:.4f}</div>
                        <div style="color:#666;font-size:0.75rem;">Average model output ({shap_unit})</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_output:
                    output_val = base_val + shap_vals.sum()
                    output_color = '#FF2D7C' if prediction == 1 else '#00FF88'
                    st.markdown(f"""
                    <div class="glass-card" style="text-align:center;">
                        <div style="color:#888;font-size:0.85rem;">FINAL OUTPUT</div>
                        <div style="font-size:1.5rem;font-weight:700;color:{output_color};">{output_val:.4f}</div>
                        <div style="color:#666;font-size:0.75rem;">Base + SHAP contributions ({shap_unit})</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("SHAP analysis not available for this transaction.")
            
            # ============================================
            # 5.6 TRANSACTION NETWORK
            # ============================================
            st.markdown("""
            <div class="section-header">
                <h2><span class="material-icons-round">hub</span>Transaction Network</h2>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<p style='color: #888;'>ธุรกรรมที่เชื่อมกับรายการนี้โดยตรง (เงินเข้า/ออก) — <span style='color:#FF2D7C'>แดง = Illicit</span>, <span style='color:#00FF88'>เขียว = Licit</span>, เทา = ไม่ทราบ label</p>", unsafe_allow_html=True)
            
            G_tx = load_transaction_graph()
            if G_tx is None:
                st.info("Edge list not found at `data/raw/elliptic_txs_edgelist.csv` — download the Elliptic dataset to enable the network view.")
            else:
                tx_id = selected_tx['txId'].values[0]
                label_map = dict(zip(df['txId'], df['label']))
                graph_html, n_nodes, n_illicit = build_neighbor_graph_html(G_tx, tx_id, label_map, hops=1)
                if graph_html is None:
                    st.info("This transaction has no connections in the edge list.")
                else:
                    st.caption(f"{n_nodes} transactions shown · {n_illicit} known-illicit neighbours")
                    components.html(graph_html, height=440)
            
            # ============================================
            # 6. EXPORT BUTTON
            # ============================================
            st.markdown("""
            <div class="section-header">
                <h2><span class="material-icons-round">download</span>Export Results</h2>
            </div>
            """, unsafe_allow_html=True)
            
            export_df = create_export_data(selected_tx, prediction, probability, selected_model_name)
            
            csv = export_df.to_csv(index=False)
            
            col_export1, col_export2, col_export3 = st.columns([1, 2, 1])
            with col_export2:
                st.download_button(
                    label="Download Prediction Result as CSV",
                    data=csv,
                    file_name=f"fraud_prediction_{selected_tx['txId'].values[0]}.csv",
                    mime="text/csv",
                    width='stretch'
                )
    
    # ============================================
    # FOOTER
    # ============================================
    st.markdown("""
    <div class="footer">
        <p><span class="material-icons-round">security</span> Crypto Fraud Detector | Built with Streamlit & Machine Learning</p>
        <p>Powered by Elliptic Bitcoin Dataset | © 2026</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Please ensure the dataset and models are available.")
    st.info("Dataset: run `python scripts/download_data.py`, then `notebooks/1.0-data-prep.ipynb`. "
            "Models: run `python scripts/train_models.py`.")
