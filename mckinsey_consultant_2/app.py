"""
McKinsey-Style AI Data Consultant
A virtual consulting team using CrewAI for proactive insight generation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import sys
import os
import time
import base64
from datetime import datetime
from dotenv import load_dotenv
from utils.charts import create_trend_chart, create_distribution_plot, create_box_plot

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for consulting style
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .step-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.3rem 0;
        font-weight: 500;
    }
    .arrow-down {
        text-align: center;
        font-size: 1.5rem;
        color: #667eea;
        margin: 0.1rem 0;
    }
    .upload-area {
        background-color: #f8f9fa;
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    .insight-card {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 4px;
    }
    .stat-pill {
        display: inline-block;
        background-color: #e9ecef;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        color: #495057;
        margin-right: 0.4rem;
    }
    .recommendation {
        background-color: #e7f3ff;
        border-left: 4px solid #667eea;
        padding: 0.8rem;
        margin: 0.4rem 0;
        border-radius: 4px;
    }
    .caveat {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.8rem;
        margin: 0.4rem 0;
        border-radius: 4px;
        color: #856404;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .report-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #2c5b88 100%);
        color: white;
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin: 0.8rem 0 1rem 0;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
    }
    .report-kicker {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.75rem;
        opacity: 0.8;
        margin-bottom: 0.5rem;
    }
    .report-headline {
        font-size: 1.8rem;
        line-height: 1.2;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .report-subtext {
        font-size: 0.98rem;
        color: rgba(255,255,255,0.88);
    }
    .report-strip {
        background: #f8fafc;
        border: 1px solid #dbe4f0;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        height: 100%;
    }
    .report-strip-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #5b6b7f;
        margin-bottom: 0.35rem;
    }
    .report-strip-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #10233c;
        margin-bottom: 0.25rem;
    }
    .report-strip-note {
        font-size: 0.9rem;
        color: #5b6b7f;
    }
    .section-label {
        display: inline-block;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #47607b;
        background: #e8eff6;
        border-radius: 999px;
        padding: 0.22rem 0.65rem;
        margin-bottom: 0.55rem;
    }
    .insight-shell {
        background: white;
        border: 1px solid #dbe4f0;
        border-radius: 18px;
        padding: 1.1rem 1.1rem 0.9rem 1.1rem;
        margin: 0.9rem 0;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    }
    .insight-index {
        display: inline-block;
        min-width: 2rem;
        text-align: center;
        background: #12395b;
        color: white;
        border-radius: 999px;
        padding: 0.28rem 0.6rem;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    .insight-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #10233c;
        margin-bottom: 0.4rem;
    }
    .insight-summary {
        font-size: 1rem;
        color: #23384f;
        margin-bottom: 0.8rem;
    }
    .insight-detail {
        background: #f8fafc;
        border-left: 4px solid #2c5b88;
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        color: #41576f;
        margin-bottom: 0.8rem;
    }
    .priority-chip {
        display: inline-block;
        padding: 0.24rem 0.55rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .recommendation-card {
        background: linear-gradient(180deg, #f8fbff 0%, #eef5fb 100%);
        border: 1px solid #cfe0f1;
        border-radius: 14px;
        padding: 0.95rem 1rem;
        margin: 0.55rem 0;
        color: black;
    }
    .caveat-card {
        background: #fff8e8;
        border: 1px solid #f3d38a;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin: 0.55rem 0;
        color: #6d5313;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.8rem 2rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'report_data' not in st.session_state:
    st.session_state.report_data = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'openrouter_key' not in st.session_state:
    st.session_state.openrouter_key = ""
if 'openrouter_model' not in st.session_state:
    st.session_state.openrouter_model = ""
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'charts_mapping' not in st.session_state:
    st.session_state.charts_mapping = {}

# Title
st.markdown('<p class="main-header">📊 AI Data Analyst</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your data. Get consulting-grade insights automatically.</p>', unsafe_allow_html=True)

# ============================================
# STEP 1: UPLOAD DATA (MOVED TO TOP)
# ============================================
st.markdown("### 📁 Step 1: Upload Your Data")

uploaded_file = st.file_uploader(
    "Drop your CSV file here",
    type=['csv'],
    help="Upload any CSV file - sales, customers, employees, etc."
)

if uploaded_file and not st.session_state.analysis_complete:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    with st.expander("👀 Preview Data"):
        st.dataframe(df.head(5), use_container_width=True)

# ============================================
# SIMPLE GETTING STARTED DIAGRAM
# ============================================
with st.expander("📖 Quick Start Guide", expanded=False):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="step-box">📁 Upload CSV File</div>
        <div class="arrow-down">⬇️</div>
        <div class="step-box">🤖 AI Analyzes Your Data</div>
        <div class="arrow-down">⬇️</div>
        <div class="step-box">📊 Get Executive Report</div>
        """, unsafe_allow_html=True)

    st.markdown("""
    **What happens:**
    1. Upload any CSV file
    2. AI runs statistical tests (correlations, trends, clusters)
    3. Receive executive summary + top 3 insights + recommendations

    **Free to use:** Get your OpenRouter key at [openrouter.ai/keys](https://openrouter.ai/keys)
    """)

# ============================================
# SIDEBAR CONFIGURATION
# ============================================
with st.sidebar:
    st.header("⚙️ Settings")

    # API Configuration
    st.subheader("🔑 OpenRouter API")
    st.caption("Get free key at [openrouter.ai/keys](https://openrouter.ai/keys)")

    openrouter_key = st.text_input(
        "API Key",
        type="password",
        value=st.session_state.openrouter_key,
        help="Paste your OpenRouter API key here"
    )
    st.session_state.openrouter_key = openrouter_key

    # Model Configuration
    if openrouter_key:
        openrouter_model = st.text_input(
            "Model Name",
            value=st.session_state.openrouter_model or "",
            help="Paste any OpenRouter model ID, for example meta-llama/llama-3.2-1b-instruct:free"
        )
        st.session_state.openrouter_model = openrouter_model.strip() or None
        if st.session_state.openrouter_model:
            st.success("✅ Ready to analyze!")
        else:
            st.warning("Enter a model name to run LLM-powered analysis.")
    else:
        st.info("ℹ️ No API key? Analysis will use template mode (still works great!)")
        st.session_state.openrouter_model = None

    st.markdown("---")

    # Analysis Settings
    st.subheader("🔬 Analysis")
    max_hypotheses = st.slider("Max Hypotheses", 3, 10, 7)
    significance_level = st.select_slider(
        "Significance",
        options=["0.01", "0.05", "0.10"],
        value="0.05"
    )


def format_metric(value):
    """Format KPI values for report cards."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        if abs(value) >= 100:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:,.1f}"
        return f"{value:,.2f}"
    return str(value)


def detect_date_column(df: pd.DataFrame):
    """Find a likely date column without mutating the source dataframe."""
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            return col
        if pd.api.types.is_numeric_dtype(series):
            continue
        parsed = pd.to_datetime(series, errors="coerce")
        if len(parsed) and parsed.notna().mean() >= 0.8 and parsed.nunique() > 1:
            return col
    return None


def make_priority_chip(label: str, tone: str) -> str:
    colors = {
        "high": ("#fee2e2", "#991b1b"),
        "medium": ("#fef3c7", "#92400e"),
        "low": ("#dcfce7", "#166534"),
        "neutral": ("#e8eff6", "#294766"),
    }
    bg, fg = colors.get(tone, colors["neutral"])
    return f"<span class='priority-chip' style='background:{bg};color:{fg};'>{label}</span>"


def get_summary_cards(df: pd.DataFrame, report: dict):
    missing_pct = float((df.isna().sum().sum() / max(df.size, 1)) * 100)
    significant = report.get("n_significant", 0)
    total_findings = max(report.get("n_findings", 0), 1)
    significance_rate = significant / total_findings * 100
    date_col = detect_date_column(df)

    cards = [
        ("Dataset Size", format_metric(len(df)), f"{df.shape[1]} variables assessed"),
        ("Signal Yield", f"{significance_rate:.0f}%", f"{significant} of {report.get('n_findings', 0)} findings significant"),
        ("Data Quality", f"{missing_pct:.1f}%", "Average missing-cell rate"),
    ]
    if date_col:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        cards.append(("Time Window", f"{parsed.min().date()} to {parsed.max().date()}", date_col))
    else:
        cards.append(("Coverage", format_metric(df.select_dtypes(include=[np.number]).shape[1]), "Numeric metrics available"))
    return cards


def render_insight_chart(df: pd.DataFrame, insight: dict, chart_key: str):
    """Render the most appropriate visual for an insight."""
    raw = insight.get("raw_finding", {}) or {}
    test_type = insight.get("test_type", "") or raw.get("test_type", "")
    variables = [v for v in insight.get("variables", []) if v in df.columns]

    try:
        if test_type == "pearson_correlation" and len(variables) >= 2:
            x_col, y_col = variables[:2]
            chart_df = df[[x_col, y_col]].dropna().copy()
            if len(chart_df) >= 3:
                slope, intercept = np.polyfit(chart_df[x_col], chart_df[y_col], 1)
                trend_y = slope * chart_df[x_col] + intercept
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=chart_df[x_col],
                    y=chart_df[y_col],
                    mode="markers",
                    marker=dict(size=8, color="#2c5b88", opacity=0.75),
                    name="Observed"
                ))
                fig.add_trace(go.Scatter(
                    x=chart_df[x_col],
                    y=trend_y,
                    mode="lines",
                    line=dict(color="#9fb9d1", width=3),
                    name="Trend"
                ))
                fig.update_layout(title=f"Relationship between {x_col} and {y_col}")
                fig.update_layout(template="plotly_white", height=360, margin=dict(t=70, l=20, r=20, b=20))
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                try:
                    img_bytes = fig.to_image(format="png")
                    st.session_state.charts_mapping[chart_key] = base64.b64encode(img_bytes).decode('utf-8')
                except Exception:
                    pass
                return True

        if test_type == "t_test" and len(variables) >= 2:
            value_col, group_col = variables[:2]
            chart_df = df[[group_col, value_col]].dropna().copy()
            if chart_df[group_col].nunique() >= 2:
                top_groups = chart_df[group_col].astype(str).value_counts().head(6).index
                chart_df = chart_df[chart_df[group_col].astype(str).isin(top_groups)]
                fig = create_box_plot(chart_df, group_col, value_col, title=f"{value_col} distribution by {group_col}")
                fig.update_layout(height=360, margin=dict(t=70, l=20, r=20, b=20))
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                try:
                    img_bytes = fig.to_image(format="png")
                    st.session_state.charts_mapping[chart_key] = base64.b64encode(img_bytes).decode('utf-8')
                except Exception:
                    pass
                return True

        if test_type == "trend_analysis":
            date_col = variables[0] if variables and variables[0] in df.columns else detect_date_column(df)
            value_col = variables[1] if len(variables) >= 2 else next((col for col in variables if col in df.columns), None)
            if date_col and value_col:
                chart_df = df[[date_col, value_col]].dropna().copy()
                chart_df[date_col] = pd.to_datetime(chart_df[date_col], errors="coerce")
                chart_df = chart_df.dropna().sort_values(date_col)
                if len(chart_df) >= 3:
                    fig = create_trend_chart(chart_df, date_col, value_col, title=f"{value_col} over time")
                    fig.update_layout(height=360, margin=dict(t=70, l=20, r=20, b=20))
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)
                    try:
                        img_bytes = fig.to_image(format="png")
                        st.session_state.charts_mapping[chart_key] = base64.b64encode(img_bytes).decode('utf-8')
                    except Exception:
                        pass
                    return True

        if test_type == "chi_square" and len(variables) >= 2:
            col1, col2 = variables[:2]
            chart_df = df[[col1, col2]].dropna().copy()
            if chart_df[col1].nunique() >= 2 and chart_df[col2].nunique() >= 2:
                crosstab = pd.crosstab(chart_df[col1], chart_df[col2])
                fig = px.imshow(
                    crosstab,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    title=f"Association between {col1} and {col2}"
                )
                fig.update_layout(template="plotly_white", height=360, margin=dict(t=70, l=20, r=20, b=20))
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                try:
                    img_bytes = fig.to_image(format="png")
                    st.session_state.charts_mapping[chart_key] = base64.b64encode(img_bytes).decode('utf-8')
                except Exception:
                    pass
                return True

        if "outlier_detection" in test_type:
            value_col = variables[0] if variables else raw.get("variable")
            if value_col in df.columns:
                fig = go.Figure(go.Box(
                    y=df[value_col].dropna(),
                    boxpoints="outliers",
                    marker_color="#2c5b88",
                    line_color="#10233c",
                    name=value_col
                ))
                fig.update_layout(
                    template="plotly_white",
                    title=f"Outlier profile for {value_col}",
                    height=360,
                    margin=dict(t=70, l=20, r=20, b=20),
                    yaxis_title=value_col
                )
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                try:
                    img_bytes = fig.to_image(format="png")
                    st.session_state.charts_mapping[chart_key] = base64.b64encode(img_bytes).decode('utf-8')
                except Exception:
                    pass
                return True

        if test_type == "distribution_analysis":
            value_col = variables[0] if variables else raw.get("variable")
            if value_col in df.columns:
                fig = create_distribution_plot(df, value_col, title=f"Distribution of {value_col}")
                fig.update_layout(height=360, margin=dict(t=70, l=20, r=20, b=20))
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                try:
                    img_bytes = fig.to_image(format="png")
                    st.session_state.charts_mapping[chart_key] = base64.b64encode(img_bytes).decode('utf-8')
                except Exception:
                    pass
                return True

        if len(variables) >= 2:
            cat_col = next((col for col in variables if not pd.api.types.is_numeric_dtype(df[col])), None)
            num_col = next((col for col in variables if pd.api.types.is_numeric_dtype(df[col])), None)
            if cat_col and num_col:
                chart_df = df[[cat_col, num_col]].dropna().copy()
                agg = chart_df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(8).reset_index()
                fig = px.bar(
                    agg,
                    x=num_col,
                    y=cat_col,
                    orientation="h",
                    color=num_col,
                    color_continuous_scale=["#cfe0f1", "#2c5b88"],
                    title=f"{num_col} by {cat_col}"
                )
                fig.update_layout(template="plotly_white", height=360, margin=dict(t=70, l=20, r=20, b=20), coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                try:
                    img_bytes = fig.to_image(format="png")
                    st.session_state.charts_mapping[chart_key] = base64.b64encode(img_bytes).decode('utf-8')
                except Exception:
                    pass
                return True
    except Exception:
        return False

    return False


def generate_follow_up_response(question: str, df: pd.DataFrame, findings: list) -> str:
    """Generate simple follow-up responses."""
    q = question.lower()

    if 'correlation' in q:
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            top_corr = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    top_corr.append((numeric_cols[i], numeric_cols[j], corr.iloc[i, j]))
            top_corr.sort(key=lambda x: abs(x[2]), reverse=True)
            if top_corr:
                return f"Strongest correlation: **{top_corr[0][0]}** vs **{top_corr[0][1]}** (r={top_corr[0][2]:.3f})"

    if 'average' in q or 'mean' in q:
        for col in df.columns:
            if col.lower() in q and df[col].dtype in ['int64', 'float64']:
                return f"Average **{col}**: {df[col].mean():.2f}"
        numeric_means = df.select_dtypes(include=[np.number]).mean().sort_values(ascending=False)
        if len(numeric_means) > 0:
            return f"Highest average: **{numeric_means.index[0]}** ({numeric_means.iloc[0]:.2f})"

    if 'top' in q or 'highest' in q:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            return f"Highest **{col}**: {df[col].max():.2f}"

    if 'missing' in q or 'null' in q:
        total = df.isnull().sum().sum()
        return f"Missing values: {total}" if total > 0 else "No missing values"

    if any(x in q for x in ['rows', 'size', 'shape']):
        return f"Dataset: **{df.shape[0]:,} rows** × **{df.shape[1]} columns**"

    return "Check the Insight Deck above, or ask about correlations, averages, or specific columns."

# ============================================
# STEP 2: RUN ANALYSIS (if data uploaded)
# ============================================
if uploaded_file and not st.session_state.analysis_complete:
    st.markdown("---")
    st.markdown("### 🚀 Step 2: Start Analysis")

    if st.button("▶️ Analyze My Data", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Import modules
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from hypothesis.generator import HypothesisGenerator, profile_data
            from hypothesis.tester import HypothesisTester
            from insights.ranker import InsightRanker, Synthesizer
            from tools.stats_tests import auto_explore

            df = st.session_state.df

            # Phase 1: Profile
            status_text.text("Profiling data...")
            progress_bar.progress(10)
            profile = profile_data(df)

            # Initialize LLM client
            llm_client = None
            if st.session_state.openrouter_key:
                try:
                    from openai import OpenAI
                    llm_client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=st.session_state.openrouter_key,
                        default_headers={
                            "HTTP-Referer": "https://localhost",
                            "X-Title": "AI Strategy Consultant"
                        }
                    )
                    llm_client.models.list()
                except Exception as e:
                    st.warning(f"API connection failed: {e}")
                    llm_client = None

            # Phase 2: Generate hypotheses
            status_text.text("Generating hypotheses...")
            progress_bar.progress(30)
            generator = HypothesisGenerator(llm_client=llm_client, model=st.session_state.openrouter_model)
            hypotheses = generator.generate_from_profile(profile)

            # Phase 3: Test
            status_text.text("Running statistical tests...")
            progress_bar.progress(50)
            tester = HypothesisTester(df)
            test_results = tester.test_all(hypotheses[:max_hypotheses])
            auto_findings = auto_explore(df)
            all_findings = test_results + auto_findings

            # Phase 4: Rank
            status_text.text("Ranking insights...")
            progress_bar.progress(75)
            ranked_findings = InsightRanker.rank_findings(all_findings, top_n=3)

            # Phase 5: Synthesize
            status_text.text("Creating report...")
            progress_bar.progress(85)
            report = Synthesizer.synthesize_insights(ranked_findings, df, llm_client=llm_client, model=st.session_state.openrouter_model)

            # Phase 6: Vector Database
            status_text.text("Storing report in Vector DB...")
            progress_bar.progress(95)
            try:
                from utils.vector_db import ReportVectorDB
                vector_db = ReportVectorDB()
                vector_db.store_report(report)
                st.session_state.vector_db = vector_db
            except Exception as e:
                st.warning(f"Failed to initialize Vector DB: {e}")

            # Store results
            st.session_state.report_data = report
            st.session_state.ranked_findings = ranked_findings
            st.session_state.analysis_complete = True

            progress_bar.progress(100)
            time.sleep(0.3)
            progress_bar.empty()
            status_text.empty()

            st.rerun()

# ============================================
# DISPLAY RESULTS
# ============================================
if st.session_state.analysis_complete and st.session_state.report_data:
    report = st.session_state.report_data
    df = st.session_state.df

    st.markdown("---")
    st.markdown("## Consulting Report")

    # Executive Summary
    top_insights = report.get("top_insights", [])
    lead_insight = top_insights[0]["title"] if top_insights else "No critical issue flagged"
    st.markdown(
        f"""
        <div class="report-hero">
            <div class="report-kicker">Executive Readout</div>
            <div class="report-headline">{lead_insight}</div>
            <div class="report-subtext">{report.get('executive_summary', 'No significant findings detected.')}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Key Metrics
    summary_cols = st.columns(4)
    for col, (label, value, note) in zip(summary_cols, get_summary_cards(df, report)):
        with col:
            st.markdown(
                f"""
                <div class="report-strip">
                    <div class="report-strip-label">{label}</div>
                    <div class="report-strip-value">{value}</div>
                    <div class="report-strip-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("")
    lead_left, lead_right = st.columns([1.3, 1])
    with lead_left:
        st.markdown("<div class='section-label'>What Management Should Know</div>", unsafe_allow_html=True)
        if top_insights:
            for insight in top_insights[:3]:
                st.markdown(
                    f"""
                    <div class="insight-detail">
                        <b>{insight.get('title', 'Key insight')}</b><br>
                        {insight.get('why_it_matters') or insight.get('narrative', '')}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No standout insights were generated from the current dataset.")
    with lead_right:
        st.markdown("<div class='section-label'>Action Agenda</div>", unsafe_allow_html=True)
        for i, rec in enumerate(report.get('recommendations', [])[:3], 1):
            st.markdown(f"<div class='recommendation-card'><b>{i}.</b> {rec}</div>", unsafe_allow_html=True)
        if not report.get('recommendations'):
            st.markdown("<div class='recommendation-card'>No immediate actions were auto-generated.</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Top Insights
    st.markdown("### Insight Deck")

    for i, insight in enumerate(top_insights, 1):
        stats = insight.get("statistics", {})
        p_value = stats.get("p_value")
        effect_size = stats.get("effect_size")
        business_score = stats.get("business_score")
        significance_chip = make_priority_chip(
            "Statistically significant" if p_value is not None and p_value < 0.05 else "Directional / descriptive",
            "high" if p_value is not None and p_value < 0.05 else "neutral"
        )
        impact_chip = make_priority_chip(
            f"Impact {business_score:.0f}/100" if business_score is not None else "Impact not scored",
            "medium" if business_score and business_score >= 50 else "low"
        )

        st.markdown("<div class='insight-shell'>", unsafe_allow_html=True)
        info_col, chart_col = st.columns([1.05, 1.2])

        with info_col:
            st.markdown(f"<div class='insight-index'>Insight {i}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='insight-title'>{insight.get('title', f'Insight {i}')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='insight-summary'>{insight.get('narrative', '')}</div>", unsafe_allow_html=True)
            st.markdown(significance_chip + impact_chip, unsafe_allow_html=True)
            if effect_size:
                st.markdown(make_priority_chip(f"Effect size: {effect_size}", "neutral"), unsafe_allow_html=True)
            if insight.get("why_it_matters"):
                st.markdown(
                    f"<div class='insight-detail'><b>Why it matters:</b><br>{insight['why_it_matters']}</div>",
                    unsafe_allow_html=True
                )
            if insight.get("interpretation"):
                st.caption(insight["interpretation"])

        with chart_col:
            chart_rendered = render_insight_chart(df, insight, chart_key=f"insight_chart_{i}")
            if not chart_rendered:
                st.info("No suitable visual could be generated for this insight.")

        evidence_cols = st.columns(3)
        evidence_cols[0].metric("p-value", f"{p_value:.4f}" if p_value is not None else "N/A")
        evidence_cols[1].metric("Effect size", str(effect_size).title() if effect_size else "N/A")
        evidence_cols[2].metric("Variables", len(insight.get("variables", [])))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    plan_col, risk_col = st.columns([1.2, 0.8])
    with plan_col:
        st.markdown("### Recommended Actions")
        for i, rec in enumerate(report.get('recommendations', [])[:5], 1):
            st.markdown(f"<div class='recommendation-card'><b>{i}.</b> {rec}</div>", unsafe_allow_html=True)
        if not report.get('recommendations'):
            st.markdown("<div class='recommendation-card'>No immediate actions were auto-generated.</div>", unsafe_allow_html=True)
    with risk_col:
        st.markdown("### Caveats")
        for caveat in report.get('caveats', []):
            st.markdown(f"<div class='caveat-card'>{caveat}</div>", unsafe_allow_html=True)
        if not report.get('caveats'):
            st.markdown("<div class='caveat-card'>No major caveats flagged.</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Export
    st.markdown("### Export")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 PowerPoint", use_container_width=True):
            with st.spinner("Generating..."):
                from utils.pptx_export import generate_pptx_report
                try:
                    pptx_bytes = generate_pptx_report(report, st.session_state.get('charts_mapping', {}))
                    st.download_button(
                        label="⬇️ Download PPTX",
                        data=pptx_bytes,
                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if st.button("📄 PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                from utils.pdf_export import generate_pdf_report
                try:
                    pdf_bytes = generate_pdf_report(report)
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    with col3:
        if st.button("📝 Markdown", use_container_width=True):
            md = f"# Report\n\n## Summary\n{report['executive_summary']}\n\n## Insights\n"
            for i, ins in enumerate(report['top_insights'], 1):
                md += f"\n### {i}. {ins['title']}\n{ins['narrative']}\n"
            md += "\n## Recommendations\n" + "\n".join([f"{i+1}. {r}" for i, r in enumerate(report.get('recommendations', []))])
            st.download_button(
                label="⬇️ Download MD",
                data=md,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True
            )

    # Chat
    st.markdown("---")
    st.markdown("### Questions?")

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat['question'])
        with st.chat_message("assistant"):
            st.write(chat['answer'])

    question = st.chat_input("Ask about the analysis...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):
            if st.session_state.get('vector_db'):
                context = st.session_state.vector_db.retrieve_context(question)
                from agents.chat_agent import chat_with_report
                
                llm_client = None
                if st.session_state.openrouter_key:
                    try:
                        from openai import OpenAI
                        llm_client = OpenAI(
                            base_url="https://openrouter.ai/api/v1",
                            api_key=st.session_state.openrouter_key,
                            default_headers={
                                "HTTP-Referer": "https://localhost",
                                "X-Title": "AI Strategy Consultant"
                            }
                        )
                    except Exception:
                        pass
                
                answer = chat_with_report(question, context, llm_client=llm_client, model=st.session_state.openrouter_model)
            else:
                answer = generate_follow_up_response(question, df, st.session_state.ranked_findings)
            st.session_state.chat_history.append({'question': question, 'answer': answer})

            with st.chat_message("assistant"):
                st.write(answer)


# Footer
st.markdown("---")
st.caption("Built with CrewAI • OpenRouter • Statistical Rigor")
