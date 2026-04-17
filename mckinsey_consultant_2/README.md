# McKinsey-Style AI Data Consultant

## 🎯 Overview

A **virtual McKinsey consulting team** that automatically analyzes your data and delivers consulting-grade insights without you asking a single question.

**Core Promise:** *"Upload your data. Our AI consultants will discover what matters, tell you what you didn't know, and recommend what to do next."*

---

## 👥 The Consulting Team

| Role | Responsibility |
|------|----------------|
| **Engagement Manager** | Orchestrates the team, prioritizes hypotheses, knows when to stop |
| **Data Scientist** | Runs statistical tests (t-tests, correlations, trend analysis) |
| **Business Analyst** | Finds segments, clusters, and high-value cohorts |
| **Visualization Lead** | Creates consulting-grade charts (waterfall, annotated bars, heatmaps) |
| **Associate** | Writes executive summaries, recommendations, caveats |

---

## 🔄 How It Works

```
Upload CSV
    │
    ▼
┌────────────────────────────────────────────┐
│ Phase 1: Discovery (Automatic)             │
│  • Data profiling                          │
│  • Statistical exploration                 │
│  • Pattern detection                       │
└────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────┐
│ Phase 2: Hypothesis Generation             │
│  • LLM generates 5-7 business hypotheses   │
│  • Prioritizes by business impact          │
└────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────┐
│ Phase 3: Hypothesis Testing                │
│  • Runs appropriate statistical tests      │
│  • Calculates p-values, effect sizes       │
│  • Ranks by significance + business score │
└────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────┐
│ Phase 4: Insight Synthesis                 │
│  • Top 3 insights with evidence            │
│  • Executive summary                       │
│  • Actionable recommendations              │
│  • Professional PowerPoint export          │
└────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project
cd mckinsey_consultant

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Usage

1. **Upload a CSV file** - Any tabular data (sales, customer, financial, etc.)
2. **Click "Start AI Analysis"** - The consulting team runs automatically
3. **Review the report** - Executive summary, key insights, recommendations
4. **Export** - Download as PowerPoint or Markdown
5. **Ask follow-ups** - Chat interface for drill-down questions

---

## 📊 Example Output

**Executive Summary:**
> Your sales declined 12% in Q3, driven entirely by the West region (-31% YoY). However, the new Premium product line grew 47% in the same period, concentrated among customers with >2 years tenure.

**Key Insight:**
- **Regional Divergence Detected** (p<0.001)
- West region decline: -$2.1M
- East & Central stable (p>0.05)
- **Recommendation:** Investigate West logistics or pricing changes

---

## 🛠️ Technical Architecture

### Agents (CrewAI)
- `orchestrator.py` - Engagement Manager agent
- `explorer.py` - Data Scientist agent
- `segmenter.py` - Business Analyst agent
- `visualiser.py` - Visualization Lead agent
- `reporter.py` - Associate agent

### Tools
- `stats_tests.py` - Statistical testing library (scipy-based)
- `clustering.py` - K-means, RFM, cohort analysis
- `charts.py` - Plotly visualizations
- `hypothesis/` - Hypothesis generation and testing
- `insights/` - Insight ranking and synthesis

### Export
- `pptx_export.py` - PowerPoint generation (python-pptx)

---

## ⚙️ Configuration

In the Streamlit sidebar, you can configure:

- **API Keys** (optional) - For enhanced LLM-powered hypothesis generation
- **Max Hypotheses** - Number of hypotheses to test (3-10)
- **Significance Level** - Statistical threshold (0.01, 0.05, 0.10)

---

## 📈 Statistical Capabilities

- ✅ Pearson/Spearman correlation
- ✅ T-tests for group comparisons
- ✅ Chi-square for categorical associations
- ✅ Trend analysis for time series
- ✅ Outlier detection (IQR, z-score)
- ✅ Distribution analysis
- ✅ K-means clustering
- ✅ RFM customer segmentation
- ✅ Cohort retention analysis
- ✅ Pareto analysis (80/20 rule)

---

## 📦 Requirements

```
crewai>=0.35.0
pandas>=2.0.0
scipy>=1.10.0
plotly>=5.15.0
streamlit>=1.28.0
python-pptx>=0.6.21
scikit-learn>=1.3.0
```

---

## 🎓 Academic Value

This project demonstrates:
- **Multi-agent collaboration** - Full consulting team simulation
- **Proactive insight generation** - No user prompt required
- **Statistical rigor** - P-values, effect sizes, proper tests
- **Business storytelling** - Executive summaries, recommendations
- **Conversational memory** - Follow-up drill-downs
- **Professional exports** - PowerPoint consulting decks

---

## 📄 License

MIT License - Free for academic and commercial use.

---

**Built with CrewAI • Statistical rigor meets business storytelling**
