# 🚀 AI-Powered Data Analyst

An intelligent AI-driven system that automates the complete data analysis pipeline — from raw data to actionable insights — with built-in visualization, reporting, and conversational querying using Retrieval-Augmented Generation (RAG).

---

## 🧠 Overview

This project acts as a **virtual data analyst**, enabling users to upload datasets and automatically receive:

* 📊 Data insights
* 📈 Visualizations
* 🧾 Executive summaries
* 💡 Business recommendations
* 💬 Chat-based interaction with data

It combines **data science, statistical analysis, and large language models (LLMs)** to deliver consulting-grade outputs.

---

## ❗ Problem Statement

Data analysis traditionally requires:

* Technical expertise (Python, statistics)
* Significant time for exploration and validation
* Manual effort in generating insights and reports

Most users struggle to extract meaningful insights from data efficiently.

---

## 💡 Solution

This system automates the entire workflow:

```bash
Upload Data → Analyze → Generate Insights → Visualize → Chat → Export Reports
```

It removes the need for deep technical knowledge and enables **fast, intelligent, and interactive analysis**.

---

## ⚙️ Features

* ✅ Automated Data Profiling
* 🤖 AI-Based Hypothesis Generation
* 📊 Statistical Testing (Correlation, T-Test, Chi-Square)
* 🏆 Insight Ranking & Prioritization
* 🧠 LLM-Based Insight Synthesis
* 🔍 Vector Database (Semantic Search)
* 💬 Chat with Data (RAG Pipeline)
* 📈 Dynamic Visualizations (Plotly)
* 📄 Export Reports (PPT, PDF, Markdown)

---

## 🧩 System Architecture

```
User Input (CSV)
        ↓
Data Profiling
        ↓
Hypothesis Generation (AI)
        ↓
Statistical Testing
        ↓
Insight Ranking
        ↓
Insight Synthesis (LLM)
        ↓
Vector Database (Embeddings)
        ↓
Visualization + Reports
        ↓
Chat Interface (RAG)
```

---

## 🔄 Workflow

1. Upload dataset (CSV)
2. System profiles the data
3. Generates hypotheses using AI
4. Validates them with statistical tests
5. Ranks the most important insights
6. Converts them into human-readable explanations
7. Stores insights in a vector database
8. Enables chat-based querying
9. Generates reports

---

## 🧠 Vector Database & RAG

### 🔹 Vector Database

* Stores insights as embeddings
* Enables semantic (meaning-based) search

### 🔹 RAG (Retrieval-Augmented Generation)

```
User Query → Retrieve Context → LLM → Answer
```

This ensures responses are:

* Context-aware
* Data-driven
* Accurate

---

## 💻 Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly
* **AI Models:** OpenRouter (LLMs)
* **Vector DB:** Custom implementation
* **Exports:** PPTX, PDF

---

## 📁 Project Structure

```
.
├── app.py
├── hypothesis/
│   ├── generator.py
│   └── tester.py
├── insights/
│   ├── ranker.py
│   └── synthesizer.py
├── utils/
│   ├── charts.py
│   ├── vector_db.py
│   ├── pptx_export.py
│   └── pdf_export.py
├── agents/
│   └── chat_agent.py
├── tools/
│   └── stats_tests.py
└── requirements.txt
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-data-analyst.git
cd ai-data-analyst
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add API Key (optional)

Create `.env` file:

```
OPENROUTER_API_KEY=your_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📊 Demo Use Cases

* 📈 Sales analysis
* 🛒 Customer segmentation
* 🧑‍💼 HR analytics
* 💰 Financial insights

---

## ✅ Advantages

* Fully automated analysis
* No coding required
* Fast and scalable
* Interactive (chat-based)
* Professional reports

---

## ⚠️ Limitations

* Depends on data quality
* Custom vector DB not highly scalable
* Limited real-time processing

---

## 🔮 Future Enhancements

* Integration with FAISS / Pinecone
* Real-time analytics
* Cloud deployment
* Multi-modal data support

---

## 🧠 Key Concept

> This project combines **AI + Statistics + RAG** to build an intelligent decision-support system.

---

## 🤝 Contributing

Feel free to fork the repository and improve features!

---

## 📜 License

This project is for academic and learning purposes.

---

## 🙌 Acknowledgements

* OpenAI / OpenRouter
* Streamlit
* Plotly
* Research papers on RAG & Vector Databases

---
