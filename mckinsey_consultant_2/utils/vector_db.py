import os
from typing import List, Dict, Any

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.docstore.document import Document
except ImportError:
    pass

class ReportVectorDB:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None

    def store_report(self, report_data: Dict[str, Any]):
        """Convert the report to text chunks and store in FAISS."""
        documents = []
        
        # 1. Executive Summary
        exec_summary = report_data.get("executive_summary", "")
        if exec_summary:
            documents.append(Document(page_content=f"Executive Summary:\n{exec_summary}", metadata={"section": "executive_summary"}))
        
        # 2. Insights
        for i, insight in enumerate(report_data.get("top_insights", [])):
            title = insight.get("title", "")
            narrative = insight.get("narrative", "")
            why = insight.get("why_it_matters", "")
            content = f"Insight {i+1}: {title}\nDescription: {narrative}\nWhy it matters: {why}"
            documents.append(Document(page_content=content, metadata={"section": "insight", "insight_index": i}))
            
        # 3. Recommendations
        for i, rec in enumerate(report_data.get("recommendations", [])):
            documents.append(Document(page_content=f"Recommendation {i+1}:\n{rec}", metadata={"section": "recommendation", "recommendation_index": i}))
            
        # 4. Caveats
        for i, caveat in enumerate(report_data.get("caveats", [])):
            documents.append(Document(page_content=f"Caveat {i+1}:\n{caveat}", metadata={"section": "caveat", "caveat_index": i}))

        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
    def retrieve_context(self, question: str, k: int = 3) -> str:
        """Fetch relevant chunks from the FAISS index based on the question."""
        if not self.vector_store:
            return "No report data is currently stored in the vector database."
            
        docs = self.vector_store.similarity_search(question, k=k)
        
        context_str = "\n\n".join([f"Relevant Context (Section: {doc.metadata.get('section')}):\n{doc.page_content}" for doc in docs])
        return context_str
