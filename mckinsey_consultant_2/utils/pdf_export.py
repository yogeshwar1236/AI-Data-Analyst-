import os
from fpdf import FPDF
import io

def clean_text(text: str) -> str:
    """Removes or replaces characters that are incompatible with fpdf2 latin-1 fonts."""
    if not isinstance(text, str):
        return str(text)
    # Encode to latin-1 and ignore errors (drops emojis and unsupported unicode)
    return text.encode('latin-1', 'ignore').decode('latin-1')

class ReportPDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('helvetica', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'AI Consultant Report', 0, 0, 'C')
        # Line break
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('helvetica', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(report_data: dict) -> bytes:
    """Generates a PDF byte string from the report data using fpdf2."""
    pdf = ReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font("helvetica", "B", 20)
    lead_insight = "No critical issue flagged"
    if report_data.get("top_insights"):
        lead_insight = report_data["top_insights"][0].get("title", lead_insight)
    pdf.multi_cell(0, 10, clean_text(lead_insight), align='C')
    pdf.ln(10)
    
    # Executive Summary
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Executive Summary", 0, 1)
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 8, clean_text(report_data.get("executive_summary", "No significant findings.")))
    pdf.ln(5)
    
    # Insights
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Key Insights", 0, 1)
    for i, insight in enumerate(report_data.get("top_insights", [])):
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, clean_text(f"{i+1}. {insight.get('title', 'Insight')}"), 0, 1)
        pdf.set_font("helvetica", "", 11)
        pdf.multi_cell(0, 6, clean_text(insight.get("narrative", "")))
        why = insight.get("why_it_matters", "")
        if why:
            pdf.set_font("helvetica", "I", 11)
            pdf.multi_cell(0, 6, clean_text(f"Why it matters: {why}"))
        pdf.ln(5)
        
    # Recommendations
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Recommendations", 0, 1)
    pdf.set_font("helvetica", "", 11)
    for i, rec in enumerate(report_data.get("recommendations", [])):
        pdf.multi_cell(0, 6, clean_text(f"{i+1}. {rec}"))
        pdf.ln(2)
        
    # Caveats
    if report_data.get("caveats"):
        pdf.ln(5)
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "Caveats", 0, 1)
        pdf.set_font("helvetica", "", 11)
        for i, cav in enumerate(report_data.get("caveats", [])):
            pdf.multi_cell(0, 6, clean_text(f"- {cav}"))
            pdf.ln(2)
            
    # Output to bytes
    pdf_bytes = pdf.output(dest="S")
    return pdf_bytes
