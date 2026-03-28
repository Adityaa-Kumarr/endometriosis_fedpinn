import os
import tempfile
import numpy as np
import time
from fpdf import FPDF
import plotly.graph_objects as go

class AdvancedPDFReport(FPDF):
    def __init__(self):
        super().__init__()
        # Initialize custom fonts from standard ones or fpdf built-ins if custom not provided,
        # but typical production would add actual TTFs. We'll use Arial for premium feel over default.
        self.set_auto_page_break(auto=True, margin=15)
        
        # Primary Brand Colors
        self.c_primary = (232, 62, 140)    # E83E8C Pink
        self.c_bg_dark = (15, 23, 42)      # Slate 900
        self.c_text_light = (248, 250, 252) # Slate 50
        self.c_text_muted = (148, 163, 184) # Slate 400
        
    def header(self):
        # Premium dark header
        self.set_fill_color(*self.c_bg_dark)
        self.rect(0, 0, 210, 40, 'F')
        
        # Title
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(*self.c_text_light)
        self.set_y(15)
        self.cell(0, 10, "Endometriosis AI Analysis Report", align="C")
        
        # Subtitle
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(*self.c_primary)
        self.set_y(25)
        self.cell(0, 10, "Advanced Multi-Modal Federated Prediction", align="C")
        self.ln(20)

    def footer(self):
        # Dark footer
        self.set_y(-15)
        self.set_fill_color(*self.c_bg_dark)
        self.rect(0, 282, 210, 15, 'F')
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*self.c_text_light)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
        
    def add_section_title(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*self.c_primary)
        self.cell(0, 15, title, ln=True)
        self.set_draw_color(*self.c_primary)
        # Line under title
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(5)
        
    def add_metric_card(self, x, y, w, h, label, value, color=(255, 255, 255)):
        # Simulate a glassmorphism card in PDF (solid dark gray with border)
        self.set_xy(x, y)
        self.set_fill_color(30, 41, 59) # Slate 800
        self.set_draw_color(*color)
        self.set_line_width(0.5)
        self.rect(x, y, w, h, 'DF')
        
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(148, 163, 184)
        self.set_xy(x, y+5)
        self.cell(w, 5, label, align="C")
        
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*color)
        self.set_xy(x, y+12)
        self.cell(w, 10, str(value), align="C")

def generate_advanced_pdf_report(patient_data, p_prob, p_std, stage_names, p_stage, f_risk, report_text, patient_plan, fig_3d, fig_radar, fig_heat, fig_xai, explanation):
    pdf = AdvancedPDFReport()
    pdf.add_page()
    
    # ---------------------------------------------------------
    # PAGE 1: Clinical Profile & AI Synthesis
    # ---------------------------------------------------------
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(40, 40, 40)
    
    pdf.add_section_title("1. Patient Clinical Metrics")
    
    cd = patient_data[0]
    metrics = [
        ("Age", f"{cd[0]:.0f}"),
        ("BMI", f"{cd[1]:.1f}"),
        ("Pelvic Pain", f"{cd[2]:.0f}/10"),
        ("Dysmenorrhea", f"{cd[3]:.0f}/10"),
        ("Dyspareunia", "Yes" if cd[4] else "No"),
        ("Family Hist.", "Yes" if cd[5] else "No"),
        ("CA-125", f"{cd[6]:.1f} U/mL"),
        ("Estradiol", f"{cd[7]:.1f} pg/mL"),
        ("Progest.", f"{cd[8]:.1f} ng/mL"),
    ]
    
    # Draw metrics grid
    start_y = pdf.get_y()
    start_x = 10
    col_w = 45
    row_h = 25
    
    for i, (label, val) in enumerate(metrics):
        row = i // 4
        col = i % 4
        x = start_x + (col * (col_w + 3))
        y = start_y + (row * (row_h + 3))
        pdf.add_metric_card(x, y, col_w, row_h, label, val, color=(232, 62, 140))
        
    pdf.set_xy(10, start_y + (3 * (row_h + 3)))
    
    # AI Risk Summary
    pdf.add_section_title("2. AI Diagnostic Prognosis")
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Confidence Score: {p_prob*100:.1f}% (+/- {p_std*100:.1f}%)", ln=True)
    pdf.cell(0, 8, f"Estimated Stage: {stage_names[p_stage]}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Future Risk Trajectory (Years 1, 3, 5):", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"1-Year: {f_risk[0]*100:.1f}%,  3-Year: {f_risk[1]*100:.1f}%,  5-Year: {f_risk[2]*100:.1f}%", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Generative Clinical Synthesis:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    
    # Clean up markdown and Unicode characters for FPDF
    def clean_text(text):
        if not text:
            return ""
        # Remove markdown bold/headers
        text = text.replace('**', '').replace('###', '')
        # Remove common problematic emojis/symbols
        for sym in ['🌿', '⚠', '⚡', '🔬', '🩸', '🧬', '🩺', '✅', '❌', '📈', '📊']:
            text = text.replace(sym, '')
        # Ensure it's latin-1 compatible or strip remaining non-latin-1
        try:
            text.encode('latin-1')
        except UnicodeEncodeError:
            # Fallback: strip any character that can't be encoded in latin-1 (FPDF default)
            text = "".join(c for c in text if ord(c) < 256)
        return text

    clean_report = clean_text(report_text)
    pdf.multi_cell(0, 6, clean_report)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Personal Actionable Plan:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    clean_plan = clean_text(patient_plan)
    pdf.multi_cell(0, 6, clean_plan)
    
    
    # ---------------------------------------------------------
    # Helper to save plotly figs to temp pngs
    # ---------------------------------------------------------
    def embed_fig(fig, width, height, pdf_x, pdf_y=None, pdf_w=190):
        # Modify layout slightly for static export to ensure visibility on white background if plot is dark
        fig_copy = go.Figure(fig)
        
        # If it's the 3D plot, we might need a specific camera or size
        fig_copy.update_layout(
            paper_bgcolor='rgba(255,255,255,1)',
            plot_bgcolor='rgba(255,255,255,1)',
            font=dict(color='black')
        )
        # Fix text colors in 3D plots and dramatically enhance lesion visibility for the printed page
        for trace in fig_copy.data:
            if hasattr(trace, 'textfont') and trace.textfont:
                trace.textfont.color = 'black'
            
            # Identify Scatter3d traces (labels and lesions)
            if hasattr(trace, 'type') and trace.type == 'scatter3d':
                if hasattr(trace, 'marker') and trace.marker:
                    # Enlarge lesions specifically for static PDF visibility
                    if trace.name and 'Lesion' in trace.name:
                        # Increase size substantially
                        current_size = tuple(trace.marker.size) if isinstance(trace.marker.size, (list, tuple, np.ndarray)) else trace.marker.size
                        if isinstance(current_size, (int, float)):
                           trace.marker.size = current_size * 2.5
                        else:
                           trace.marker.size = [s * 2.5 for s in current_size]
                        
                        # Make fully opaque and add heavy black outline
                        trace.marker.opacity = 1.0
                        if trace.marker.line:
                            trace.marker.line.color = 'black'
                            trace.marker.line.width = 3
                        else:
                            trace.marker.line = dict(color='black', width=3)
                            
                    elif trace.marker.line:
                        trace.marker.line.color='black'
            
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            # high scale for retina quality in PDF
            fig_copy.write_image(tmp.name, width=width, height=height, scale=2)
            if pdf_y:
                pdf.image(tmp.name, x=pdf_x, y=pdf_y, w=pdf_w)
            else:
                pdf.image(tmp.name, x=pdf_x, w=pdf_w)
            
            # small delay before deleting to ensure fpdf read it
            time.sleep(0.1)
            tmp_name = tmp.name
        return tmp_name
    
    temp_files = []
    
    # ---------------------------------------------------------
    # PAGE 2: 3D Uterus Visualization
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.add_section_title("3. Physics-Informed Digital Twin (3D Render)")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 6, "High-resolution rendering of the generated patient organ geometry, mapping lesions, endometriomas, and simulated adhesions based on the PINN predictions.")
    
    # Embed 3D Plot
    try:
        # Tweak camera angle for static shot
        fig_3d.update_layout(scene_camera=dict(eye=dict(x=-1.25, y=-1.5, z=0.8)))
        t1 = embed_fig(fig_3d, 1000, 800, 10)
        temp_files.append(t1)
    except Exception as e:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"Error rendering 3D image: {e}", ln=True)
        pdf.set_text_color(0, 0, 0)
        
    # ---------------------------------------------------------
    # PAGE 3: Advanced Analytics
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.add_section_title("4. Advanced Analytics & Explainable AI")
    
    # Radar and Heatmap Side-by-Side horizontally
    try:
        # First we need to adjust the layout for PDF
        t2 = embed_fig(fig_radar, 500, 400, 10, pdf.get_y(), 90)
        t3 = embed_fig(fig_heat, 500, 400, 110, pdf.get_y(), 90)
        temp_files.append(t2)
        temp_files.append(t3)
        pdf.set_y(pdf.get_y() + 85) # move down past charts
    except Exception as e:
        pdf.cell(0, 10, f"Error rendering charts: {e}", ln=True)
        
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Feature Importance (SHAP/XAI Explainer):", ln=True)
    pdf.set_font("Helvetica", "", 10)
    
    try:
        t4 = embed_fig(fig_xai, 800, 400, 10, pdf.get_y(), 190)
        temp_files.append(t4)
        pdf.set_y(pdf.get_y() + 95)
    except Exception as e:
         pdf.cell(0, 10, f"Error rendering XAI plot: {e}", ln=True)
         
    clean_exp = explanation.replace('**', '')
    pdf.multi_cell(0, 6, f"XAI Insight: {clean_exp}")

    # Output bytes directly from fpdf
    # In more recent pyfpdf2, output() without a destination returns bytearray
    # By calling output with no arguments or storing as pure bytes, we bypass encode issues
    content = pdf.output()
    if isinstance(content, str):
        content = content.encode('latin-1')
    elif isinstance(content, bytearray):
        content = bytes(content)

    # Cleanup temp files
    for t in temp_files:
        if os.path.exists(t):
            try:
                os.remove(t)
            except:
                pass
                
    return content
