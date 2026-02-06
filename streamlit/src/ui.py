# src/ui.py
import streamlit as st

CSS = """
<style>
.kpi-card {
  background: #ffffff;
  border: 1px solid #e8e8e8;
  border-radius: 14px;
  padding: 16px 16px 12px 16px;
  box-shadow: 0 1px 10px rgba(0,0,0,0.04);

  /* Igualar tamaño de tarjetas */
  min-height: 130px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.kpi-title { font-size: 12px; color: #6b7280 !important; margin-bottom: 8px; }
.kpi-value { font-size: 28px; font-weight: 700; margin: 0; color: #111827 !important; }
.kpi-sub   { font-size: 12px; color: #6b7280 !important; margin-top: 6px; line-height: 1.3; }
.small-note { font-size: 12px; color: #6b7280 !important; }
</style>
"""

def apply_css():
    st.markdown(CSS, unsafe_allow_html=True)

def kpi_card(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <p class="kpi-value">{value}</p>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
