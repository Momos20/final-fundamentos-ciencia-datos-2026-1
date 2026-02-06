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

def show_splash(msg="Procesando...", sub=""):
    ph = st.empty()
    ph.markdown(
        f"""
        <style>
        .splash-overlay {{
            position: fixed;
            inset: 0;
            background: rgba(17, 24, 39, 0.35); /* oscurece el fondo */
            z-index: 999999;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .splash-card {{
            width: min(680px, 92vw);
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 18px;
            padding: 22px 22px 18px 22px;
            box-shadow: 0 18px 50px rgba(0,0,0,0.25);
            background: rgba(255,255,255,0.96);
        }}
        .splash-row {{
            display: flex;
            gap: 12px;
            align-items: center;
            margin-bottom: 10px;
        }}
        .spinner {{
            width: 18px;
            height: 18px;
            border: 3px solid rgba(17,24,39,0.18);
            border-top-color: rgba(17,24,39,0.8);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

        .splash-title {{
            font-size: 16px;
            font-weight: 750;
            color: #111827;
            margin: 0;
        }}
        .splash-sub {{
            font-size: 13px;
            color: #4b5563;
            margin: 0 0 12px 0;
        }}

        .bar {{
            height: 8px;
            border-radius: 999px;
            background: rgba(17,24,39,0.10);
            overflow: hidden;
        }}
        .bar > div {{
            height: 100%;
            width: 35%;
            background: rgba(17,24,39,0.60);
            border-radius: 999px;
            animation: move 1.1s ease-in-out infinite;
        }}
        @keyframes move {{
            0% {{ transform: translateX(-60%); }}
            50% {{ transform: translateX(140%); }}
            100% {{ transform: translateX(-60%); }}
        }}
        </style>

        <div class="splash-overlay">
          <div class="splash-card">
            <div class="splash-row">
              <div class="spinner"></div>
              <p class="splash-title">{msg}</p>
            </div>
            <p class="splash-sub">{sub}</p>
            <div class="bar"><div></div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return ph

def update_splash(ph, msg="Procesando...", sub=""):
    # re-render del mismo placeholder con el nuevo mensaje
    ph.markdown(
        f"""
        <div class="splash-overlay">
          <div class="splash-card">
            <div class="splash-row">
              <div class="spinner"></div>
              <p class="splash-title">{msg}</p>
            </div>
            <p class="splash-sub">{sub}</p>
            <div class="bar"><div></div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def hide_splash(ph):
    ph.empty()


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
