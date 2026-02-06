# app.py
# streamlit run app.py
import numpy as np
import pandas as pd
import streamlit as st

from src.ui import apply_css, kpi_card
from src.data import preprocess_all
from src.kpis import compute_kpi_tables, KPI_TEXT
from src.eda import (
    target_distribution, quantitative_summary, correlation_pairs,
    kruskal_numeric_vs_target, outliers_by_class, cramers_table, perfil_categorico
)
from src import charts
from src.groq_page import render_groq_page


st.set_page_config(page_title="Saber Pro - Dashboard", layout="wide")
apply_css()

st.sidebar.title("Menú")
uploaded = st.sidebar.file_uploader("Subir saber_pro.csv", type=["csv"])

page = st.sidebar.selectbox(
    "Navegación",
    ["Resumen", "Procesamiento", "KPI", "EDA", "Groq IA"],
    index=0
)

if not uploaded:
    st.title("Saber Pro — Streamlit")
    st.info("Cargue el archivo **saber_pro.csv** para comenzar.")
    st.stop()

@st.cache_data(show_spinner=True)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_data(show_spinner=True)
def build_bundle(df: pd.DataFrame) -> dict:
    return preprocess_all(df)

data_raw = load_csv(uploaded)
bundle = build_bundle(data_raw)
data_imp = bundle["data_imp"]
kpi_pack = compute_kpi_tables(data_imp)


# =========================
# Páginas
# =========================
if page == "Resumen":
    st.title("Resumen / Observaciones")
    st.caption("Métricas base + auditoría + hallazgos clave (missingness MNAR).")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Filas", f"{data_raw.shape[0]:,}".replace(",", "."), "Dataset original")
    with c2:
        kpi_card("Columnas", f"{data_raw.shape[1]:,}".replace(",", "."), "Dataset original")
    with c3:
        avg_null = float(data_raw.isna().mean().mean() * 100)
        kpi_card("Nulos promedio", f"{avg_null:.2f}%", "Promedio global (todas las columnas)")
    with c4:
        kw = bundle["kw_p"]
        val = "p≈0" if (isinstance(kw, float) and kw < 1e-6) else f"p={kw:.4g}"
        kpi_card("Kruskal (N_NULOS vs target)", val, "Evidencia de MNAR")

    st.markdown("---")

    st.subheader("Muestra de datos")
    n = st.slider("Filas a mostrar", 5, 200, 20, step=5)
    st.dataframe(data_raw.head(n), use_container_width=True)

    st.subheader("Auditoría de nulos")
    min_null = st.slider("Filtrar: nulos_% mínimo", 0.0, 100.0, 0.0, 1.0)
    audit_view = bundle["audit_df"].loc[bundle["audit_df"]["nulos_%"] >= min_null]
    st.dataframe(audit_view, use_container_width=True)

    st.subheader("Missingness (observaciones)")
    if not bundle["miss_desc"].empty:
        st.write("Descripción de N_NULOS por clase de rendimiento")
        st.dataframe(bundle["miss_desc"], use_container_width=True)
    st.write("Asociación con MISSING_HEAVY (Cramér’s V)")
    st.dataframe(bundle["missing_assoc"].to_frame("Cramér’s V"), use_container_width=True)


elif page == "Procesamiento":
    st.title("Procesamiento (trazabilidad)")
    st.caption("Evidencia MNAR → imputación condicional → validación → transformación + codificación.")

    st.subheader("1) Evaluación de eliminación por nulos (k>=2)")
    if not bundle["comparacion_drop"].empty:
        st.dataframe(bundle["comparacion_drop"], use_container_width=True)
    else:
        st.info("No se pudo construir la comparación (revise target).")

    st.subheader("2) Evidencia de MNAR")
    st.write(f"Kruskal-Wallis p-value: **{bundle['kw_p']}**")
    st.dataframe(bundle["missing_assoc"].head(20).to_frame("Cramér’s V"), use_container_width=True)

    st.subheader("3) Imputación condicional por (Estrato, Departamento)")
    st.write("Variables tratadas:")
    st.code(", ".join(bundle["vars_with_nulls"]))
    st.write("Nulos restantes post-imputación (%):")
    if not bundle["nulls_after"].empty:
        st.dataframe(bundle["nulls_after"].to_frame("nulos_%"), use_container_width=True)
    else:
        st.info("No hay nulos restantes (o variables no presentes).")

    st.subheader("4) Validación de estabilidad (Cramér’s V Antes vs Después)")
    if not bundle["cramers_compare"].empty:
        st.dataframe(bundle["cramers_compare"], use_container_width=True)
    else:
        st.info("No se pudo calcular la validación antes/después.")

    st.subheader("5) Dataset procesado (muestra)")
    st.dataframe(data_imp.head(30), use_container_width=True)


elif page == "KPI":
    st.title("KPI")
    st.caption("Resultados en tarjetas + detalle por KPI (descripción + tabla).")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("P(Nacional) bajo", f"{kpi_pack.get('p_nat', np.nan)*100:.2f}%", "Base nacional")
    with c2:
        kpi_card("KRAS (pp)", f"{kpi_pack.get('KRAS', np.nan)*100:.3f}", "Estrato 1–2 vs 5–6")
    with c3:
        kpi_card("FD (pp)", f"{kpi_pack.get('FD', np.nan)*100:.3f}", "Sin PC vs Con PC")
    with c4:
        kpi_card("KCC (pp)", f"{kpi_pack.get('KCC', np.nan)*100:.3f}", "Madre baja vs alta edu")
    with c5:
        kpi_card("KDS", f"{kpi_pack.get('KDS', np.nan):.4f}", "Promedio Cramér’s V")

    st.markdown("---")

    # KRAS
    st.subheader("KRAS")
    st.markdown(KPI_TEXT["KRAS"])
    if "KRAS_table" in kpi_pack:
        st.dataframe(kpi_pack["KRAS_table"], use_container_width=True)

    # FD
    st.subheader("FD")
    st.markdown(KPI_TEXT["FD"])
    if "FD_table" in kpi_pack:
        st.dataframe(kpi_pack["FD_table"], use_container_width=True)

    # KCC
    st.subheader("KCC")
    st.markdown(KPI_TEXT["KCC"])
    if "KCC_table" in kpi_pack:
        st.dataframe(kpi_pack["KCC_table"], use_container_width=True)

    # KDS
    st.subheader("KDS")
    st.markdown(KPI_TEXT["KDS"])
    if "KDS_table" in kpi_pack and not kpi_pack["KDS_table"].empty:
        st.dataframe(kpi_pack["KDS_table"], use_container_width=True)

    # Equidad regional
    st.subheader("Equidad regional")
    st.markdown(KPI_TEXT["EQ_REG"])
    if "regional_risk" in kpi_pack and not kpi_pack["regional_risk"].empty:
        st.dataframe(kpi_pack["regional_risk"], use_container_width=True)

    # Impacto
    st.subheader("Impacto territorial")
    st.markdown(KPI_TEXT["IMPACTO"])
    if "kpi_dept" in kpi_pack and not kpi_pack["kpi_dept"].empty:
        st.write("Top 15 departamentos por casos")
        st.dataframe(kpi_pack["kpi_dept"].head(15), use_container_width=True)
    if "kpi_region" in kpi_pack and not kpi_pack["kpi_region"].empty:
        st.write("Región (casos)")
        st.dataframe(kpi_pack["kpi_region"], use_container_width=True)


elif page == "EDA":
    st.title("EDA")
    st.caption("Exploración cuantitativa, cualitativa y visual del dataset procesado.")

    tab1, tab2, tab3 = st.tabs(["Cuantitativo", "Cualitativo", "Gráficos"])

    with tab1:
        st.subheader("Distribución del target (%)")
        st.dataframe(target_distribution(data_imp), use_container_width=True)

        st.subheader("Resumen estadístico")
        st.dataframe(quantitative_summary(data_imp), use_container_width=True)

        st.subheader("Pares correlacionados (|corr| > 0.7)")
        st.dataframe(correlation_pairs(data_imp, threshold=0.7), use_container_width=True)

        # Kruskal numéricas vs target (si existen)
        num_cols = [c for c in ["RENDIMIENTO_GLOBAL","INDICADOR_1","INDICADOR_2","INDICADOR_3","INDICADOR_4"] if c in data_imp.columns]
        st.subheader("Kruskal numéricas vs target")
        st.dataframe(kruskal_numeric_vs_target(data_imp, num_cols=num_cols), use_container_width=True)

        st.subheader("Outliers por clase (IQR) — indicadores")
        vars_num = [c for c in ["INDICADOR_1","INDICADOR_2"] if c in data_imp.columns]
        st.dataframe(outliers_by_class(data_imp, vars_num=vars_num), use_container_width=True)

    with tab2:
        st.subheader("Cramér’s V (categóricas vs target)")
        st.dataframe(cramers_table(data_imp), use_container_width=True)

        st.subheader("Perfiles categóricos (Top 15 por riesgo)")
        for col in ["F_ESTRATOVIVIENDA", "F_TIENECOMPUTADOR", "F_TIENEINTERNET", "REGION"]:
            if col in data_imp.columns:
                st.write(f"**{col}**")
                st.dataframe(perfil_categorico(data_imp, col), use_container_width=True)

    with tab3:
        st.subheader("Gráficas")
    
        # Opciones base (se activan según columnas disponibles)
        options = ["Distribución del target"]
    
        TARGET = "RENDIMIENTO_GLOBAL"
        if TARGET in data_imp.columns:
            if "F_ESTRATOVIVIENDA" in data_imp.columns:
                options += ["KRAS (Estrato) - Barras", "KRAS (Estrato) - Gradiente"]
            if "F_TIENECOMPUTADOR" in data_imp.columns:
                options += ["FD (Computador) - Barras", "FD (Computador) - Interacción"]
            if "F_EDUCACIONMADRE" in data_imp.columns:
                options += ["KCC (Educación madre) - Barras", "KCC (Educación madre) - Distribución"]
            if "REGION" in data_imp.columns:
                options += ["Riesgo regional (pérdida)"]
    
            if "kpi_dept" in kpi_pack and not kpi_pack["kpi_dept"].empty:
                options += ["Impacto Top-10 (casos)", "Impacto (burbuja riesgo vs casos)"]
    
            if "kpi_region" in kpi_pack and not kpi_pack["kpi_region"].empty:
                options += ["Impacto por región (casos)"]
    
        choice = st.selectbox("Seleccione la gráfica", options, index=0)
    
        # Render condicional
        if choice == "Distribución del target":
            st.pyplot(charts.fig_target_distribution(data_imp), clear_figure=True)
    
        elif choice == "KRAS (Estrato) - Barras":
            low_str = data_imp["F_ESTRATOVIVIENDA"].isin([1, 2])
            high_str = data_imp["F_ESTRATOVIVIENDA"].isin([5, 6])
            p_low = float((data_imp.loc[low_str, TARGET] == 0).mean()) if low_str.any() else np.nan
            p_high = float((data_imp.loc[high_str, TARGET] == 0).mean()) if high_str.any() else np.nan
            kras = p_low - p_high
            st.pyplot(charts.fig_kras_bar(p_low, p_high, kras), clear_figure=True)
    
        elif choice == "KRAS (Estrato) - Gradiente":
            fig = charts.fig_kras_gradient(data_imp)
            if fig is not None:
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("No se pudo construir la figura de gradiente.")
    
        elif choice == "FD (Computador) - Barras":
            no_pc = (data_imp["F_TIENECOMPUTADOR"] == 0)
            pc = (data_imp["F_TIENECOMPUTADOR"] == 1)
            p_no = float((data_imp.loc[no_pc, TARGET] == 0).mean()) if no_pc.any() else np.nan
            p_pc = float((data_imp.loc[pc, TARGET] == 0).mean()) if pc.any() else np.nan
            fd = p_no - p_pc
            st.pyplot(charts.fig_fd_bar(p_no, p_pc, fd), clear_figure=True)
    
        elif choice == "FD (Computador) - Interacción":
            fig = charts.fig_fd_interaction(data_imp)
            if fig is not None:
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("No se pudo construir la figura de interacción.")
    
        elif choice == "KCC (Educación madre) - Barras":
            low_mom = data_imp["F_EDUCACIONMADRE"].isin(["Ninguno","Primaria incompleta","Primaria completa"])
            high_mom = data_imp["F_EDUCACIONMADRE"].isin(["Educación profesional completa","Postgrado","Posgrado","Postgrado completo"])
            p_low = float((data_imp.loc[low_mom, TARGET] == 0).mean()) if low_mom.any() else np.nan
            p_high = float((data_imp.loc[high_mom, TARGET] == 0).mean()) if high_mom.any() else np.nan
            kcc = p_low - p_high
            st.pyplot(charts.fig_kcc_bar(p_low, p_high, kcc), clear_figure=True)
    
        elif choice == "KCC (Educación madre) - Distribución":
            fig = charts.fig_kcc_mother_education(data_imp)
            if fig is not None:
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("No se pudo construir la figura de educación de la madre.")
    
        elif choice == "Riesgo regional (pérdida)":
            regional_risk = data_imp.groupby("REGION")[TARGET].apply(lambda x: (x == 0).mean()).sort_values()
            st.pyplot(charts.fig_regional_risk(regional_risk), clear_figure=True)
    
        elif choice == "Impacto Top-10 (casos)":
            st.pyplot(charts.fig_top10_impact(kpi_pack["kpi_dept"], top_n=10), clear_figure=True)
    
        elif choice == "Impacto (burbuja riesgo vs casos)":
            st.pyplot(
                charts.fig_impact_risk_bubble(kpi_pack["kpi_dept"], p_nat=kpi_pack.get("p_nat"), top_n=40),
                clear_figure=True
            )
    
        elif choice == "Impacto por región (casos)":
            st.pyplot(charts.fig_region_impact_cases(kpi_pack["kpi_region"]), clear_figure=True)


elif page == "Groq IA":
    st.title("Groq IA")

    try:
        # Consejo: NO pase dataframes completos al LLM si no es necesario
        render_groq_page(bundle=bundle, kpi_pack=kpi_pack)

    except Exception as e:
        st.error("Falló la página de Groq IA. Revise el detalle abajo.")
        st.exception(e)  # muestra traceback completo en la UI
        st.stop()
