
# app_streamlit_mvp.py
# Streamlit MVP: Procesamiento + Imputación + EDA + KPIs + Resultados (tipo "análisis de negocio")
# - EDA con 3 sub-vistas: Cuantitativo, Cualitativo y Gráficos (seleccionables)
# - KPIs y Resultados en formato "tarjeta"
# - Gráficas interactivas con Plotly

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import kruskal
import scipy.stats as ss
import plotly.express as px

st.set_page_config(page_title="MVP Analítica — Saber Pro (Colombia)", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .kpi-card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(250, 250, 252, 0.6);
        height: 100%;
    }
    .kpi-title { font-size: 0.9rem; opacity: 0.75; margin-bottom: 4px; }
    .kpi-value { font-size: 1.55rem; font-weight: 700; margin-bottom: 6px; }
    .kpi-note  { font-size: 0.88rem; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True
)

TARGET_DEFAULT = "RENDIMIENTO_GLOBAL"

def audit(df: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.DataFrame({
            "nulos_%": (df.isna().mean() * 100).round(2),
            "tipo": df.dtypes.astype(str),
            "nulos_n": df.isna().sum()
        })
        .sort_values("nulos_%", ascending=False)
    )

def cramers_v(x, y) -> float:
    confusion = pd.crosstab(x, y)
    if confusion.size == 0:
        return np.nan
    chi2 = ss.chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    r, k = confusion.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan

def impute_conditional(
    data: pd.DataFrame,
    group_vars=("F_ESTRATOVIVIENDA", "E_PRGM_DEPARTAMENTO"),
    vars_with_nulls=(
        "F_TIENEAUTOMOVIL","F_TIENELAVADORA","F_TIENECOMPUTADOR","E_HORASSEMANATRABAJA","F_TIENEINTERNET",
        "F_EDUCACIONMADRE","F_EDUCACIONPADRE","E_PAGOMATRICULAPROPIO","E_VALORMATRICULAUNIVERSIDAD",
    ),
) -> pd.DataFrame:
    data_imp = data.copy()

    if "F_TIENEINTERNET.1" in data_imp.columns:
        data_imp = data_imp.drop(columns=["F_TIENEINTERNET.1"])

    for col in group_vars:
        if col in data_imp.columns:
            data_imp[col] = data_imp[col].fillna("Desconocido")

    for col in vars_with_nulls:
        if col not in data_imp.columns:
            continue
        if pd.api.types.is_numeric_dtype(data_imp[col]):
            data_imp[col] = data_imp.groupby(list(group_vars))[col].transform(lambda x: x.fillna(x.median()))
        else:
            data_imp[col] = data_imp.groupby(list(group_vars))[col].transform(
                lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else "Desconocido")
            )
    return data_imp

def compute_missingness_patterns(df: pd.DataFrame, target: str) -> dict:
    out = {}
    work = df.copy()
    work["N_NULOS"] = work.isna().sum(axis=1)

    if target in work.columns:
        out["nulls_by_class_desc"] = work.groupby(target)["N_NULOS"].describe()

        groups = [work.loc[work[target] == c, "N_NULOS"] for c in work[target].dropna().unique()]
        if len(groups) >= 2:
            _, p = kruskal(*groups)
            out["kruskal_p"] = float(p)
        else:
            out["kruskal_p"] = np.nan

        work["MISSING_HEAVY"] = (work["N_NULOS"] >= 2).astype(int)
        candidate_cols = [
            "F_ESTRATOVIVIENDA","F_TIENECOMPUTADOR","F_TIENEINTERNET","F_TIENEAUTOMOVIL",
            "E_EDAD","E_PRGM_DEPARTAMENTO","E_EDUCACIONPADRE","E_EDUCACIONMADRE",
        ]
        assoc = {}
        for col in candidate_cols:
            if col in work.columns:
                assoc[col] = cramers_v(work[col], work["MISSING_HEAVY"])
        out["missing_assoc"] = pd.Series(assoc).sort_values(ascending=False)
    return out

def compare_cramers_before_after(data: pd.DataFrame, data_imp: pd.DataFrame, target: str, cols: list[str]) -> pd.DataFrame:
    before, after = {}, {}
    for col in cols:
        if col in data.columns and col in data_imp.columns and target in data.columns:
            before[col] = cramers_v(data[col], data[target])
            after[col] = cramers_v(data_imp[col], data_imp[target])
    comp = pd.DataFrame({"Antes": before, "Después": after})
    comp["Cambio"] = comp["Después"] - comp["Antes"]
    return comp.sort_values("Cambio", ascending=False)

def fig_rendimiento_dist(df: pd.DataFrame, target: str):
    if target not in df.columns:
        return None
    t = df[target].value_counts(dropna=False).rename_axis("clase").reset_index(name="n")
    t["pct"] = t["n"] / t["n"].sum()
    fig = px.bar(t, x="clase", y="n", text=t["pct"].map(lambda v: f"{v:.1%}"))
    fig.update_traces(textposition="outside")
    fig.update_layout(title="Distribución del rendimiento académico", xaxis_title="Nivel de rendimiento", yaxis_title="Número de estudiantes", height=420)
    return fig

def fig_kras_brecha(df: pd.DataFrame, target: str, estrato_col="F_ESTRATOVIVIENDA", low_class=0):
    if target not in df.columns or estrato_col not in df.columns:
        return None, None
    work = df[[target, estrato_col]].copy()
    low = work[work[estrato_col].isin([1, 2])]
    high = work[work[estrato_col].isin([5, 6])]
    if len(low) == 0 or len(high) == 0:
        return None, None
    p_low = (low[target] == low_class).mean()
    p_high = (high[target] == low_class).mean()
    kras_pp = (p_low - p_high) * 100
    t = pd.DataFrame({"segmento": ["Estrato 1–2", "Estrato 5–6"], "p_bajo": [p_low, p_high]})
    fig = px.bar(t, x="segmento", y="p_bajo", text=t["p_bajo"].map(lambda v: f"{v*100:.1f}%"))
    fig.update_traces(textposition="inside")
    fig.update_layout(title=f"Brecha socioeconómica (KRAS) — brecha = {kras_pp:.1f} pp", yaxis_tickformat=".0%", yaxis_title="Proporción de estudiantes en bajo rendimiento", height=420)
    return fig, {"p_low": p_low, "p_high": p_high, "kras_pp": kras_pp}

def fig_kcc_brecha(df: pd.DataFrame, target: str, mom_col="F_EDUCACIONMADRE", low_class=0):
    if target not in df.columns or mom_col not in df.columns:
        return None, None
    work = df[[target, mom_col]].dropna(subset=[mom_col]).copy()
    work["BAJO"] = (work[target] == low_class).astype(int)
    t = work.groupby(mom_col)["BAJO"].agg(p_bajo="mean", N="size").sort_values("p_bajo", ascending=False)
    if len(t) < 2:
        return None, None
    worst, best = t.index[0], t.index[-1]
    p_worst, p_best = float(t.iloc[0]["p_bajo"]), float(t.iloc[-1]["p_bajo"])
    kcc_pp = (p_worst - p_best) * 100
    comp = pd.DataFrame({"segmento": [f"Madre: {worst}", f"Madre: {best}"], "p_bajo": [p_worst, p_best]})
    fig = px.bar(comp, x="segmento", y="p_bajo", text=comp["p_bajo"].map(lambda v: f"{v*100:.1f}%"))
    fig.update_traces(textposition="inside")
    fig.update_layout(title=f"Capital cultural materno (KCC) — brecha = {kcc_pp:.1f} pp", yaxis_tickformat=".0%", yaxis_title="Proporción de estudiantes en bajo rendimiento", height=420)
    return fig, {"worst": worst, "best": best, "kcc_pp": kcc_pp}

def fig_riesgo_regional(df: pd.DataFrame, target: str, region_col="REGION", low_class=0):
    if target not in df.columns or region_col not in df.columns:
        return None, None
    work = df[[target, region_col]].copy()
    work["BAJO"] = (work[target] == low_class).astype(int)
    t = work.groupby(region_col)["BAJO"].mean().sort_values(ascending=False).rename("p_bajo").reset_index()
    fig = px.bar(t, x=region_col, y="p_bajo", text=t["p_bajo"].map(lambda v: f"{v*100:.1f}%"))
    fig.update_layout(title="Riesgo de bajo rendimiento por región", yaxis_tickformat=".0%", yaxis_title="Proporción de bajo rendimiento", xaxis_title="Región", height=420)
    return fig, t

def fig_riesgo_dept(df: pd.DataFrame, target: str, dept_col="E_PRGM_DEPARTAMENTO", low_class=0, top_n=25):
    if target not in df.columns or dept_col not in df.columns:
        return None, None
    work = df[[target, dept_col]].copy()
    work["BAJO"] = (work[target] == low_class).astype(int)
    t = work.groupby(dept_col)["BAJO"].mean().sort_values(ascending=False).rename("p_bajo").reset_index()
    t2 = t.head(top_n)
    fig = px.bar(t2, x=dept_col, y="p_bajo", text=t2["p_bajo"].map(lambda v: f"{v*100:.1f}%"))
    fig.update_layout(title=f"Top {top_n} departamentos con mayor riesgo de bajo rendimiento", yaxis_tickformat=".0%", yaxis_title="Proporción de bajo rendimiento", xaxis_title="Departamento", height=420)
    fig.update_xaxes(tickangle=45)
    return fig, t

def fig_cramers_delta(comp: pd.DataFrame):
    if comp is None or comp.empty:
        return None
    t = comp.reset_index(names="variable").sort_values("Cambio", ascending=False)
    fig = px.bar(t, x="variable", y="Cambio", hover_data=["Antes", "Después"])
    fig.update_layout(title="Validación de imputación: cambio en Cramér’s V (Después - Antes)", xaxis_title="Variable imputada", yaxis_title="Δ Cramér’s V", height=420)
    fig.update_xaxes(tickangle=45)
    return fig

def kpi_card(title: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

st.sidebar.header("📦 Datos")
uploaded = st.sidebar.file_uploader("Cargar CSV (ej. saber_pro.csv)", type=["csv"])
target_col = st.sidebar.text_input("Columna target", value=TARGET_DEFAULT)

st.sidebar.divider()
page = st.sidebar.radio("Navegación", ["📌 Resumen", "🧹 Procesamiento", "🔎 EDA", "📈 KPI (negocio)", "🏁 Resultados"], index=0)

if not uploaded:
    st.info("Cargue un CSV para iniciar el MVP.")
    st.stop()

data = load_csv(uploaded)
if target_col not in data.columns:
    st.error(f"No se encontró la columna target '{target_col}'. Revise el nombre.")
    st.stop()

@st.cache_data(show_spinner=False)
def pipeline(data: pd.DataFrame, target: str):
    patterns = compute_missingness_patterns(data, target)
    data_imp = impute_conditional(data)
    cols_imputadas = [
        "F_TIENEAUTOMOVIL","F_TIENELAVADORA","F_TIENECOMPUTADOR","E_HORASSEMANATRABAJA","F_TIENEINTERNET",
        "F_EDUCACIONMADRE","F_EDUCACIONPADRE","E_PAGOMATRICULAPROPIO","E_VALORMATRICULAUNIVERSIDAD"
    ]
    comp = compare_cramers_before_after(data, data_imp, target, cols_imputadas)
    return patterns, data_imp, comp

patterns, data_imp, cramers_comp = pipeline(data, target_col)

if page == "📌 Resumen":
    st.title("📌 Resumen ejecutivo")
    st.caption("MVP orientado a cliente: transparencia de procesamiento, imputación y hallazgos accionables.")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Registros", f"{len(data):,}".replace(",", "."), "Tamaño del dataset original")
    with c2: kpi_card("Columnas", f"{data.shape[1]:,}".replace(",", "."), "Variables disponibles")
    with c3:
        null_rate = float(data.isna().mean().mean())
        kpi_card("Nulos (promedio)", f"{null_rate*100:.2f}%", "Promedio de nulos por variable")
    with c4:
        classes = data[target_col].nunique(dropna=True)
        kpi_card("Clases target", f"{classes}", f"En {target_col}")

    st.subheader("Distribución del target (vista rápida)")
    fig = fig_rendimiento_dist(data_imp, target_col)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

elif page == "🧹 Procesamiento":
    st.title("🧹 Procesamiento e imputación")

    st.subheader("1) Auditoría de nulos")
    st.dataframe(audit(data), use_container_width=True, height=420)

    st.subheader("2) Patrones de missingness (evidencia)")
    p = patterns.get("kruskal_p", np.nan)
    if pd.notna(p):
        st.write(f"**Kruskal–Wallis p-value (N_NULOS vs clase de rendimiento):** {p:.4g}")
        st.caption("p≈0 sugiere missingness dependiente del rendimiento (no MCAR).")

    assoc = patterns.get("missing_assoc")
    if isinstance(assoc, pd.Series) and not assoc.empty:
        st.write("**Asociación (Cramér’s V) con MISSING_HEAVY (≥2 nulos):**")
        st.dataframe(assoc.rename("Cramér’s V").to_frame(), use_container_width=True, height=240)

    st.subheader("3) Imputación condicional (por grupo)")
    st.caption("Mediana para numéricas / Moda para categóricas. Grupos: estrato + departamento.")

    cols = [c for c in [
        "F_TIENEAUTOMOVIL","F_TIENELAVADORA","F_TIENECOMPUTADOR","E_HORASSEMANATRABAJA","F_TIENEINTERNET",
        "F_EDUCACIONMADRE","F_EDUCACIONPADRE","E_PAGOMATRICULAPROPIO","E_VALORMATRICULAUNIVERSIDAD"
    ] if c in data.columns]

    col1, col2 = st.columns(2)
    with col1:
        st.write("**% nulos (antes)** en variables tratadas")
        before = (data[cols].isna().mean()*100).round(2).to_frame("nulos_%")
        st.dataframe(before, use_container_width=True, height=240)
    with col2:
        st.write("**% nulos (después)** en variables tratadas")
        after = (data_imp[cols].isna().mean()*100).round(2).to_frame("nulos_%")
        st.dataframe(after, use_container_width=True, height=240)

    st.subheader("4) Validación: preservación de estructura (Cramér’s V)")
    fig = fig_cramers_delta(cramers_comp)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cramers_comp, use_container_width=True, height=260)

elif page == "🔎 EDA":
    st.title("🔎 EDA")
    tab_q, tab_c, tab_g = st.tabs(["📐 Cuantitativo", "🔤 Cualitativo", "🖼️ Gráficos (selección)"])

    with tab_q:
        st.subheader("Resumen cuantitativo")
        num_cols = data_imp.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.info("No se detectaron variables numéricas.")
        else:
            sel = st.multiselect("Variables numéricas", num_cols, default=num_cols[:6])
            if sel:
                st.dataframe(data_imp[sel].describe().T, use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    x = st.selectbox("Histograma: variable", sel, index=0)
                    fig = px.histogram(data_imp, x=x, nbins=40, marginal="box")
                    fig.update_layout(title=f"Distribución de {x}", height=420)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    y = st.selectbox("Boxplot por target: variable", sel, index=min(1, len(sel)-1))
                    fig = px.box(data_imp, x=target_col, y=y, points="outliers")
                    fig.update_layout(title=f"{y} vs {target_col}", height=420)
                    st.plotly_chart(fig, use_container_width=True)

    with tab_c:
        st.subheader("Resumen cualitativo")
        cat_cols = data_imp.select_dtypes(exclude=[np.number]).columns.tolist()
        cat_cols = [c for c in cat_cols if data_imp[c].nunique(dropna=True) <= 80]
        if not cat_cols:
            st.info("No se detectaron variables categóricas con cardinalidad moderada (<=80).")
        else:
            col = st.selectbox("Variable categórica", cat_cols, index=0)
            topn = st.slider("Top N categorías", 5, 30, 15)
            t = data_imp[col].value_counts(dropna=False).head(topn).rename_axis(col).reset_index(name="n")
            t["pct"] = t["n"] / t["n"].sum()
            fig = px.bar(t, x=col, y="n", text=t["pct"].map(lambda v: f"{v:.1%}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(title=f"Top {topn} categorías — {col}", height=420)
            st.plotly_chart(fig, use_container_width=True)

            t2 = data_imp[[col, target_col]].dropna(subset=[col, target_col]).groupby([col, target_col]).size().reset_index(name="n")
            if not t2.empty:
                fig2 = px.bar(t2, x=col, y="n", color=target_col, barmode="stack")
                fig2.update_layout(title=f"{col} segmentado por {target_col}", height=420)
                st.plotly_chart(fig2, use_container_width=True)

    with tab_g:
        st.subheader("Galería de gráficos (seleccione cuáles mostrar)")
        catalog = {
            "Distribución del rendimiento académico": lambda: fig_rendimiento_dist(data_imp, target_col),
            "Riesgo de bajo rendimiento por región": lambda: fig_riesgo_regional(data_imp, target_col)[0],
            "Top departamentos por riesgo": lambda: fig_riesgo_dept(data_imp, target_col, top_n=25)[0],
            "Brecha socioeconómica (KRAS)": lambda: fig_kras_brecha(data_imp, target_col)[0],
            "Capital cultural materno (KCC)": lambda: fig_kcc_brecha(data_imp, target_col)[0],
            "Δ Cramér’s V (validación imputación)": lambda: fig_cramers_delta(cramers_comp),
        }

        picks = st.multiselect("Gráficos disponibles", list(catalog.keys()), default=[
            "Distribución del rendimiento académico",
            "Riesgo de bajo rendimiento por región",
            "Brecha socioeconómica (KRAS)",
        ])

        for name in picks:
            fig = catalog[name]()
            if fig is None:
                st.warning(f"No se pudo construir el gráfico: {name} (faltan columnas en el CSV).")
            else:
                st.plotly_chart(fig, use_container_width=True)

elif page == "📈 KPI (negocio)":
    st.title("📈 KPIs (visión de negocio)")
    st.caption("KPIs orientados a decisión: brechas, riesgo territorial y segmentos críticos.")

    fig_reg, t_reg = fig_riesgo_regional(data_imp, target_col)
    fig_dept, t_dept = fig_riesgo_dept(data_imp, target_col, top_n=15)

    fig_kras, k_kras = fig_kras_brecha(data_imp, target_col)
    fig_kcc, k_kcc = fig_kcc_brecha(data_imp, target_col)

    row1 = st.columns(4)
    with row1[0]:
        kpi_card("Riesgo nacional", f"{(data_imp[target_col]==0).mean()*100:.1f}%", "Proporción en clase de bajo rendimiento (low_class=0)")
    with row1[1]:
        kpi_card("Brecha KRAS", f"{k_kras['kras_pp']:.1f} pp" if k_kras else "N/D", "Estrato 1–2 vs 5–6")
    with row1[2]:
        kpi_card("Brecha KCC", f"{k_kcc['kcc_pp']:.1f} pp" if k_kcc else "N/D", "Extremos por educación materna")
    with row1[3]:
        if t_reg is not None and not t_reg.empty:
            best = t_reg.sort_values("p_bajo").iloc[-1]
            worst = t_reg.sort_values("p_bajo").iloc[0]
            gap = (worst["p_bajo"] - best["p_bajo"]) * 100
            kpi_card("Brecha regional", f"{gap:.1f} pp", f"Mejor: {best.iloc[0]} | Peor: {worst.iloc[0]}")
        else:
            kpi_card("Brecha regional", "N/D", "Falta columna REGION")

    st.subheader("KPIs visuales (interactivos)")
    c1, c2 = st.columns(2)
    with c1:
        if fig_kras: st.plotly_chart(fig_kras, use_container_width=True)
        if fig_reg: st.plotly_chart(fig_reg, use_container_width=True)
    with c2:
        if fig_kcc: st.plotly_chart(fig_kcc, use_container_width=True)
        if fig_dept: st.plotly_chart(fig_dept, use_container_width=True)

elif page == "🏁 Resultados":
    st.title("🏁 Resultados (en tarjetas)")
    st.info("Este apartado está listo para conectar el modelo (p. ej., Random Forest) y mostrar métricas + explicaciones.")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Clase más frecuente", str(data_imp[target_col].mode(dropna=True).iloc[0]), "Modo del target")
    with c2: kpi_card("Nulos post-imputación", f"{data_imp.isna().mean().mean()*100:.2f}%", "Promedio de nulos (global)")
    with c3: kpi_card("Variables imputadas", "9", "Lista base (según script)")
    with c4: kpi_card("Validación (máx |Δ Cramér’s V|)", f"{(cramers_comp['Cambio'].abs().max() if not cramers_comp.empty else 0):.4f}", "Antes vs después")

