# app.py
# streamlit run app.py

import numpy as np
import pandas as pd
import streamlit as st
import scipy.stats as ss
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.express as px

from mappings import grupo_dict_programas, grupo_dict_regiones


# ----------------------------
# Configuración UI
# ----------------------------
st.set_page_config(page_title="Saber Pro - Dashboard", layout="wide")

CSS = """
<style>
.kpi-card {
  background: #ffffff;
  border: 1px solid #e8e8e8;
  border-radius: 14px;
  padding: 16px 16px 12px 16px;
  box-shadow: 0 1px 10px rgba(0,0,0,0.04);
  height: 100%;
}
.kpi-title { font-size: 12px; color: #666; margin-bottom: 8px; }
.kpi-value { font-size: 28px; font-weight: 700; margin: 0; }
.kpi-sub { font-size: 12px; color: #888; margin-top: 6px; }
.section-title { margin-top: 6px; }
hr { margin: 0.6rem 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------
# Utilidades (estadística / limpieza)
# ----------------------------
def audit(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "nulos_%": (df.isna().mean() * 100).round(2),
        "tipo": df.dtypes.astype(str),
        "nulos_n": df.isna().sum()
    }).sort_values("nulos_%", ascending=False)
    return out

def drop_rows_with_many_nulls(df: pd.DataFrame, k: int = 2):
    null_count = df.isna().sum(axis=1)
    mask_drop = null_count >= k

    cols_nulas = (
        df[mask_drop]
        .isna()
        .apply(lambda row: list(df.columns[row]), axis=1)
    )

    df_eliminados = df[mask_drop].copy()
    df_eliminados["Columnas_Nulas"] = cols_nulas
    df_eliminados["N_Nulos"] = null_count[mask_drop]

    df_limpio = df[~mask_drop].copy()
    return df_limpio, df_eliminados

def cramers_v(x, y) -> float:
    confusion = pd.crosstab(x, y)
    if confusion.size == 0:
        return np.nan
    chi2 = ss.chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    r, k = confusion.shape
    if min(r, k) <= 1:
        return np.nan
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def map_to_group_program(programa: str) -> str:
    for group, programas in grupo_dict_programas.items():
        if programa in programas:
            return group
    return "OTROS"

def map_to_group_region(departamento: str) -> str:
    for group, dptos in grupo_dict_regiones.items():
        if departamento in dptos:
            return group
    return "OTROS"


# ----------------------------
# Pipeline principal (cacheado)
# ----------------------------
@st.cache_data(show_spinner=True)
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_data(show_spinner=True)
def build_dataset(data: pd.DataFrame):
    """
    Retorna:
      - data_raw (original)
      - data_imp (procesado/imputado/codificado)
      - artefactos útiles: audit_df, comparacion_drop, missing_tests, cramers_before_after, etc.
    """
    data_raw = data.copy()
    target = "RENDIMIENTO_GLOBAL"

    # Auditoría inicial
    audit_df = audit(data_raw)

    # --- Evaluación de eliminación por nulos (transparencia)
    df_limpio, df_eliminados = drop_rows_with_many_nulls(data_raw.copy(), k=2)

    dist_elim = df_eliminados[target].value_counts(normalize=True).mul(100).round(2) if target in df_eliminados.columns else pd.Series(dtype=float)
    dist_total = data_raw[target].value_counts(normalize=True).mul(100).round(2) if target in data_raw.columns else pd.Series(dtype=float)

    comparacion_drop = pd.concat([dist_elim, dist_total], axis=1, keys=["Eliminados %", "Total %"])
    if not comparacion_drop.empty:
        comparacion_drop["Diferencia_abs_pp"] = (comparacion_drop["Eliminados %"] - comparacion_drop["Total %"]).abs()
        comparacion_drop = comparacion_drop.sort_values("Diferencia_abs_pp", ascending=False)

    # --- Missingness patterns
    df_miss = data_raw.copy()
    df_miss["N_NULOS"] = df_miss.isna().sum(axis=1)

    kw_p_value = np.nan
    if target in df_miss.columns:
        groups = [df_miss[df_miss[target] == c]["N_NULOS"] for c in df_miss[target].dropna().unique()]
        if len(groups) >= 2:
            kw_stat, kw_p_value = kruskal(*groups)

    df_miss["MISSING_HEAVY"] = (df_miss["N_NULOS"] >= 2).astype(int)

    candidate_cols = [
        "F_ESTRATOVIVIENDA", "F_TIENECOMPUTADOR", "F_TIENEINTERNET", "F_TIENEAUTOMOVIL",
        "E_EDAD", "E_PRGM_DEPARTAMENTO", "E_EDUCACIONPADRE", "E_EDUCACIONMADRE"
    ]

    missing_assoc = {}
    for col in candidate_cols:
        if col in df_miss.columns:
            try:
                missing_assoc[col] = cramers_v(df_miss[col], df_miss["MISSING_HEAVY"])
            except Exception:
                missing_assoc[col] = np.nan
    missing_assoc_s = pd.Series(missing_assoc).sort_values(ascending=False)

    # --- Imputación condicional
    data_imp = data_raw.copy()
    if "F_TIENEINTERNET.1" in data_imp.columns:
        data_imp = data_imp.drop(columns=["F_TIENEINTERNET.1"])

    group_vars = ["F_ESTRATOVIVIENDA", "E_PRGM_DEPARTAMENTO"]
    for col in group_vars:
        if col in data_imp.columns:
            data_imp[col] = data_imp[col].fillna("Desconocido")

    vars_with_nulls = [
        "F_TIENEAUTOMOVIL", "F_TIENELAVADORA", "F_TIENECOMPUTADOR", "E_HORASSEMANATRABAJA",
        "F_TIENEINTERNET", "F_EDUCACIONMADRE", "F_EDUCACIONPADRE", "E_PAGOMATRICULAPROPIO",
        "E_VALORMATRICULAUNIVERSIDAD"
    ]

    for col in vars_with_nulls:
        if col not in data_imp.columns:
            continue
        if pd.api.types.is_numeric_dtype(data_imp[col]):
            data_imp[col] = data_imp.groupby(group_vars)[col].transform(lambda x: x.fillna(x.median()))
        else:
            data_imp[col] = data_imp.groupby(group_vars)[col].transform(
                lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else "Desconocido")
            )

    nulls_after = (data_imp[vars_with_nulls].isna().mean() * 100).round(2) if all([c in data_imp.columns for c in vars_with_nulls if c in data_imp.columns]) else pd.Series(dtype=float)

    # --- Validación Cramér's V antes/después (vs target)
    cramers_before, cramers_after = {}, {}
    if target in data_raw.columns:
        for col in vars_with_nulls:
            if col in data_raw.columns and col in data_imp.columns:
                try:
                    cramers_before[col] = cramers_v(data_raw[col], data_raw[target])
                    cramers_after[col] = cramers_v(data_imp[col], data_imp[target])
                except Exception:
                    pass
    cramers_compare = pd.DataFrame({"Antes": cramers_before, "Después": cramers_after})
    if not cramers_compare.empty:
        cramers_compare["Cambio"] = cramers_compare["Después"] - cramers_compare["Antes"]
        cramers_compare = cramers_compare.sort_values("Cambio", ascending=False)

    # --- Reducción de cardinalidad (programas) + región
    if "E_PRGM_ACADEMICO" in data_imp.columns:
        data_imp["E_PRGM_ACADEMICO"] = data_imp["E_PRGM_ACADEMICO"].apply(map_to_group_program)

    if "E_PRGM_DEPARTAMENTO" in data_imp.columns:
        data_imp["REGION"] = data_imp["E_PRGM_DEPARTAMENTO"].apply(map_to_group_region)

    # --- Codificación (para KPIs)
    if "F_ESTRATOVIVIENDA" in data_imp.columns:
        estrato_map = {
            "Sin Estrato": 0, "Desconocido": 0,
            "Estrato 1": 1, "Estrato 2": 2, "Estrato 3": 3,
            "Estrato 4": 4, "Estrato 5": 5, "Estrato 6": 6
        }
        # Ojo: si ya viene numérico, esto lo dejará en 0 si no matchea string; se protege:
        if data_imp["F_ESTRATOVIVIENDA"].dtype == "object":
            data_imp["F_ESTRATOVIVIENDA"] = data_imp["F_ESTRATOVIVIENDA"].map(estrato_map).fillna(0).astype(int)
        else:
            data_imp["F_ESTRATOVIVIENDA"] = data_imp["F_ESTRATOVIVIENDA"].fillna(0).astype(int)

    binary_cols = [
        "F_TIENEINTERNET", "F_TIENELAVADORA", "F_TIENEAUTOMOVIL", "F_TIENECOMPUTADOR",
        "E_PRIVADO_LIBERTAD", "E_PAGOMATRICULAPROPIO"
    ]
    for col in binary_cols:
        if col in data_imp.columns and data_imp[col].dtype == "object":
            data_imp[col] = data_imp[col].map({"Si": 1, "Sí": 1, "No": 0}).fillna(0).astype(int)

    if target in data_imp.columns and data_imp[target].dtype == "object":
        target_map = {"bajo": 0, "medio-bajo": 1, "medio-alto": 2, "alto": 3}
        data_imp[target] = data_imp[target].map(target_map)

    # --- EDA num_cols sugeridas
    num_cols = [c for c in ["RENDIMIENTO_GLOBAL", "INDICADOR_1", "INDICADOR_2", "INDICADOR_3", "INDICADOR_4"] if c in data_imp.columns]
    cat_cols = data_imp.select_dtypes(include=["object"]).columns.tolist()

    # --- Reglas de eliminación por redundancia (como su notebook)
    cols_to_drop = [c for c in ["INDICADOR_3", "INDICADOR_4"] if c in data_imp.columns]
    data_imp = data_imp.drop(columns=cols_to_drop)

    return {
        "data_raw": data_raw,
        "data_imp": data_imp,
        "audit_df": audit_df,
        "comparacion_drop": comparacion_drop,
        "kw_p_value": kw_p_value,
        "missing_assoc": missing_assoc_s,
        "nulls_after": nulls_after,
        "cramers_compare": cramers_compare,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cols_dropped": cols_to_drop,
    }


# ----------------------------
# KPIs
# ----------------------------
def compute_kpis(data_imp: pd.DataFrame):
    TARGET = "RENDIMIENTO_GLOBAL"
    if TARGET not in data_imp.columns:
        return {}

    is_low = (data_imp[TARGET] == 0)
    p_nat = float(is_low.mean()) if len(data_imp) else np.nan

    # KRAS
    KRAS = np.nan
    p_low_str, p_high_str = np.nan, np.nan
    if "F_ESTRATOVIVIENDA" in data_imp.columns:
        low_str = data_imp["F_ESTRATOVIVIENDA"].isin([1, 2])
        high_str = data_imp["F_ESTRATOVIVIENDA"].isin([5, 6])
        p_low_str = float((data_imp.loc[low_str, TARGET] == 0).mean()) if low_str.any() else np.nan
        p_high_str = float((data_imp.loc[high_str, TARGET] == 0).mean()) if high_str.any() else np.nan
        KRAS = p_low_str - p_high_str

    # FD
    FD = np.nan
    p_no_pc, p_yes_pc = np.nan, np.nan
    if "F_TIENECOMPUTADOR" in data_imp.columns:
        no_pc = (data_imp["F_TIENECOMPUTADOR"] == 0)
        yes_pc = (data_imp["F_TIENECOMPUTADOR"] == 1)
        p_no_pc = float((data_imp.loc[no_pc, TARGET] == 0).mean()) if no_pc.any() else np.nan
        p_yes_pc = float((data_imp.loc[yes_pc, TARGET] == 0).mean()) if yes_pc.any() else np.nan
        FD = p_no_pc - p_yes_pc

    # KCC
    KCC = np.nan
    p_low_mom, p_high_mom = np.nan, np.nan
    if "F_EDUCACIONMADRE" in data_imp.columns:
        low_mom = data_imp["F_EDUCACIONMADRE"].isin(["Ninguno", "Primaria incompleta", "Primaria completa"])
        high_mom = data_imp["F_EDUCACIONMADRE"].isin(["Educación profesional completa", "Postgrado", "Posgrado", "Postgrado completo"])
        p_low_mom = float((data_imp.loc[low_mom, TARGET] == 0).mean()) if low_mom.any() else np.nan
        p_high_mom = float((data_imp.loc[high_mom, TARGET] == 0).mean()) if high_mom.any() else np.nan
        KCC = p_low_mom - p_high_mom

    # KDS (promedio de Cramér’s V con variables socioeconómicas)
    socio_cols = [c for c in [
        "F_ESTRATOVIVIENDA", "F_TIENECOMPUTADOR", "F_TIENEINTERNET", "F_TIENEAUTOMOVIL",
        "F_EDUCACIONMADRE", "F_EDUCACIONPADRE", "REGION", "E_PRGM_DEPARTAMENTO"
    ] if c in data_imp.columns]
    cramers_vals = []
    for c in socio_cols:
        try:
            v = cramers_v(data_imp[c], data_imp[TARGET])
            if not np.isnan(v):
                cramers_vals.append(v)
        except Exception:
            pass
    KDS = float(np.mean(cramers_vals)) if cramers_vals else np.nan

    # Equidad regional (brecha)
    equidad_regional_pp = np.nan
    regional_risk = pd.Series(dtype=float)
    if "REGION" in data_imp.columns:
        regional_risk = (
            data_imp.groupby("REGION")[TARGET]
            .apply(lambda x: (x == 0).mean())
            .sort_values()
        )
        if len(regional_risk) >= 2:
            equidad_regional_pp = float((regional_risk.max() - regional_risk.min()) * 100)

    # KPI dept / región (impacto)
    kpi_dept = pd.DataFrame()
    kpi_region = pd.DataFrame()
    if "E_PRGM_DEPARTAMENTO" in data_imp.columns:
        kpi_dept = (
            data_imp.assign(BAJO=is_low.astype(int))
            .groupby("E_PRGM_DEPARTAMENTO")
            .agg(N=("BAJO", "size"), p_bajo=("BAJO", "mean"), casos=("BAJO", "sum"))
            .sort_values("casos", ascending=False)
        )
        kpi_dept["exceso_pp"] = (kpi_dept["p_bajo"] - p_nat) * 100
        kpi_dept["exceso_casos"] = (kpi_dept["p_bajo"] - p_nat) * kpi_dept["N"]

    if "REGION" in data_imp.columns:
        kpi_region = (
            data_imp.assign(BAJO=is_low.astype(int))
            .groupby("REGION")
            .agg(N=("BAJO", "size"), p_bajo=("BAJO", "mean"), casos=("BAJO", "sum"))
            .sort_values("casos", ascending=False)
        )

    return {
        "p_nat": p_nat,
        "KRAS": KRAS, "p_low_str": p_low_str, "p_high_str": p_high_str,
        "FD": FD, "p_no_pc": p_no_pc, "p_yes_pc": p_yes_pc,
        "KCC": KCC, "p_low_mom": p_low_mom, "p_high_mom": p_high_mom,
        "KDS": KDS,
        "regional_risk": regional_risk,
        "equidad_regional_pp": equidad_regional_pp,
        "kpi_dept": kpi_dept,
        "kpi_region": kpi_region,
    }


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


# ----------------------------
# Gráficos (Plotly, selector de tipo)
# ----------------------------
def chart_selector(kind: str, df: pd.DataFrame, x: str, y: str | None = None, title: str = ""):
    if kind == "Barras":
        fig = px.bar(df, x=x, y=y, title=title)
    elif kind == "Líneas":
        fig = px.line(df, x=x, y=y, title=title)
    elif kind == "Boxplot":
        fig = px.box(df, x=x, y=y, title=title)
    elif kind == "Histograma":
        fig = px.histogram(df, x=x, title=title)
    elif kind == "Heatmap (correlación)":
        corr = df.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr, title=title)
    else:
        fig = px.scatter(df, x=x, y=y, title=title)
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Sidebar: carga + navegación
# ----------------------------
st.sidebar.title("Menú")
uploaded = st.sidebar.file_uploader("Subir saber_pro.csv", type=["csv"])

page = st.sidebar.selectbox(
    "Navegación",
    ["Resumen", "Procesamiento", "KPI", "EDA"],
    index=0
)

if not uploaded:
    st.title("Dashboard Saber Pro")
    st.info("Cargue el archivo **saber_pro.csv** para comenzar.")
    st.stop()

data = load_data(uploaded)
bundle = build_dataset(data)
data_raw = bundle["data_raw"]
data_imp = bundle["data_imp"]
kpis = compute_kpis(data_imp)


# ----------------------------
# Páginas
# ----------------------------
if page == "Resumen":
    st.title("Resumen")
    st.caption("Visión ejecutiva: estado del dataset, señales clave y puntos de decisión.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Filas", f"{len(data_raw):,}".replace(",", "."), "Dataset original")
    with c2:
        kpi_card("Columnas", f"{data_raw.shape[1]:,}".replace(",", "."), "Dataset original")
    with c3:
        nulos_pct = float(data_raw.isna().mean().mean() * 100)
        kpi_card("Nulos promedio", f"{nulos_pct:.2f}%", "Promedio global (todas las columnas)")
    with c4:
        kw = bundle["kw_p_value"]
        kpi_card("Kruskal (Missingness)", "p≈0" if (isinstance(kw, float) and kw < 1e-6) else f"p={kw:.4g}", "N_NULOS vs RENDIMIENTO")

    st.subheader("Muestra del dataset")
    st.dataframe(data_raw.head(20), use_container_width=True)

    st.subheader("Auditoría de nulos (top 25)")
    st.dataframe(bundle["audit_df"].head(25), use_container_width=True)

    st.subheader("Asociación con missingness (Cramér’s V)")
    st.dataframe(bundle["missing_assoc"].to_frame("Cramér’s V").head(10), use_container_width=True)


elif page == "Procesamiento":
    st.title("Procesamiento")
    st.caption("Transparencia: decisiones, imputación condicional y validación de estabilidad.")

    st.subheader("1) ¿Eliminar filas con ≥2 nulos?")
    st.write("Comparación de distribución del target entre eliminados vs total (si el target existe).")
    if not bundle["comparacion_drop"].empty:
        st.dataframe(bundle["comparacion_drop"], use_container_width=True)
    else:
        st.info("No fue posible construir la comparación (revise que exista RENDIMIENTO_GLOBAL).")

    st.subheader("2) Imputación condicional por (Estrato, Departamento)")
    st.write("Porcentaje de nulos restantes en variables imputadas (post-tratamiento):")
    if isinstance(bundle["nulls_after"], pd.Series) and not bundle["nulls_after"].empty:
        st.dataframe(bundle["nulls_after"].to_frame("nulos_%").sort_values("nulos_%", ascending=False), use_container_width=True)
    else:
        st.info("No se encontró vector de nulos post-imputación (revise columnas).")

    st.subheader("3) Validación: Cramér’s V Antes vs Después")
    if not bundle["cramers_compare"].empty:
        st.dataframe(bundle["cramers_compare"], use_container_width=True)
    else:
        st.info("No se pudo calcular Cramér’s V antes/después para las variables listadas.")

    st.subheader("4) Columnas eliminadas por redundancia")
    st.write(bundle["cols_dropped"] if bundle["cols_dropped"] else "No se eliminaron columnas.")

    st.subheader("Dataset procesado (muestra)")
    st.dataframe(data_imp.head(20), use_container_width=True)


elif page == "KPI":
    st.title("KPI")
    st.caption("Indicadores accionables (asociativos/diagnósticos; no causales).")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("P(Nacional) bajo", f"{kpis.get('p_nat', np.nan)*100:.1f}%", "Base nacional")
    with c2:
        kpi_card("KRAS (pp)", f"{kpis.get('KRAS', np.nan)*100:.1f}", "Estrato 1–2 vs 5–6")
    with c3:
        kpi_card("FD (pp)", f"{kpis.get('FD', np.nan)*100:.1f}", "Sin PC vs Con PC")
    with c4:
        kpi_card("KCC (pp)", f"{kpis.get('KCC', np.nan)*100:.1f}", "Madre baja vs alta edu")
    with c5:
        kpi_card("KDS", f"{kpis.get('KDS', np.nan):.3f}", "Promedio Cramér’s V")

    st.markdown("---")

    st.subheader("KPIs comparativos (selector de gráfico)")
    chart_kind = st.selectbox("Tipo de gráfico", ["Barras", "Líneas", "Boxplot"], index=0)

    # Tabla base para KPIs de brecha
    df_k = pd.DataFrame({
        "KPI": ["KRAS", "FD", "KCC"],
        "Brecha_pp": [kpis.get("KRAS", np.nan)*100, kpis.get("FD", np.nan)*100, kpis.get("KCC", np.nan)*100]
    })
    chart_selector(chart_kind, df_k, x="KPI", y="Brecha_pp", title="Brechas KPI (puntos porcentuales)")

    st.subheader("Riesgo por región (si existe REGION)")
    if isinstance(kpis.get("regional_risk", pd.Series()), pd.Series) and not kpis["regional_risk"].empty:
        rr = kpis["regional_risk"].reset_index()
        rr.columns = ["REGION", "P_bajo"]
        chart_kind2 = st.selectbox("Tipo de gráfico (región)", ["Barras", "Líneas"], index=0, key="regplot")
        chart_selector(chart_kind2, rr, x="REGION", y="P_bajo", title=f"Riesgo de bajo rendimiento por región (Brecha={kpis.get('equidad_regional_pp', np.nan):.1f} pp)")
        st.dataframe(rr.sort_values("P_bajo", ascending=False), use_container_width=True)
    else:
        st.info("No hay columna REGION o no se pudo calcular.")

    st.subheader("Impacto absoluto (casos) por departamento / región")
    colA, colB = st.columns(2)
    with colA:
        st.write("Top departamentos por casos (bajo)")
        kpi_dept = kpis.get("kpi_dept", pd.DataFrame())
        if not kpi_dept.empty:
            top = kpi_dept.head(15).reset_index()
            top.columns = ["Departamento", "N", "p_bajo", "casos", "exceso_pp", "exceso_casos"]
            chart_selector("Barras", top, x="Departamento", y="casos", title="Top 15 departamentos (casos de bajo rendimiento)")
            st.dataframe(top, use_container_width=True)
        else:
            st.info("No se encontró E_PRGM_DEPARTAMENTO para construir este KPI.")
    with colB:
        st.write("Impacto por región")
        kpi_region = kpis.get("kpi_region", pd.DataFrame())
        if not kpi_region.empty:
            kr = kpi_region.reset_index()
            kr.columns = ["REGION", "N", "p_bajo", "casos"]
            chart_selector("Barras", kr, x="REGION", y="casos", title="Casos de bajo rendimiento por región")
            st.dataframe(kr, use_container_width=True)
        else:
            st.info("No se pudo construir kpi_region.")


elif page == "EDA":
    st.title("EDA")
    st.caption("Exploración: cuantitativo, cualitativo y gráficos con selector.")

    tab1, tab2, tab3 = st.tabs(["Cuantitativo", "Cualitativo", "Gráficos"])

    with tab1:
        st.subheader("Cuantitativo")
        num_df = data_imp.select_dtypes(include=[np.number]).copy()
        if num_df.empty:
            st.info("No hay columnas numéricas disponibles.")
        else:
            st.write("Descriptivos")
            desc = num_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
            desc["skew"] = num_df.skew(numeric_only=True)
            desc["kurtosis"] = num_df.kurtosis(numeric_only=True)
            st.dataframe(desc, use_container_width=True)

            st.write("Correlación (heatmap)")
            chart_selector("Heatmap (correlación)", num_df, x=None, y=None, title="Matriz de correlación (numéricas)")

    with tab2:
        st.subheader("Cualitativo")
        TARGET = "RENDIMIENTO_GLOBAL"
        cat_cols = data_imp.select_dtypes(include=["object"]).columns.tolist()

        if TARGET not in data_imp.columns:
            st.info("No existe RENDIMIENTO_GLOBAL en el dataset procesado.")
        elif not cat_cols:
            st.info("No hay columnas categóricas.")
        else:
            # Ranking Cramér's V
            cr = {}
            for c in cat_cols:
                try:
                    cr[c] = cramers_v(data_imp[c], data_imp[TARGET])
                except Exception:
                    pass
            crs = pd.Series(cr).sort_values(ascending=False).head(25)
            st.write("Top 25 variables categóricas por asociación con el target (Cramér’s V)")
            st.dataframe(crs.to_frame("Cramér’s V"), use_container_width=True)

            st.write("Perfil categórico (seleccionable)")
            col = st.selectbox("Variable categórica", cat_cols, index=0)
            low_class = st.selectbox("Clase considerada 'bajo'", [0, 1, 2, 3], index=0)

            tmp = (
                data_imp.assign(is_low=(data_imp[TARGET] == low_class).astype(int))
                .groupby(col)["is_low"]
                .agg(N="size", p_bajo="mean")
                .sort_values(["p_bajo", "N"], ascending=False)
                .head(25)
                .reset_index()
            )
            st.dataframe(tmp, use_container_width=True)

            kind = st.selectbox("Tipo de gráfico", ["Barras", "Líneas"], index=0, key="catplot")
            chart_selector(kind, tmp, x=col, y="p_bajo", title=f"Riesgo (P(bajo)) por {col}")

    with tab3:
        st.subheader("Gráficos")
        st.write("Seleccione una vista y el tipo de gráfico.")

        view = st.selectbox(
            "Vista",
            ["Distribución del target", "Riesgo por estrato", "Riesgo por región", "Impacto vs riesgo (departamentos)"],
            index=0
        )
        chart_kind = st.selectbox("Tipo de gráfico", ["Barras", "Líneas", "Boxplot", "Histograma"], index=0, key="gkind")

        TARGET = "RENDIMIENTO_GLOBAL"
        if TARGET not in data_imp.columns:
            st.info("No existe RENDIMIENTO_GLOBAL.")
        else:
            if view == "Distribución del target":
                vc = data_imp[TARGET].value_counts(dropna=False).reset_index()
                vc.columns = ["Clase", "N"]
                chart_selector("Barras", vc, x="Clase", y="N", title="Distribución del rendimiento (codificado)")

            elif view == "Riesgo por estrato":
                if "F_ESTRATOVIVIENDA" not in data_imp.columns:
                    st.info("No existe F_ESTRATOVIVIENDA.")
                else:
                    t = (
                        data_imp.assign(BAJO=(data_imp[TARGET] == 0).astype(int))
                        .groupby("F_ESTRATOVIVIENDA")
                        .agg(p_bajo=("BAJO", "mean"), N=("BAJO", "size"))
                        .reset_index()
                        .sort_values("F_ESTRATOVIVIENDA")
                    )
                    chart_selector(chart_kind if chart_kind != "Histograma" else "Barras", t, x="F_ESTRATOVIVIENDA", y="p_bajo",
                                   title="P(bajo) por estrato (0–6)")
                    st.dataframe(t, use_container_width=True)

            elif view == "Riesgo por región":
                if "REGION" not in data_imp.columns:
                    st.info("No existe REGION.")
                else:
                    rr = (
                        data_imp.groupby("REGION")[TARGET]
                        .apply(lambda x: (x == 0).mean())
                        .reset_index()
                    )
                    rr.columns = ["REGION", "p_bajo"]
                    chart_selector(chart_kind if chart_kind != "Histograma" else "Barras", rr, x="REGION", y="p_bajo",
                                   title="P(bajo) por región")
                    st.dataframe(rr.sort_values("p_bajo", ascending=False), use_container_width=True)

            elif view == "Impacto vs riesgo (departamentos)":
                if "E_PRGM_DEPARTAMENTO" not in data_imp.columns:
                    st.info("No existe E_PRGM_DEPARTAMENTO.")
                else:
                    dfb = (
                        data_imp.assign(BAJO=(data_imp[TARGET] == 0).astype(int))
                        .groupby("E_PRGM_DEPARTAMENTO")
                        .agg(N=("BAJO", "size"), p_bajo=("BAJO", "mean"), casos=("BAJO", "sum"))
                        .sort_values("casos", ascending=False)
                        .head(40)
                        .reset_index()
                    )
                    # Bubble (siempre scatter)
                    fig = px.scatter(
                        dfb,
                        x="p_bajo",
                        y="casos",
                        size="N",
                        hover_name="E_PRGM_DEPARTAMENTO",
                        title="Priorización territorial (Top 40 por casos): Riesgo vs Impacto"
                    )
                    fig.update_xaxes(tickformat=".0%")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(dfb, use_container_width=True)
