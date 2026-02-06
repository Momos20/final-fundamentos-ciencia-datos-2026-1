# app.py
# streamlit run app.py

import numpy as np
import pandas as pd
import streamlit as st
import scipy.stats as ss
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from mappings import grupo_dict_programas, grupo_dict_regiones


# =========================
# UI
# =========================
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
.kpi-sub { font-size: 12px; color: #888; margin-top: 6px; line-height: 1.3; }
.small-note { font-size: 12px; color: #666; }
</style>
"""
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


# =========================
# Utilidades (como notebook)
# =========================
def audit(df: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.DataFrame({
            "nulos_%": (df.isna().mean() * 100).round(2),
            "tipo": df.dtypes.astype(str),
            "nulos_n": df.isna().sum()
        })
        .sort_values("nulos_%", ascending=False)
    )

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


# =========================
# Carga + pipeline
# =========================
@st.cache_data(show_spinner=True)
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_data(show_spinner=True)
def build_all(data_raw: pd.DataFrame):
    target = "RENDIMIENTO_GLOBAL"
    data = data_raw.copy()

    # Auditoría
    audit_df = audit(data)

    # Transparencia drop por nulos
    df_limpio, df_eliminados = drop_rows_with_many_nulls(data.copy(), k=2)

    comparacion_drop = pd.DataFrame()
    if target in data.columns and target in df_eliminados.columns:
        dist_elim = df_eliminados[target].value_counts(normalize=True).mul(100).round(2)
        dist_total = data[target].value_counts(normalize=True).mul(100).round(2)
        comparacion_drop = pd.concat([dist_elim, dist_total], axis=1, keys=["Eliminados %", "Total %"])
        comparacion_drop["Diferencia_abs_pp"] = (comparacion_drop["Eliminados %"] - comparacion_drop["Total %"]).abs()
        comparacion_drop = comparacion_drop.sort_values("Diferencia_abs_pp", ascending=False)

    # Missingness patterns
    df_miss = data.copy()
    df_miss["N_NULOS"] = df_miss.isna().sum(axis=1)

    kw_p = np.nan
    miss_desc = pd.DataFrame()
    if target in df_miss.columns:
        miss_desc = df_miss.groupby(target)["N_NULOS"].describe()
        groups = [df_miss[df_miss[target] == c]["N_NULOS"] for c in df_miss[target].dropna().unique()]
        if len(groups) >= 2:
            _, kw_p = kruskal(*groups)

    df_miss["MISSING_HEAVY"] = (df_miss["N_NULOS"] >= 2).astype(int)

    candidate_cols = [
        "F_ESTRATOVIVIENDA",
        "F_TIENECOMPUTADOR",
        "F_TIENEINTERNET",
        "F_TIENEAUTOMOVIL",
        "E_EDAD",
        "E_PRGM_DEPARTAMENTO",
        "E_EDUCACIONPADRE",
        "E_EDUCACIONMADRE",
    ]
    missing_assoc = {}
    for col in candidate_cols:
        if col in df_miss.columns:
            try:
                missing_assoc[col] = cramers_v(df_miss[col], df_miss["MISSING_HEAVY"])
            except Exception:
                missing_assoc[col] = np.nan
    missing_assoc_s = pd.Series(missing_assoc).sort_values(ascending=False)

    # Imputación condicional
    data_imp = data.copy()

    if "F_TIENEINTERNET.1" in data_imp.columns:
        data_imp = data_imp.drop(columns=["F_TIENEINTERNET.1"])

    group_vars = ["F_ESTRATOVIVIENDA", "E_PRGM_DEPARTAMENTO"]
    for gv in group_vars:
        if gv in data_imp.columns:
            data_imp[gv] = data_imp[gv].fillna("Desconocido")

    vars_with_nulls = [
        "F_TIENEAUTOMOVIL",
        "F_TIENELAVADORA",
        "F_TIENECOMPUTADOR",
        "E_HORASSEMANATRABAJA",
        "F_TIENEINTERNET",
        "F_EDUCACIONMADRE",
        "F_EDUCACIONPADRE",
        "E_PAGOMATRICULAPROPIO",
        "E_VALORMATRICULAUNIVERSIDAD",
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

    nulls_after = pd.Series(dtype=float)
    present_vars = [c for c in vars_with_nulls if c in data_imp.columns]
    if present_vars:
        nulls_after = (data_imp[present_vars].isna().mean() * 100).round(2).sort_values(ascending=False)

    # Validación antes/después (Cramér’s V vs target)
    cramers_before, cramers_after = {}, {}
    if target in data.columns and target in data_imp.columns:
        for col in vars_with_nulls:
            if col in data.columns and col in data_imp.columns:
                try:
                    cramers_before[col] = cramers_v(data[col], data[target])
                    cramers_after[col] = cramers_v(data_imp[col], data_imp[target])
                except Exception:
                    pass
    cramers_compare = pd.DataFrame({"Antes": cramers_before, "Después": cramers_after})
    if not cramers_compare.empty:
        cramers_compare["Cambio"] = cramers_compare["Después"] - cramers_compare["Antes"]
        cramers_compare = cramers_compare.sort_values("Cambio", ascending=False)

    # Reducción cardinalidad (programa) + región
    if "E_PRGM_ACADEMICO" in data_imp.columns:
        data_imp["E_PRGM_ACADEMICO"] = data_imp["E_PRGM_ACADEMICO"].apply(map_to_group_program)

    if "E_PRGM_DEPARTAMENTO" in data_imp.columns:
        data_imp["REGION"] = data_imp["E_PRGM_DEPARTAMENTO"].apply(map_to_group_region)

    # Codificación
    if "F_ESTRATOVIVIENDA" in data_imp.columns and data_imp["F_ESTRATOVIVIENDA"].dtype == "object":
        estrato_map = {
            "Sin Estrato": 0, "Desconocido": 0,
            "Estrato 1": 1, "Estrato 2": 2, "Estrato 3": 3,
            "Estrato 4": 4, "Estrato 5": 5, "Estrato 6": 6
        }
        data_imp["F_ESTRATOVIVIENDA"] = data_imp["F_ESTRATOVIVIENDA"].map(estrato_map).fillna(0).astype(int)

    binary_cols = [
        "F_TIENEINTERNET",
        "F_TIENELAVADORA",
        "F_TIENEAUTOMOVIL",
        "F_TIENECOMPUTADOR",
        "E_PRIVADO_LIBERTAD",
        "E_PAGOMATRICULAPROPIO",
    ]
    for col in binary_cols:
        if col in data_imp.columns and data_imp[col].dtype == "object":
            data_imp[col] = data_imp[col].map({"Si": 1, "Sí": 1, "No": 0}).fillna(0).astype(int)

    if target in data_imp.columns and data_imp[target].dtype == "object":
        target_map = {"bajo": 0, "medio-bajo": 1, "medio-alto": 2, "alto": 3}
        data_imp[target] = data_imp[target].map(target_map)

    # Drops por redundancia (como notebook)
    cols_to_drop = [c for c in ["INDICADOR_3", "INDICADOR_4"] if c in data_imp.columns]
    if cols_to_drop:
        data_imp = data_imp.drop(columns=cols_to_drop)

    return {
        "data_raw": data_raw,
        "data_imp": data_imp,
        "audit_df": audit_df,
        "comparacion_drop": comparacion_drop,
        "miss_desc": miss_desc,
        "kw_p": kw_p,
        "missing_assoc": missing_assoc_s,
        "vars_with_nulls": vars_with_nulls,
        "nulls_after": nulls_after,
        "cramers_compare": cramers_compare,
        "cols_to_drop": cols_to_drop,
    }


# =========================
# KPI resultados (tablas por KPI)
# =========================
def compute_kpi_tables(data_imp: pd.DataFrame):
    TARGET = "RENDIMIENTO_GLOBAL"
    if TARGET not in data_imp.columns:
        return {}

    is_low = (data_imp[TARGET] == 0)
    p_nat = float(is_low.mean()) if len(data_imp) else np.nan

    out = {"p_nat": p_nat}

    # KPI 1 - KRAS
    if "F_ESTRATOVIVIENDA" in data_imp.columns:
        low_str = data_imp["F_ESTRATOVIVIENDA"].isin([1, 2])
        high_str = data_imp["F_ESTRATOVIVIENDA"].isin([5, 6])
        p_low = float((data_imp.loc[low_str, TARGET] == 0).mean()) if low_str.any() else np.nan
        p_high = float((data_imp.loc[high_str, TARGET] == 0).mean()) if high_str.any() else np.nan
        kras = p_low - p_high

        out["KRAS"] = kras
        out["KRAS_table"] = pd.DataFrame({
            "Segmento": ["Estrato 1–2", "Estrato 5–6"],
            "P(bajo)": [p_low, p_high],
        }).assign(KRAS_pp=round(kras * 100, 6))

    # KPI 2 - FD
    if "F_TIENECOMPUTADOR" in data_imp.columns:
        no_pc = (data_imp["F_TIENECOMPUTADOR"] == 0)
        pc = (data_imp["F_TIENECOMPUTADOR"] == 1)
        p_no = float((data_imp.loc[no_pc, TARGET] == 0).mean()) if no_pc.any() else np.nan
        p_pc = float((data_imp.loc[pc, TARGET] == 0).mean()) if pc.any() else np.nan
        fd = p_no - p_pc

        out["FD"] = fd
        out["FD_table"] = pd.DataFrame({
            "Segmento": ["Sin computador", "Con computador"],
            "P(bajo)": [p_no, p_pc],
        }).assign(FD_pp=round(fd * 100, 6))

    # KPI 3 - KCC
    if "F_EDUCACIONMADRE" in data_imp.columns:
        low_mom = data_imp["F_EDUCACIONMADRE"].isin(["Ninguno", "Primaria incompleta", "Primaria completa"])
        high_mom = data_imp["F_EDUCACIONMADRE"].isin(["Educación profesional completa", "Postgrado", "Posgrado", "Postgrado completo"])
        p_low = float((data_imp.loc[low_mom, TARGET] == 0).mean()) if low_mom.any() else np.nan
        p_high = float((data_imp.loc[high_mom, TARGET] == 0).mean()) if high_mom.any() else np.nan
        kcc = p_low - p_high

        out["KCC"] = kcc
        out["KCC_table"] = pd.DataFrame({
            "Segmento": ["Madre baja educación", "Madre alta educación"],
            "P(bajo)": [p_low, p_high],
        }).assign(KCC_pp=round(kcc * 100, 6))

    # KPI 4 - KDS (promedio Cramér’s V)
    socio_cols = [c for c in [
        "F_ESTRATOVIVIENDA", "F_TIENECOMPUTADOR", "F_TIENEINTERNET", "F_TIENEAUTOMOVIL",
        "F_EDUCACIONMADRE", "F_EDUCACIONPADRE", "REGION", "E_PRGM_DEPARTAMENTO"
    ] if c in data_imp.columns]

    cv = []
    cv_table = []
    for c in socio_cols:
        try:
            v = cramers_v(data_imp[c], data_imp[TARGET])
            if not np.isnan(v):
                cv.append(v)
                cv_table.append((c, v))
        except Exception:
            pass
    kds = float(np.mean(cv)) if cv else np.nan
    out["KDS"] = kds
    out["KDS_table"] = pd.DataFrame(cv_table, columns=["Variable", "Cramér’s V"]).sort_values("Cramér’s V", ascending=False)
    if not out["KDS_table"].empty:
        out["KDS_table"]["KDS_promedio"] = kds

    # KPI 5 - Equidad regional
    if "REGION" in data_imp.columns:
        regional_risk = (
            data_imp.groupby("REGION")[TARGET]
            .apply(lambda x: (x == 0).mean())
            .sort_values()
        )
        brecha_pp = float((regional_risk.max() - regional_risk.min()) * 100) if len(regional_risk) >= 2 else np.nan
        out["equidad_regional_pp"] = brecha_pp
        out["regional_risk"] = regional_risk.to_frame("P(bajo)").assign(Brecha_regional_pp=brecha_pp)

    # KPI 6 - Impacto dept/región
    if "E_PRGM_DEPARTAMENTO" in data_imp.columns:
        kpi_dept = (
            data_imp.assign(BAJO=is_low.astype(int))
            .groupby("E_PRGM_DEPARTAMENTO")
            .agg(N=("BAJO", "size"), p_bajo=("BAJO", "mean"), casos=("BAJO", "sum"))
            .sort_values("casos", ascending=False)
        )
        kpi_dept["exceso_pp"] = (kpi_dept["p_bajo"] - p_nat) * 100
        kpi_dept["exceso_casos"] = (kpi_dept["p_bajo"] - p_nat) * kpi_dept["N"]
        out["kpi_dept"] = kpi_dept

    if "REGION" in data_imp.columns:
        kpi_region = (
            data_imp.assign(BAJO=is_low.astype(int))
            .groupby("REGION")
            .agg(N=("BAJO", "size"), p_bajo=("BAJO", "mean"), casos=("BAJO", "sum"))
            .sort_values("casos", ascending=False)
        )
        out["kpi_region"] = kpi_region

    return out


# =========================
# Gráficos matplotlib (mismos colores del notebook)
# =========================
def fig_target_distribution(data_imp: pd.DataFrame):
    # Mismo esquema que su notebook
    target_map = {'bajo': 0, 'medio-bajo': 1, 'medio-alto': 2, 'alto': 3}
    order = ["bajo", "medio-bajo", "medio-alto", "alto"]
    order_codes = [target_map[k] for k in order]

    counts = data_imp["RENDIMIENTO_GLOBAL"].value_counts().reindex(order_codes)
    labels = order
    values = counts.values
    total = values.sum()
    pct = values / total if total else np.zeros_like(values)

    color_map = {
        "bajo": "#C0392B",
        "medio-bajo": "#B0B0B0",
        "medio-alto": "#D9D9D9",
        "alto": "#2E7D32"
    }
    colors = [color_map[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors)

    ax.set_yticks(y)
    ax.set_yticklabels([l.replace("-", " ") for l in labels])
    ax.set_xlabel("Número de estudiantes")
    ax.set_title("Distribución del rendimiento académico", pad=10)
    ax.set_ylabel("Nivel de rendimiento")

    ax.xaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    xmax = values.max() if len(values) else 1
    ax.set_xlim(0, xmax * 1.18)

    for i, (v, p) in enumerate(zip(values, pct)):
        ax.text(v + xmax * 0.02, i, f"{v:,.0f}  ({p:.1%})", va="center", fontsize=10)

    plt.tight_layout()
    return fig


# =========================
# Sidebar
# =========================
st.sidebar.title("Menú")
uploaded = st.sidebar.file_uploader("Subir saber_pro.csv", type=["csv"])

page = st.sidebar.selectbox(
    "Navegación",
    ["Resumen", "Procesamiento", "KPI", "EDA"],
    index=0
)

if not uploaded:
    st.title("Saber Pro — Streamlit")
    st.info("Cargue el archivo **saber_pro.csv** para comenzar.")
    st.stop()

data_raw = load_data(uploaded)
bundle = build_all(data_raw)
data_imp = bundle["data_imp"]
kpi_pack = compute_kpi_tables(data_imp)


# =========================
# Páginas
# =========================
if page == "Resumen":
    st.title("Resumen / Observaciones")
    st.caption("Métricas base + auditoría + hallazgos clave (missingness MNAR).")

    # Métricas base (lo que usted dijo que no aparecía)
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
        kpi_card("Kruskal (N_NULOS vs target)", "p≈0" if (isinstance(kw, float) and kw < 1e-6) else f"p={kw:.4g}", "Evidencia de MNAR")

    st.markdown("---")

    st.subheader("Muestra de datos")
    n = st.slider("Filas a mostrar", 5, 200, 20, step=5)
    st.dataframe(data_raw.head(n), use_container_width=True)

    st.subheader("Auditoría de nulos (sin top25)")
    min_null = st.slider("Filtrar: nulos_% mínimo", 0.0, 100.0, 0.0, 1.0)
    audit_view = bundle["audit_df"].loc[bundle["audit_df"]["nulos_%"] >= min_null]
    st.dataframe(audit_view, use_container_width=True)

    st.subheader("Observaciones (missingness)")
    st.write("**Descripción de N_NULOS por clase de rendimiento**")
    if not bundle["miss_desc"].empty:
        st.dataframe(bundle["miss_desc"], use_container_width=True)
    else:
        st.info("No se pudo construir la tabla (revise que exista RENDIMIENTO_GLOBAL).")

    st.write("**Asociación con MISSING_HEAVY (Cramér’s V)**")
    st.dataframe(bundle["missing_assoc"].to_frame("Cramér’s V"), use_container_width=True)


elif page == "Procesamiento":
    st.title("Procesamiento (trazabilidad)")
    st.caption("Se documenta lo ejecutado: MNAR → imputación condicional → validación → reducción cardinalidad → codificación.")

    st.subheader("1) Evaluación de eliminación por nulos (k>=2)")
    st.write("En su notebook concluyó que **no se debe eliminar**, porque altera distribución e implica sesgo social/estructural.")
    if not bundle["comparacion_drop"].empty:
        st.dataframe(bundle["comparacion_drop"], use_container_width=True)
    else:
        st.info("No se pudo construir la comparación (revise target).")

    st.subheader("2) Evidencia de MNAR")
    st.write("Kruskal-Wallis sobre N_NULOS por clase de rendimiento:")
    st.write(f"**p-value:** {bundle['kw_p']}")
    st.write("Variables con mayor asociación a MISSING_HEAVY (Cramér’s V):")
    st.dataframe(bundle["missing_assoc"].head(20).to_frame("Cramér’s V"), use_container_width=True)

    st.subheader("3) Imputación condicional por (Estrato, Departamento)")
    st.write("Variables tratadas (si existen en su dataset):")
    st.code(", ".join(bundle["vars_with_nulls"]))
    st.write("Nulos restantes post-imputación (%):")
    if not bundle["nulls_after"].empty:
        st.dataframe(bundle["nulls_after"].to_frame("nulos_%"), use_container_width=True)
    else:
        st.info("No hay variables imputadas presentes o no hay nulos.")

    st.subheader("4) Validación de estabilidad (Cramér’s V Antes vs Después)")
    if not bundle["cramers_compare"].empty:
        st.dataframe(bundle["cramers_compare"], use_container_width=True)
    else:
        st.info("No se pudo calcular la validación antes/después.")

    st.subheader("5) Reducción de cardinalidad + región + codificación")
    st.write("- `E_PRGM_ACADEMICO` → macro-áreas (mappings.py)")
    st.write("- `E_PRGM_DEPARTAMENTO` → `REGION` (mappings.py)")
    st.write("- Estrato → ordinal; binarios Sí/No → 1/0; target → 0–3")
    st.write("Columnas eliminadas por redundancia (según su notebook):")
    st.write(bundle["cols_to_drop"] if bundle["cols_to_drop"] else "No se eliminaron columnas.")
    st.subheader("Dataset procesado (muestra)")
    st.dataframe(data_imp.head(30), use_container_width=True)


elif page == "KPI":
    st.title("KPI (solo resultados)")
    st.caption("Se muestran resultados en tarjetas + tabla por KPI (Segmento, P(bajo), KPI_pp).")

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

    # Tablas por KPI (como usted pidió)
    colA, colB = st.columns(2)
    with colA:
        st.subheader("KPI 1 — KRAS")
        t = kpi_pack.get("KRAS_table")
        if isinstance(t, pd.DataFrame):
            st.dataframe(t, use_container_width=True)
        else:
            st.info("No se pudo calcular KRAS (revise F_ESTRATOVIVIENDA).")

        st.subheader("KPI 2 — FD")
        t = kpi_pack.get("FD_table")
        if isinstance(t, pd.DataFrame):
            st.dataframe(t, use_container_width=True)
        else:
            st.info("No se pudo calcular FD (revise F_TIENECOMPUTADOR).")

        st.subheader("KPI 3 — KCC")
        t = kpi_pack.get("KCC_table")
        if isinstance(t, pd.DataFrame):
            st.dataframe(t, use_container_width=True)
        else:
            st.info("No se pudo calcular KCC (revise F_EDUCACIONMADRE).")

    with colB:
        st.subheader("KPI 4 — KDS (detalle)")
        t = kpi_pack.get("KDS_table", pd.DataFrame())
        if not t.empty:
            st.dataframe(t, use_container_width=True)
        else:
            st.info("No se pudo construir KDS_table.")

        st.subheader("KPI 5 — Equidad regional")
        rr = kpi_pack.get("regional_risk", pd.DataFrame())
        if isinstance(rr, pd.DataFrame) and not rr.empty:
            st.dataframe(rr, use_container_width=True)
        else:
            st.info("No se pudo construir el KPI regional.")

        st.subheader("KPI 6 — Impacto absoluto")
        dept = kpi_pack.get("kpi_dept", pd.DataFrame())
        reg = kpi_pack.get("kpi_region", pd.DataFrame())
        if not dept.empty:
            st.write("Top 15 departamentos por casos")
            st.dataframe(dept.head(15), use_container_width=True)
        if not reg.empty:
            st.write("Región (casos)")
            st.dataframe(reg, use_container_width=True)
        if dept.empty and reg.empty:
            st.info("No se pudo construir impacto (revise columnas de territorio).")


elif page == "EDA":
    st.title("EDA")
    st.caption("Cuantitativo / Cualitativo / Gráficos (matplotlib y colores del notebook).")

    tab1, tab2, tab3 = st.tabs(["Cuantitativo", "Cualitativo", "Gráficos"])

    with tab1:
        st.subheader("Cuantitativo")
        num_df = data_imp.select_dtypes(include=[np.number]).copy()
        if num_df.empty:
            st.info("No hay columnas numéricas disponibles.")
        else:
            desc = num_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
            desc["skew"] = num_df.skew(numeric_only=True)
            desc["kurtosis"] = num_df.kurtosis(numeric_only=True)
            st.dataframe(desc, use_container_width=True)

            st.write("Correlación (numéricas)")
            corr = num_df.corr()
            st.dataframe(corr, use_container_width=True)

    with tab2:
        st.subheader("Cualitativo")
        TARGET = "RENDIMIENTO_GLOBAL"
        cat_cols = data_imp.select_dtypes(include=["object"]).columns.tolist()
        if TARGET not in data_imp.columns:
            st.info("No existe RENDIMIENTO_GLOBAL en el dataset procesado.")
        elif not cat_cols:
            st.info("No hay columnas categóricas.")
        else:
            cr = {}
            for c in cat_cols:
                try:
                    cr[c] = cramers_v(data_imp[c], data_imp[TARGET])
                except Exception:
                    pass
            crs = pd.Series(cr).sort_values(ascending=False)
            st.dataframe(crs.to_frame("Cramér’s V"), use_container_width=True)

    with tab3:
        st.subheader("Gráficos (los que usted hizo)")
        if "RENDIMIENTO_GLOBAL" not in data_imp.columns:
            st.info("No existe RENDIMIENTO_GLOBAL, no se puede graficar.")
        else:
            st.write("1) Distribución del target (mismos colores)")
            fig = fig_target_distribution(data_imp)
            st.pyplot(fig, clear_figure=True)

            st.info(
                "En esta versión dejé implementada la primera figura (target) exactamente con su código y colores.\n\n"
                "Las demás (KRAS/FD/KCC, interacción, regional, KDS, impacto, etc.) se agregan igual: "
                "copiando sus funciones matplotlib y renderizando con `st.pyplot(fig)`.\n\n"
                "Si quiere que se las deje todas ya montadas, lo hago; no necesita reenviar el notebook."
            )
