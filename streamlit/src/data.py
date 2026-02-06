# src/data.py
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import kruskal

from mappings import grupo_dict_programas, grupo_dict_regiones


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


def preprocess_all(data_raw: pd.DataFrame) -> dict:
    """
    Replica su pipeline de notebook:
    - auditoría
    - evaluación de drop k>=2 (pero NO aplica drop, solo evidencia)
    - missingness patterns (N_NULOS, Kruskal, Cramér con MISSING_HEAVY)
    - imputación condicional por (Estrato, Departamento)
    - validación Cramér antes/después
    - reducción cardinalidad: programa -> macrogrupo, depto -> región
    - codificación: estrato/binarias/target
    - drops: INDICADOR_3/INDICADOR_4 si existen
    """
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

    # ----------------------
    # Imputación condicional
    # ----------------------
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

    present_vars = [c for c in vars_with_nulls if c in data_imp.columns]
    nulls_after = pd.Series(dtype=float)
    if present_vars:
        nulls_after = (data_imp[present_vars].isna().mean() * 100).round(2).sort_values(ascending=False)

    # ----------------------
    # Validación antes/después
    # ----------------------
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

    # ----------------------
    # Reducción cardinalidad + región
    # ----------------------
    if "E_PRGM_ACADEMICO" in data_imp.columns:
        data_imp["E_PRGM_ACADEMICO"] = data_imp["E_PRGM_ACADEMICO"].apply(map_to_group_program)

    if "E_PRGM_DEPARTAMENTO" in data_imp.columns:
        data_imp["REGION"] = data_imp["E_PRGM_DEPARTAMENTO"].apply(map_to_group_region)

    # ----------------------
    # Codificación
    # ----------------------
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

    # Drops por redundancia (como su notebook)
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
