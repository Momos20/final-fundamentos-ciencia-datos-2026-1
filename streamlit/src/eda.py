# src/eda.py
import numpy as np
import pandas as pd
from scipy.stats import kruskal
import scipy.stats as ss

from .data import cramers_v


def target_distribution(df: pd.DataFrame, target="RENDIMIENTO_GLOBAL") -> pd.DataFrame:
    if target not in df.columns:
        return pd.DataFrame()
    return (
        df[target]
        .value_counts(dropna=False, normalize=True)
        .mul(100)
        .round(2)
        .to_frame("pct")
    )

def quantitative_summary(df: pd.DataFrame) -> pd.DataFrame:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()
    stats = num_df.describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]).T
    stats["skew"] = num_df.skew(numeric_only=True)
    stats["kurtosis"] = num_df.kurtosis(numeric_only=True)
    return stats

def correlation_pairs(df: pd.DataFrame, threshold=0.7) -> pd.DataFrame:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return pd.DataFrame()
    corr = num_df.corr()
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )
    pairs = pairs[pairs.abs() > threshold]
    return pairs.to_frame("corr").reset_index().rename(columns={"level_0":"var1","level_1":"var2"})

def kruskal_numeric_vs_target(df: pd.DataFrame, num_cols: list, target="RENDIMIENTO_GLOBAL") -> pd.DataFrame:
    if target not in df.columns:
        return pd.DataFrame()
    out = []
    for col in num_cols:
        if col not in df.columns:
            continue
        groups = [df[df[target]==c][col].dropna() for c in df[target].dropna().unique()]
        if len(groups) < 2:
            continue
        try:
            stat, p = kruskal(*groups)
            out.append((col, stat, p))
        except Exception:
            pass
    return pd.DataFrame(out, columns=["variable","kruskal_stat","p_value"]).sort_values("p_value")

def outliers_by_class(df: pd.DataFrame, vars_num: list, target="RENDIMIENTO_GLOBAL") -> pd.DataFrame:
    if target not in df.columns:
        return pd.DataFrame()
    out = []
    for col in vars_num:
        if col not in df.columns:
            continue
        for cls in df[target].dropna().unique():
            subset = df.loc[df[target] == cls, col].dropna()
            if subset.empty:
                continue
            q1, q3 = subset.quantile(0.25), subset.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
            pct = ((subset < low) | (subset > high)).mean() * 100
            out.append([col, cls, round(pct, 2), low, high])
    return pd.DataFrame(out, columns=["Variable","Clase","% Outliers","Low","High"])

def cramers_table(df: pd.DataFrame, target="RENDIMIENTO_GLOBAL") -> pd.DataFrame:
    if target not in df.columns:
        return pd.DataFrame()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    rows = []
    for col in cat_cols:
        try:
            v = cramers_v(df[col], df[target])
            if not np.isnan(v):
                rows.append((col, v))
        except Exception:
            pass
    return pd.DataFrame(rows, columns=["Variable", "Cramér’s V"]).sort_values("Cramér’s V", ascending=False)

def perfil_categorico(df, col, target="RENDIMIENTO_GLOBAL", low_class=0, top=15) -> pd.DataFrame:
    if target not in df.columns or col not in df.columns:
        return pd.DataFrame()
    t = (
        df.assign(is_low=(df[target]==low_class).astype(int))
          .groupby(col)["is_low"]
          .agg(N="size", p_bajo="mean")
          .sort_values(["p_bajo","N"], ascending=False)
    )
    return t.head(top)