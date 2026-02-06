# src/charts.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Helpers
def pct_axis(ax):
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# 1) Distribución target (exacto)
def fig_target_distribution(data_imp):
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
    ax.barh(y, values, color=colors)

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


# 2) KRAS barra (exacto estilo)
def fig_kras_bar(p_low, p_high, kras):
    labels = ["Estrato 1–2", "Estrato 5–6"]
    values = [p_low, p_high]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=["#C0392B", "#2E7D32"])
    ax.set_ylabel("Proporción de estudiantes en bajo rendimiento")
    ax.set_title("Brecha socioeconómica en el rendimiento académico")

    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y/2, f"{y*100:.1f}%",
                ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    x1, x2 = 0, 1
    y1, y2 = values
    y_bracket = max(values) + 0.03

    ax.plot([x1, x1, x2, x2], [y1, y_bracket, y_bracket, y2], color="black", linewidth=1.5)
    ax.text(0.5, y_bracket + 0.01, f"Brecha = {kras*100:.1f} pp", ha="center",
            fontsize=11, fontweight="bold")

    ax.set_ylim(0, y_bracket + 0.08)
    plt.tight_layout()
    return fig


# 3) KRAS por estrato (línea + puntos)
def fig_kras_gradient(data_imp, target="RENDIMIENTO_GLOBAL", estrato_col="F_ESTRATOVIVIENDA", low_class=0):
    if estrato_col not in data_imp.columns:
        return None

    t = (
        data_imp
        .assign(BAJO=(data_imp[target] == low_class).astype(int))
        .groupby(estrato_col)["BAJO"]
        .agg(p_bajo="mean", N="size")
        .sort_index()
    )

    t = t.loc[t.index.isin([0,1,2,3,4,5,6])]
    x = t.index.astype(int).values
    y = t["p_bajo"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, color="0.6", linewidth=2.2, zorder=1)
    ax.scatter(x, y, s=70, color="tab:blue", edgecolor="white", linewidth=1.2, zorder=2)

    ax.set_xticks(x)
    ax.set_xlabel("Estrato socioeconómico (0–6)")
    ax.set_ylabel("Probabilidad de bajo rendimiento")
    ax.set_title("Riesgo de bajo rendimiento por estrato socioeconómico")
    pct_axis(ax)

    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.grid(False, axis="x")

    for xi, ni in zip(x, t["N"].values):
        ax.annotate(f"n={ni}", (xi, 0), xytext=(0, -18), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color="0.4")

    plt.tight_layout()
    return fig


# 4) FD barra
def fig_fd_bar(p_no, p_pc, fd):
    labels = ["Sin computador", "Con computador"]
    values = [p_no, p_pc]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=["#C62828", "#2E7D32"])

    ax.set_ylabel("Proporción de estudiantes en bajo rendimiento")
    ax.set_title("Impacto del acceso a computador en el rendimiento")

    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y/2, f"{y*100:.1f}%",
                ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    x1, x2 = 0, 1
    y1, y2 = values
    y_bracket = max(values) + 0.03

    ax.plot([x1, x1, x2, x2], [y1, y_bracket, y_bracket, y2], color="black", linewidth=1.5)
    ax.text(0.5, y_bracket + 0.01, f"Brecha = {fd*100:.1f} pp", ha="center",
            fontsize=11, fontweight="bold")

    ax.set_ylim(0, y_bracket + 0.08)
    plt.tight_layout()
    return fig


# 5) FD interacción estrato x pc
def fig_fd_interaction(data_imp, target="RENDIMIENTO_GLOBAL", low_class=0,
                       estrato_col="F_ESTRATOVIVIENDA", pc_col="F_TIENECOMPUTADOR"):
    if estrato_col not in data_imp.columns or pc_col not in data_imp.columns:
        return None

    low_str = data_imp[estrato_col].isin([1,2])
    high_str = data_imp[estrato_col].isin([5,6])
    df2 = data_imp.loc[low_str | high_str, [target, pc_col, estrato_col]].copy()

    df2["SEG_ESTRATO"] = np.where(df2[estrato_col].isin([1,2]), "Estrato 1–2", "Estrato 5–6")
    df2["PC"] = np.where(df2[pc_col] == 1, "Con computador", "Sin computador")
    df2["BAJO"] = (df2[target] == low_class).astype(int)

    g = (
        df2.groupby(["SEG_ESTRATO","PC"])["BAJO"]
        .mean()
        .unstack("PC")
        .reindex(index=["Estrato 1–2","Estrato 5–6"], columns=["Sin computador","Con computador"])
    )

    x = np.arange(len(g.index))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width/2, g["Sin computador"].values, width, color="#C62828", label="Sin computador", zorder=2)
    ax.bar(x + width/2, g["Con computador"].values, width, color="#2E7D32", label="Con computador", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(g.index)
    ax.set_ylabel("Probabilidad de bajo rendimiento")
    ax.set_title("Interacción Estrato × Computador en el riesgo de bajo rendimiento")
    pct_axis(ax)

    ax.grid(True, axis="y", linestyle="--", alpha=0.35, zorder=1)
    ax.grid(False, axis="x")

    for i, estrato in enumerate(g.index):
        for col in g.columns:
            val = g.loc[estrato, col]
            xpos = x[i] + (-width/2 if col == "Sin computador" else width/2)
            ax.text(xpos, val + 0.01, f"{val:.1%}", ha="center", va="bottom", fontsize=10, color="black")

    ax.legend(frameon=False)
    plt.tight_layout()
    return fig


# 6) KCC barra
def fig_kcc_bar(p_low, p_high, kcc):
    labels = ["Madre baja educación", "Madre alta educación"]
    values = [p_low, p_high]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=["#C62828", "#2E7D32"])

    ax.set_ylabel("Proporción de estudiantes en bajo rendimiento")
    ax.set_title("Impacto del capital cultural materno")

    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y/2, f"{y*100:.1f}%",
                ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    x1, x2 = 0, 1
    y1, y2 = values
    y_bracket = max(values) + 0.03

    ax.plot([x1, x1, x2, x2], [y1, y_bracket, y_bracket, y2], color="black", linewidth=1.5)
    ax.text(0.5, y_bracket + 0.01, f"Brecha = {kcc*100:.1f} pp", ha="center",
            fontsize=11, fontweight="bold")

    ax.set_ylim(0, y_bracket + 0.08)
    plt.tight_layout()
    return fig


# 7) KCC por educación madre (barh)
def fig_kcc_mother_education(data_imp, target="RENDIMIENTO_GLOBAL", low_class=0, mom_col="F_EDUCACIONMADRE"):
    if mom_col not in data_imp.columns:
        return None

    df2 = data_imp[[target, mom_col]].copy().dropna(subset=[mom_col])
    df2["BAJO"] = (df2[target] == low_class).astype(int)

    t = (
        df2.groupby(mom_col)["BAJO"]
        .agg(p_bajo="mean", N="size")
        .sort_values("p_bajo", ascending=False)
    )

    colors = ["0.7"] * len(t)
    if len(colors) > 0:
        colors[0] = "#C62828"
        colors[-1] = "#2E7D32"

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(t.index, t["p_bajo"].values, color=colors, zorder=2)
    ax.invert_yaxis()

    ax.set_xlabel("Probabilidad de bajo rendimiento")
    ax.set_title("Riesgo de bajo rendimiento según nivel educativo de la madre")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.grid(True, axis="x", linestyle="--", alpha=0.35, zorder=1)
    ax.grid(False, axis="y")

    xmax = t["p_bajo"].max() if len(t) else 1
    ax.set_xlim(0, xmax * 1.18)
    offset = xmax * 0.02

    for i, (cat, row) in enumerate(t.iterrows()):
        val = row["p_bajo"]
        if i == 0:
            txt_color, fw = "#C62828", "bold"
        elif i == len(t)-1:
            txt_color, fw = "#2E7D32", "bold"
        else:
            txt_color, fw = "0.35", "normal"

        ax.text(val + offset, cat, f"{val:.1%}", va="center", ha="left",
                fontsize=10, fontweight=fw, color=txt_color)

    plt.tight_layout()
    return fig


# 8) KPI territorial región (riesgo)
def fig_regional_risk(regional_risk):
    labels = regional_risk.index.tolist()
    values = regional_risk.values

    best = regional_risk.idxmin()
    worst = regional_risk.idxmax()

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, values, color="lightgray")

    best_idx = labels.index(best)
    worst_idx = labels.index(worst)

    ax.bar(best_idx, values[best_idx], color="#2E7D32", label="Mejor región")
    ax.bar(worst_idx, values[worst_idx], color="#C62828", label="Peor región")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Proporción de bajo rendimiento")
    ax.set_xlabel("Region")
    ax.set_title("Brecha regional de bajo rendimiento")
    ax.axhline(regional_risk.mean(), linestyle="--", label="Promedio nacional")
    ax.legend()
    plt.tight_layout()
    return fig


# 9) KPI territorial depto (riesgo)
def fig_dept_risk(regional_risk_d):
    labels = regional_risk_d.index.tolist()
    values = regional_risk_d.values

    best = regional_risk_d.idxmin()
    worst = regional_risk_d.idxmax()

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, values, color="lightgray")

    best_idx = labels.index(best)
    worst_idx = labels.index(worst)

    ax.bar(best_idx, values[best_idx], color="#2E7D32", label="Mejor región")
    ax.bar(worst_idx, values[worst_idx], color="#C62828", label="Peor región")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Proporción de bajo rendimiento")
    ax.set_xlabel("Departamento")
    ax.set_title("Brecha Departamental de bajo rendimiento")
    ax.axhline(regional_risk_d.mean(), linestyle="--", label="Promedio nacional")
    ax.legend()
    plt.tight_layout()
    return fig


# 10) KDS bar (Cramér’s V)
def fig_kds_bars(comparison_sorted, kds):
    labels = comparison_sorted.index.tolist()
    values = comparison_sorted.values

    best = comparison_sorted.idxmax()
    worst = comparison_sorted.idxmin()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x, values, color="lightgray")

    best_idx = labels.index(best)
    worst_idx = labels.index(worst)

    ax.bar(best_idx, values[best_idx], color="#2E7D32", label="Mayor dependencia")
    ax.bar(worst_idx, values[worst_idx], color="#C62828", label="Menor dependencia")

    ax.axhline(kds, linestyle="--", label="Promedio (KDS)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Cramér’s V con RENDIMIENTO")
    ax.set_xlabel("Variable socioeconómica")
    ax.set_title("Dependencia socioeconómica del rendimiento académico")
    ax.legend()
    plt.tight_layout()
    return fig


# 11) Top 10 deptos por casos (impacto)
def fig_top10_impact(kpi_dept, top_n=10):
    top = kpi_dept.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index, top["casos"], color="lightgray")

    ax.barh(top.index[0], top["casos"].iloc[0], color="#C62828", label="Mayor impacto")
    ax.set_xlabel("Estudiantes en bajo rendimiento")
    ax.set_title(f"Top {top_n} departamentos por impacto absoluto")
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    return fig


# 12) Bubble impacto vs riesgo (resaltando mayor impacto)
def fig_impact_risk_bubble(kpi_dept, p_nat=None, top_n=40):
    dfb = kpi_dept.sort_values("casos", ascending=False).head(top_n).copy()
    x = dfb["p_bajo"].values
    y = dfb["casos"].values
    sizes = (dfb["N"].values / dfb["N"].max()) * 1200 + 50
    top_dept = dfb.index[0]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(x, y, s=sizes, color="0.78", alpha=0.6, edgecolor="white", linewidth=0.6, zorder=2)

    ax.scatter(
        dfb.loc[top_dept, "p_bajo"],
        dfb.loc[top_dept, "casos"],
        s=(dfb.loc[top_dept, "N"] / dfb["N"].max()) * 1200 + 50,
        color="#E57373", alpha=0.9, edgecolor="white", linewidth=0.9, zorder=3
    )

    ax.annotate(
        str(top_dept),
        xy=(dfb.loc[top_dept, "p_bajo"], dfb.loc[top_dept, "casos"]),
        xytext=(10, 0),
        textcoords="offset points",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="black"
    )

    if p_nat is not None:
        ax.axvline(p_nat, linestyle="--", linewidth=1.5, color="0.35", label="Promedio nacional (riesgo)")
        ax.legend(frameon=False)

    ax.set_xlabel("Riesgo: P(bajo)")
    ax.set_ylabel("Impacto: casos (bajo rendimiento)")
    ax.set_title(f"Priorización territorial: mayor impacto resaltado (Top {top_n})")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, linestyle="--", alpha=0.25)

    plt.tight_layout()
    return fig


# 13) Impacto absoluto por región (casos)
def fig_region_impact_cases(kpi_region):
    kpi_region_plot = kpi_region.sort_values("casos", ascending=False)
    labels = kpi_region_plot.index.tolist()
    values = kpi_region_plot["casos"].values

    best = kpi_region_plot["casos"].idxmax()
    worst = kpi_region_plot["casos"].idxmin()

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, values, color="lightgray")

    best_idx = labels.index(best)
    worst_idx = labels.index(worst)

    ax.bar(best_idx, values[best_idx], color="#C62828", label="Mayor impacto")
    ax.bar(worst_idx, values[worst_idx], color="#2E7D32", label="Menor impacto")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Número de estudiantes en bajo rendimiento")
    ax.set_title("Impacto absoluto del bajo rendimiento por región")
    ax.legend()
    plt.tight_layout()
    return fig


# 14) Segmentos críticos (barras)
def fig_segments_risk(kpi_segments):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(kpi_segments["segmento"], kpi_segments["p_bajo"], color="lightgray")

    idx = kpi_segments["p_bajo"].idxmax()
    ax.bar(kpi_segments.loc[idx, "segmento"], kpi_segments.loc[idx, "p_bajo"], color="#C62828", label="Mayor riesgo")

    ax.set_ylabel("Proporción de bajo rendimiento")
    ax.set_title("Riesgo de bajo rendimiento por segmento de doble vulnerabilidad")
    ax.tick_params(axis='x', rotation=20)
    ax.legend()
    plt.tight_layout()
    return fig


# 15) Desigualdad interna por región (brecha)
def fig_region_internal_inequality(ineq):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ineq.index, ineq["brecha_max_min_pp"], color="lightgray")
    worst = ineq["brecha_max_min_pp"].idxmax()
    ax.bar(worst, ineq.loc[worst, "brecha_max_min_pp"], color="red", label="Mayor desigualdad interna")
    ax.set_ylabel("Brecha interna de riesgo (pp)")
    ax.set_title("Desigualdad del bajo rendimiento dentro de cada región")
    ax.legend()
    plt.tight_layout()
    return fig


# 16) Rango min–max por región (líneas)
def fig_region_minmax(kpi_ineq_region, highlight="PACIFICA"):
    fig, ax = plt.subplots(figsize=(9, 5))
    for region in kpi_ineq_region.index:
        ax.plot([region, region],
                [kpi_ineq_region.loc[region, "min_"]*100, kpi_ineq_region.loc[region, "max_"]*100],
                linewidth=8, color="lightgray")

    if highlight in kpi_ineq_region.index:
        ax.plot([highlight, highlight],
                [kpi_ineq_region.loc[highlight, "min_"]*100, kpi_ineq_region.loc[highlight, "max_"]*100],
                linewidth=8, color="#C62828", label=highlight)

    ax.set_ylabel("Riesgo de bajo rendimiento (%)")
    ax.set_title("Rango de riesgo por región (mín–máx por departamento)")
    ax.legend()
    plt.tight_layout()
    return fig


# 17) Palancas: casos evitables (proxy) y deltas (pp)
def fig_palancas_cases(palancas, casos):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(palancas, casos, color="lightgray")
    max_idx = int(np.argmax(casos))
    ax.bar(palancas[max_idx], casos[max_idx], color="#2E7D32", label="Mayor impacto potencial")
    ax.set_ylabel("Casos de bajo rendimiento evitables (proxy)")
    ax.set_title("Impacto potencial de palancas de política pública")
    ax.legend()
    plt.tight_layout()
    return fig

def fig_palancas_deltas(palancas, deltas):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(palancas, deltas, color="lightgray")
    max_idx = int(np.argmax(deltas))
    ax.bar(palancas[max_idx], deltas[max_idx], color="#2E7D32", label="Mayor reducción de riesgo")
    ax.set_ylabel("Reducción de riesgo de bajo rendimiento (pp)")
    ax.set_title("Brecha de riesgo asociada a cada palanca")
    ax.legend()
    plt.tight_layout()
    return fig