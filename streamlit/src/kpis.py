# src/kpis.py
import numpy as np
import pandas as pd
from .data import cramers_v

KPI_TEXT = {
    "KRAS": """
**KPI 1 — KRAS: Riesgo Académico Social (brecha por estrato)**

**KRAS = P(Bajo | Estrato bajo) - P(Bajo | Estrato alto)**

Este KPI se define porque el estrato socioeconómico es uno de los determinantes estructurales más persistentes del desempeño educativo en contextos como el colombiano. El KRAS sintetiza, en una sola métrica, la brecha de probabilidad de bajo rendimiento entre estudiantes de estratos bajos y altos, expresada en puntos porcentuales.

Su valor radica en que permite responder, de manera clara y cuantificable: ¿cuánto mayor es el riesgo de bajo rendimiento que enfrentan los estudiantes de contextos desfavorecidos frente a aquellos de estratos altos?
""",
    "FD": """
**KPI 2 — FD: Fricción Digital (brecha por computador)**

**FD = P(Bajo | Sin computador) - P(Bajo | Con computador)**

Este KPI se define porque no contar con un computador marca una diferencia real en el desempeño académico de los estudiantes. Permite dimensionar cuánto mayor es el riesgo de bajo rendimiento para quienes carecen de este recurso, convirtiendo la brecha digital en un impacto fácil de entender para la toma de decisiones.
""",
    "KCC": """
**KPI 3 — KCC: Capital Cultural (educación materna)**

Este KPI mide la diferencia en el riesgo de bajo rendimiento académico entre estudiantes cuyas madres tienen bajos niveles educativos y aquellos cuyas madres cuentan con formación profesional o de posgrado.
""",
    "KDS": """
**KPI 4 — KDS: Dependencia socioeconómica del rendimiento (promedio Cramér’s V)**

Este KPI muestra qué tan fuertemente el desempeño académico está asociado a condiciones socioeconómicas clave (estrato, acceso a tecnología, educación parental o territorio). Ofrece un panorama de hasta qué punto el rendimiento está condicionado por factores estructurales.
""",
    "EQ_REG": """
**KPI 5 — Equidad regional (brecha de riesgo)**

Este KPI muestra la diferencia en el riesgo de bajo rendimiento entre regiones, ayudando a enfocar recursos en las zonas que realmente están quedando atrás.
""",
    "IMPACTO": """
**KPI 6 — Impacto absoluto (casos) y carga estructural por territorio**

Este KPI permite priorizar intervenciones por volumen de casos (impacto absoluto) y por exceso de riesgo vs promedio nacional (carga estructural).
"""
}

def compute_kpi_tables(data_imp: pd.DataFrame) -> dict:
    TARGET = "RENDIMIENTO_GLOBAL"
    out = {}

    if TARGET not in data_imp.columns:
        return out

    is_low = (data_imp[TARGET] == 0)
    p_nat = float(is_low.mean()) if len(data_imp) else np.nan
    out["p_nat"] = p_nat

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
            "KRAS_pp": [round(kras * 100, 6)] * 2
        })

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
            "FD_pp": [round(fd * 100, 6)] * 2
        })

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
            "KCC_pp": [round(kcc * 100, 6)] * 2
        })

    # KPI 4 - KDS
    socio_cols = [c for c in [
        "F_ESTRATOVIVIENDA", "F_TIENECOMPUTADOR", "F_TIENEINTERNET", "F_TIENEAUTOMOVIL",
        "F_EDUCACIONMADRE", "F_EDUCACIONPADRE", "REGION", "E_PRGM_DEPARTAMENTO"
    ] if c in data_imp.columns]

    cv = []
    rows = []
    for c in socio_cols:
        try:
            v = cramers_v(data_imp[c], data_imp[TARGET])
            if not np.isnan(v):
                cv.append(v)
                rows.append((c, v))
        except Exception:
            pass

    kds = float(np.mean(cv)) if cv else np.nan
    out["KDS"] = kds
    out["KDS_table"] = (
        pd.DataFrame(rows, columns=["Variable", "Cramér’s V"])
        .sort_values("Cramér’s V", ascending=False)
    )
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

    # KPI 6 - Impacto territorio
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
