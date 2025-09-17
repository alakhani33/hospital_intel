# caliber_utils/kpis.py
from typing import Optional, List
import numpy as np
import pandas as pd

# ---------- helpers ----------
def _coerce_numeric(x):
    if isinstance(x, (pd.Series, pd.Index)):
        return pd.to_numeric(x, errors="coerce")
    try:
        return float(x)
    except Exception:
        return np.nan

def safe_div(num, den):
    a = _coerce_numeric(num); b = _coerce_numeric(den)
    if not isinstance(a, (pd.Series, pd.Index)) and not isinstance(b, (pd.Series, pd.Index)):
        if b is None or (isinstance(b, float) and (np.isnan(b) or b == 0.0)):
            return np.nan
        return a / b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(a, b)
    if isinstance(out, (pd.Series, pd.Index)):
        return out.replace([np.inf, -np.inf], np.nan)
    return np.nan if out is None or not np.isfinite(out) else float(out)

def sumcol(df, name):
    """
    Safe sum of a column or dict-like 'name'.
    Handles missing columns, Series, arrays, lists and scalars.
    """
    import numpy as np
    import pandas as pd

    if isinstance(df, pd.DataFrame) and name in df.columns:
        s = pd.to_numeric(df[name], errors="coerce")
        return float(np.nansum(s.to_numpy()))
    if isinstance(df, pd.Series) and df.name == name:
        s = pd.to_numeric(df, errors="coerce")
        return float(np.nansum(s.to_numpy()))
    try:
        arr = pd.to_numeric(df.get(name, 0), errors="coerce")
    except Exception:
        return 0.0
    if isinstance(arr, (pd.Series, pd.Index)):
        return float(np.nansum(arr.to_numpy()))
    if isinstance(arr, (list, tuple, np.ndarray)):
        return float(np.nansum(arr))
    try:
        val = float(arr)
        return 0.0 if (val != val) else val
    except Exception:
        return 0.0

# ---------- peer building ----------
def get_peer_frame(df: pd.DataFrame, selected: pd.DataFrame, scope: str,
                   require_same_care: bool = True,
                   require_same_control: bool = False,
                   comparability: str = "Comparable only",
                   extra_counties: Optional[List[str]] = None) -> pd.DataFrame:
    """
    scope: 'County', 'HSA', 'HFPA', 'System', 'Statewide'
    comparability: 'Comparable only' | 'Like with Like (by TYPE_HOSP)' | 'All hospitals'
    extra_counties: optional list of additional counties to include (County scope only)
    """
    if selected.empty:
        return df.copy()

    base = df.copy()

    if "TYPE_HOSP" in base.columns:
        if comparability == "Comparable only":
            base = base[base["TYPE_HOSP"] == "Comparable"]
        elif comparability == "Like with Like (by TYPE_HOSP)":
            base = base[base["TYPE_HOSP"] == selected["TYPE_HOSP"].iloc[0]]

    if require_same_care and "TYPE_CARE" in base.columns:
        base = base[base["TYPE_CARE"] == selected["TYPE_CARE"].iloc[0]]

    if require_same_control and "TYPE_CNTRL" in base.columns:
        base = base[base["TYPE_CNTRL"] == selected["TYPE_CNTRL"].iloc[0]]

    # Geography/System scope
    if scope == "County" and "COUNTY" in base.columns:
        sel_cty = selected["COUNTY"].iloc[0] if "COUNTY" in selected.columns and not selected.empty else None
        allowed = [sel_cty] if sel_cty else []
        if extra_counties:
            allowed += list(extra_counties)
        if allowed:
            base = base[base["COUNTY"].isin(allowed)]
    elif scope == "HSA" and "HSA" in base.columns:
        base = base[base["HSA"] == selected["HSA"].iloc[0]]
    elif scope == "HFPA" and "HFPA" in base.columns:
        base = base[base["HFPA"] == selected["HFPA"].iloc[0]]
    elif scope == "System" and "OWNER" in base.columns:
        base = base[base["OWNER"] == selected["OWNER"].iloc[0]]
    # else statewide

    return base.copy()

# ---------- financial performance ----------
def operating_margin(df: pd.DataFrame):
    op_rev = sumcol(df, "NET_PT_REV") + sumcol(df, "OTH_OP_REV") + sumcol(df, "TOT_CAP_REV")
    return float(safe_div(sumcol(df, "NET_FRM_OP"), op_rev))

def total_margin(df: pd.DataFrame):
    total_rev = (sumcol(df, "NET_PT_REV") + sumcol(df, "OTH_OP_REV") +
                 sumcol(df, "TOT_CAP_REV") + sumcol(df, "NONOP_REV"))
    return float(safe_div(sumcol(df, "NET_INCOME"), total_rev))

def npr_per_discharge(df: pd.DataFrame):
    return float(safe_div(sumcol(df, "NET_PT_REV"), sumcol(df, "DIS_TOT")))

def outpatient_share(df: pd.DataFrame):
    op = sumcol(df, "GR_OP_TOT"); ip = sumcol(df, "GR_IP_TOT")
    return float(safe_div(op, op + ip))

# ---------- access & throughput ----------
def occ_lic(df: pd.DataFrame):
    if "OCC_LIC" in df.columns and pd.to_numeric(df["OCC_LIC"], errors="coerce").notna().any():
        return float(pd.to_numeric(df["OCC_LIC"], errors="coerce").mean())
    day_per = float(pd.to_numeric(df.get("DAY_PER", 365), errors="coerce").iloc[0] if "DAY_PER" in df.columns else 365)
    denom = sumcol(df, "BED_LIC") * day_per
    return float(safe_div(sumcol(df, "DAY_TOT"), denom))

def alos_all(df: pd.DataFrame):
    if "ALOS_ALL" in df.columns and pd.to_numeric(df["ALOS_ALL"], errors="coerce").notna().any():
        return float(pd.to_numeric(df["ALOS_ALL"], errors="coerce").mean())
    return float(safe_div(sumcol(df, "DAY_TOT"), sumcol(df, "DIS_TOT")))

def ed_visits_per_1k_dis(df: pd.DataFrame):
    return float(safe_div(sumcol(df, "VIS_ER"), sumcol(df, "DIS_TOT")) * 1000.0)

def c_section_rate(df: pd.DataFrame):
    return float(safe_div(sumcol(df, "C_SECTIONS"), sumcol(df, "NAT_BIRTHS")))

# ---------- cost structure & productivity ----------
def expense_per_adjusted_pd(df: pd.DataFrame):
    ip = sumcol(df, "GR_IP_TOT"); allr = ip + sumcol(df, "GR_OP_TOT")
    adj_factor = safe_div(allr, ip) if ip > 0 else 1.0
    adj_days = sumcol(df, "DAY_TOT") * (adj_factor if adj_factor and adj_factor > 0 else 1.0)
    return float(safe_div(sumcol(df, "TOT_OP_EXP"), adj_days))

def labor_pct_of_op_revenue(df: pd.DataFrame):
    labor = sumcol(df, "EXP_SAL") + sumcol(df, "EXP_BEN") + sumcol(df, "EXP_PHYS") + sumcol(df, "EXP_OTHPRO")
    op_rev = sumcol(df, "NET_PT_REV") + sumcol(df, "OTH_OP_REV") + sumcol(df, "TOT_CAP_REV")
    return float(safe_div(labor, op_rev))

def rn_hppd(df: pd.DataFrame):
    return float(safe_div(sumcol(df, "PRD_HR_RN"), sumcol(df, "DAY_TOT")))

def paid_hppd(df: pd.DataFrame):
    return float(safe_div(sumcol(df, "PAID_HRS"), sumcol(df, "DAY_TOT")))

def contract_labor_share(df: pd.DataFrame):
    num = sumcol(df, "CNT_HR_RN") + sumcol(df, "CNT_HR_OTH")
    denom = sumcol(df, "PROD_HRS") if "PROD_HRS" in df.columns else sum(
        sumcol(df, c) for c in [
            "PRD_HR_MGT","PRD_HR_TCH","PRD_HR_RN","PRD_HR_LVN","PRD_HR_AID",
            "PRD_HR_CLR","PRD_HR_ENV","PRD_HR_OTH"
        ] if c in df.columns
    )
    return float(safe_div(num, denom))

# ---------- payer & visits mix ----------
def net_revenue_mix(df: pd.DataFrame) -> dict:
    parts = {
        "Medicare": sumcol(df,"NETRV_MCAR_TR") + sumcol(df,"NETRV_MCAR_MC"),
        "Medi-Cal": sumcol(df,"NETRV_MCAL_TR") + sumcol(df,"NETRV_MCAL_MC"),
        "County": sumcol(df,"NETRV_CNTY"),
        "Third": sumcol(df,"NETRV_THRD_TR") + sumcol(df,"NETRV_THRD_MC"),
        "Other": sumcol(df,"NETRV_OTH") if "NETRV_OTH" in df.columns else 0.0,
    }
    tot = sum(parts.values())
    return {k: (v / tot if tot > 0 else np.nan) for k, v in parts.items()}

def visit_mix(df: pd.DataFrame) -> dict:
    parts = {
        "ED": sumcol(df,"VIS_ER"),
        "Clinic": sumcol(df,"VIS_CLIN"),
        "Home": sumcol(df,"VIS_HOME"),
        "Other": sumcol(df,"VIS_OTH") if "VIS_OTH" in df.columns else 0.0,
    }
    tot = sum(parts.values())
    return {k: (v / tot if tot > 0 else np.nan) for k, v in parts.items()}

# ---------- peer stats ----------
def peer_stats(peers_df: pd.DataFrame, metric_func):
    if peers_df is None or peers_df.empty:
        return {"median": np.nan, "p25": np.nan, "p75": np.nan}
    vals = []
    if "FAC_NAME" in peers_df.columns:
        for _, g in peers_df.groupby("FAC_NAME"):
            try:
                v = metric_func(g)
            except Exception:
                v = np.nan
            if v is not None and pd.notna(v):
                vals.append(float(v))
    else:
        try:
            v = metric_func(peers_df)
            if v is not None and pd.notna(v):
                vals.append(float(v))
        except Exception:
            pass
    if not vals:
        return {"median": np.nan, "p25": np.nan, "p75": np.nan}
    s = pd.Series(vals)
    return {"median": float(s.median()), "p25": float(s.quantile(0.25)), "p75": float(s.quantile(0.75))}
