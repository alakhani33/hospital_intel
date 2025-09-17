# app.py — CALIBER360 Hospital Intelligence (fixed peer exclusion)

import streamlit as st
import pandas as pd
import numpy as np
import importlib
import re
from typing import Optional, List
import plotly.express as px


from caliber_utils.data_loader import load_workbook, period_overlap_mask
import caliber_utils.kpis as kpis
import caliber_utils.charts as charts

# hot-reload during dev
kpis = importlib.reload(kpis)
charts = importlib.reload(charts)

st.set_page_config(page_title="CALIBER360 Hospital Intelligence", layout="wide")

# ---------- Load workbook ----------
@st.cache_data
def get_data():
    # Update path if your workbook is elsewhere
    data, glossary, charts_sheet, profile = load_workbook("data/hafd2023pivot.xls")
    return data, glossary, charts_sheet, profile

df, glossary, charts_sheet, profile = get_data()

# Guard
if "FAC_NAME" not in df.columns:
    st.error("The workbook doesn't include a recognizable facility name column (e.g., FAC_NAME / Facility Name). Please check headers.")
    st.stop()

min_beg = pd.to_datetime(df.get("BEG_DATE"), errors="coerce").min()
max_end = pd.to_datetime(df.get("END_DATE"), errors="coerce").max()

# ---------- Regions for auto-neighboring ----------
def county_regions():
    return {
        "Bay Area": ["San Francisco","San Mateo","Santa Clara","Alameda","Contra Costa","Marin","Napa","Sonoma","Solano"],
        "North Bay": ["Sonoma","Marin","Napa","Solano","Lake"],
        "Central Coast": ["Santa Cruz","Monterey","San Benito","San Luis Obispo","Santa Barbara","Ventura"],
        "LA Metro": ["Los Angeles","Orange","Ventura"],
        "Inland Empire": ["San Bernardino","Riverside"],
        "San Diego-Imperial": ["San Diego","Imperial"],
        "San Joaquin Valley": ["San Joaquin","Stanislaus","Merced","Madera","Fresno","Kings","Tulare","Kern"],
        "Sacramento Area": ["Sacramento","Yolo","Placer","El Dorado"],
        "North Coast": ["Del Norte","Humboldt","Mendocino"],
        "Shasta Cascade": ["Siskiyou","Shasta","Trinity","Modoc","Lassen","Tehama","Plumas"],
        "Mother Lode / Sierra": ["Nevada","Placer","El Dorado","Amador","Calaveras","Tuolumne","Alpine","Mono","Inyo","Sierra"],
        "North Valley": ["Butte","Glenn","Colusa","Sutter","Yuba"],
        "Wine Country": ["Napa","Sonoma","Marin","Solano"],
    }

def suggested_neighbors_for(county_name: str, available: List[str]) -> List[str]:
    regs = county_regions()
    carrier = [r for r, lst in regs.items() if county_name in lst]
    suggested = set()
    for r in carrier:
        suggested.update(regs[r])
    suggested.discard(county_name)
    return sorted([c for c in suggested if c in available], key=lambda x: x.lower())

# ---------- Glossary helpers ----------
def _canon(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(s).upper().strip())

@st.cache_data
def _prepare_glossary(glossary_df: Optional[pd.DataFrame]):
    if glossary_df is None or len(glossary_df) == 0:
        return None
    g = glossary_df.copy()
    g.columns = [str(c).strip() for c in g.columns]

    def _pick(cols, want):
        want_set = {
            "TERM","TERMS","FIELD","VARIABLE","VARNAME","COLUMN","COLUMNNAME","METRIC","ITEM"
        } if want == "term" else {
            "DEFINITION","DEFINITIONS","DESCRIPTION","DESC","MEANING","NOTES","NOTE","DETAILS"
        }
        cands = [c for c in cols if _canon(c) in want_set]
        if cands:
            return cands[0]
        return (max(cols, key=lambda c: g[c].astype(str).map(len).mean())
                if want != "term" else cols[0])

    term_col = _pick(g.columns, "term")
    def_col  = _pick(g.columns, "def")

    tidy = g[[term_col, def_col]].rename(columns={term_col: "TERM", def_col: "DEFINITION"}).copy()
    tidy["TERM"] = tidy["TERM"].astype("string").str.strip()
    tidy["DEFINITION"] = tidy["DEFINITION"].astype("string")
    tidy["CANON"] = tidy["TERM"].map(_canon)
    tidy = tidy[tidy["CANON"] != ""].drop_duplicates(subset=["CANON"])
    return tidy.reset_index(drop=True)

GLOSS = _prepare_glossary(glossary)

GLOSS_ALIASES = {
    "ALOS_ALL": ["Average Length Of Stay", "ALOS", "ALOS All", "Average Length Of Stay (All)"],
    "OCC_LIC": ["Occupancy", "Occupancy (Licensed)", "Licensed Bed Occupancy", "Occupancy Rate"],
    "NET_PT_REV": ["Net Patient Revenue", "NPR"],
    "OTH_OP_REV": ["Other Operating Revenue"],
    "TOT_CAP_REV": ["Total Capitation Revenue", "Capitation Revenue"],
    "NONOP_REV": ["Nonoperating Revenue", "Non-Operating Revenue"],
    "NET_FRM_OP": ["Operating Income", "Net From Operations", "Operating Gain (Loss)"],
    "DIS_TOT": ["Total Discharges", "Discharges"],
    "DAY_TOT": ["Total Patient Days", "Patient Days", "Inpatient Days"],
    "BED_LIC": ["Licensed Beds", "Beds - Licensed"],
    "DAY_PER": ["Days In Period", "Reporting Days"],
    "GR_OP_TOT": ["Gross Outpatient Revenue", "Outpatient Gross Revenue"],
    "GR_IP_TOT": ["Gross Inpatient Revenue", "Inpatient Gross Revenue"],
    "EXP_SAL": ["Salaries Expense", "Wages", "Salary Expense"],
    "EXP_BEN": ["Employee Benefits", "Benefits Expense"],
    "EXP_PHYS": ["Physician Expense"],
    "EXP_OTHPRO": ["Other Professional Fees", "Other Professional Services"],
    "PAID_HRS": ["Paid Hours"],
    "PRD_HR_RN": ["RN Productive Hours", "RN Hours"],
    "CNT_HR_RN": ["Contract RN Hours", "Agency RN Hours"],
    "CNT_HR_OTH": ["Contract Other Hours", "Agency Other Hours"],
    "PROD_HRS": ["Productive Hours", "Total Productive Hours"],
    "C_SECTIONS": ["C-Sections", "Cesarean Sections", "Cesarean Deliveries"],
    "NAT_BIRTHS": ["Live Births", "Births", "Nat Births"],
    "VIS_ER": ["ED Visits", "Emergency Department Visits"],
    "VIS_CLIN": ["Clinic Visits", "Outpatient Clinic Visits"],
    "VIS_HOME": ["Home Health Visits", "Home Visits"],
    "NETRV_MCAR_TR": ["Net Revenue Medicare (Traditional)", "Net Rev Medicare FFS"],
    "NETRV_MCAR_MC": ["Net Revenue Medicare (Managed Care)"],
    "NETRV_MCAL_TR": ["Net Revenue Medi-Cal (Traditional)", "Net Rev Medicaid FFS"],
    "NETRV_MCAL_MC": ["Net Revenue Medi-Cal (Managed Care)"],
    "NETRV_CNTY": ["Net Revenue County"],
    "NETRV_THRD_TR": ["Net Revenue Commercial (Traditional)", "Net Revenue Third (Traditional)"],
    "NETRV_THRD_MC": ["Net Revenue Commercial (Managed Care)", "Net Revenue Third (Managed Care)"],
    "NETRV_OTH": ["Net Revenue Other"],
    "CUR_ASST": ["Current Assets"],
    "CUR_LIAB": ["Current Liabilities"],
    "CASH": ["Cash And Cash Equivalents", "Cash"],
    "TOT_LTDEBT": ["Total Long-Term Debt", "Long-Term Debt"],
    "EQUITY": ["Net Assets", "Equity", "Fund Balance"],
    "ACC_DEPRE": ["Accumulated Depreciation"],
    "EXP_DEPRE": ["Depreciation Expense"],
    "ACCTS_REC": ["Accounts Receivable", "Net Patient Accounts Receivable", "A/R"],
}

def _find_gloss_row(term: str):
    if GLOSS is None:
        return None
    canon = _canon(term)
    row = GLOSS.loc[GLOSS["CANON"] == canon]
    if not row.empty:
        return row.iloc[0]
    row = GLOSS.loc[GLOSS["CANON"] == _canon(term.replace("_", " "))]
    if not row.empty:
        return row.iloc[0]
    for alt in GLOSS_ALIASES.get(canon, []):
        row = GLOSS.loc[GLOSS["CANON"] == _canon(alt)]
        if not row.empty:
            return row.iloc[0]
    return None

def lookup_definitions(fields: List[str], show_missing: bool = False) -> pd.DataFrame:
    rows = []
    for f in fields:
        r = _find_gloss_row(f)
        if r is None:
            if show_missing:
                rows.append({"Field": f, "Definition": "(not found in Glossary)"})
        else:
            rows.append({"Field": r["TERM"], "Definition": r["DEFINITION"]})
    return pd.DataFrame(rows)

def definitions_expander(section_title: str, formulas: list[str], fields: list[str], show_missing: bool = False):
    with st.expander("📖 Definitions"):
        if formulas:
            st.markdown("**Formulas shown in this section**")
            for f in formulas:
                st.markdown(f"- {f}")
        defs = lookup_definitions(sorted(set(fields)), show_missing=show_missing)
        if not defs.empty:
            st.markdown("**Field definitions (from Glossary)**")
            st.dataframe(defs, use_container_width=True, hide_index=True)
        # else: stay silent (no 'not found' message)

# ---- Which fields each section uses (for the expander) -----------------------
SECTION_DOCS = {
    "Executive Overview": {
        "formulas": [
            "Operating Margin = NET_FRM_OP / (NET_PT_REV + OTH_OP_REV + TOT_CAP_REV)",
            "Total Margin = NET_INCOME / (NET_PT_REV + OTH_OP_REV + TOT_CAP_REV + NONOP_REV)",
            "NPR/Discharge = NET_PT_REV / DIS_TOT",
            "Occupancy ≈ DAY_TOT / (BED_LIC × DAY_PER)  (or OCC_LIC if present)",
            "ALOS = DAY_TOT / DIS_TOT",
        ],
        "fields": [
            "NET_FRM_OP","NET_PT_REV","OTH_OP_REV","TOT_CAP_REV","NONOP_REV","NET_INCOME",
            "DIS_TOT","DAY_TOT","BED_LIC","DAY_PER","OCC_LIC","ALOS_ALL"
        ],
    },
    "Payer & Revenue Mix": {
        "formulas": ["Net Revenue Mix = share of {Medicare, Medi-Cal, County, Third, Other} of total net revenue"],
        "fields": [
            "NETRV_MCAR_TR","NETRV_MCAR_MC","NETRV_MCAL_TR","NETRV_MCAL_MC",
            "NETRV_CNTY","NETRV_THRD_TR","NETRV_THRD_MC","NETRV_OTH"
        ],
    },
    "Cost & Productivity": {
        "formulas": [
            "Expense per Adjusted Patient Day ≈ TOT_OP_EXP / (DAY_TOT × ( (GR_IP_TOT + GR_OP_TOT) / GR_IP_TOT ))",
            "Labor % of Op Revenue = (EXP_SAL + EXP_BEN + EXP_PHYS + EXP_OTHPRO) / (NET_PT_REV + OTH_OP_REV + TOT_CAP_REV)",
            "RN HPPD = PRD_HR_RN / DAY_TOT",
            "Paid HPPD = PAID_HRS / DAY_TOT",
            "Contract Labor Share = (CNT_HR_RN + CNT_HR_OTH) / PROD_HRS",
        ],
        "fields": [
            "TOT_OP_EXP","DAY_TOT","GR_IP_TOT","GR_OP_TOT",
            "EXP_SAL","EXP_BEN","EXP_PHYS","EXP_OTHPRO","NET_PT_REV","OTH_OP_REV","TOT_CAP_REV",
            "PRD_HR_RN","PAID_HRS","CNT_HR_RN","CNT_HR_OTH","PROD_HRS",
            "PRD_HR_MGT","PRD_HR_TCH","PRD_HR_LVN","PRD_HR_AID","PRD_HR_CLR","PRD_HR_ENV","PRD_HR_OTH",
        ],
    },
    "Access & Throughput": {
        "formulas": [
            "ED per 1,000 Discharges = (VIS_ER / DIS_TOT) × 1000",
            "ALOS vs Occupancy: review ALOS_ALL vs OCC_LIC snapshot across peers",
        ],
        "fields": [
            "VIS_ER","DIS_TOT","ALOS_ALL","OCC_LIC","DAY_TOT","BED_LIC","DAY_PER",
            "SURG_OP","SURG_IP","OP_MIN_IP","OP_MIN_OP"
        ],
    },
    "Service Lines": {
        "formulas": ["C-Section Rate = C_SECTIONS / NAT_BIRTHS", "Visit Mix share across ED/Clinic/Home/Other"],
        "fields": ["C_SECTIONS","NAT_BIRTHS","VIS_ER","VIS_CLIN","VIS_HOME","VIS_OTH"],
    },
    "Liquidity & Capital": {
        "formulas": [
            "Current Ratio = CUR_ASST / CUR_LIAB",
            "Days Cash (simple) ≈ CASH / (TOT_OP_EXP / DAY_PER)",
            "Debt to Capital = TOT_LTDEBT / (TOT_LTDEBT + EQUITY)",
            "Avg Age of Plant ≈ ACC_DEPRE / EXP_DEPRE",
            "Days in A/R (simple) ≈ ACCTS_REC / (NET_PT_REV / DAY_PER)",
        ],
        "fields": ["CUR_ASST","CUR_LIAB","CASH","TOT_OP_EXP","DAY_PER","TOT_LTDEBT","EQUITY","ACC_DEPRE","EXP_DEPRE","ACCTS_REC","NET_PT_REV"],
    },
    "Leaderboard & Compare": {
        "formulas": ["All KPIs above used for ranking and comparison."],
        "fields": [
            "NET_FRM_OP","NET_PT_REV","OTH_OP_REV","TOT_CAP_REV","NONOP_REV","NET_INCOME",
            "DIS_TOT","DAY_TOT","BED_LIC","DAY_PER","OCC_LIC","ALOS_ALL",
            "GR_IP_TOT","GR_OP_TOT","TOT_OP_EXP",
            "EXP_SAL","EXP_BEN","EXP_PHYS","EXP_OTHPRO",
            "PRD_HR_RN","PAID_HRS","CNT_HR_RN","CNT_HR_OTH","PROD_HRS",
            "VIS_ER","C_SECTIONS","NAT_BIRTHS"
        ],
    },
}

SECTION_DOCS["Ownership Models"] = {
    "formulas": [
        "Group medians by TYPE_CNTRL for ratio metrics (per facility)",
        "Operating Margin = NET_FRM_OP / (NET_PT_REV + OTH_OP_REV + TOT_CAP_REV)",
        "Total Margin = NET_INCOME / (NET_PT_REV + OTH_OP_REV + TOT_CAP_REV + NONOP_REV)",
        "Expense per Adjusted Patient Day ≈ TOT_OP_EXP / (DAY_TOT × ((GR_IP_TOT + GR_OP_TOT) / GR_IP_TOT))",
        "Labor % of Op Revenue = (EXP_SAL + EXP_BEN + EXP_PHYS + EXP_OTHPRO) / (NET_PT_REV + OTH_OP_REV + TOT_CAP_REV)",
        "RN HPPD = PRD_HR_RN / DAY_TOT; Paid HPPD = PAID_HRS / DAY_TOT",
        "Occupancy ≈ DAY_TOT / (BED_LIC × DAY_PER) (or OCC_LIC if present); ALOS = DAY_TOT / DIS_TOT",
        "Net Income (Total) = sum of NET_INCOME across facilities in each TYPE_CNTRL",
    ],
    "fields": [
        "TYPE_CNTRL","NET_FRM_OP","NET_PT_REV","OTH_OP_REV","TOT_CAP_REV","NONOP_REV","NET_INCOME",
        "TOT_OP_EXP","GR_IP_TOT","GR_OP_TOT","DAY_TOT","DIS_TOT","BED_LIC","DAY_PER",
        "EXP_SAL","EXP_BEN","EXP_PHYS","EXP_OTHPRO","PRD_HR_RN","PAID_HRS","OCC_LIC","ALOS_ALL"
    ],
}

def render_page_header(active_section: str):
    """Top-of-page title + context line. Ownership is generic; other tabs show the hospital."""
    # Build a concise window string
    if pd.notna(min_beg) and pd.notna(max_end):
        window_text = f"Window: **{start_ts.date()} → {end_ts.date()}**"
    else:
        window_text = "Window: **All available data**"

    # Geo text
    if county_exists and county != "(All Counties)":
        if "Ownership" in active_section:
            # Ownership tab is generic; just indicate geo scope if user limited it elsewhere
            geo_bit = f"County: **{county}**" + (f"; Neighbors: {', '.join(extra_counties)}" if extra_counties else "")
        else:
            geo_bit = f"County: **{county}**" + (f"; Neighbors: {', '.join(extra_counties)}" if extra_counties else "")
    else:
        geo_bit = "County: **All**"

    # Common bits
    bits = [
        window_text,
        f"Peers: **{scope_used}**",
        ("Same Care" if same_care else None),
        ("Same Ownership" if same_own else None),
        comparability,
        geo_bit,
    ]
    bits = [b for b in bits if b]

    # Title logic
    if active_section == "Ownership Models":
        st.title("High-value benchmarking of hospital performance by ownership model")
    else:
        st.title(f"High-value benchmarking for {hospital} — {scope_used} peers")

    st.caption(" • ".join(bits))


# ---------- Sidebar filters ----------
st.sidebar.title("Filters")

# use_full = st.sidebar.checkbox("Use full reporting range", value=True)
# if use_full or pd.isna(min_beg) or pd.isna(max_end):
#     start_ts = min_beg if pd.notna(min_beg) else pd.Timestamp("1900-01-01")
#     end_ts   = max_end if pd.notna(max_end) else pd.Timestamp("2100-01-01")
# else:
#     start_date, end_date = st.sidebar.date_input(
#         "Reporting period",
#         value=(min_beg.date(), max_end.date()),
#         min_value=min_beg.date(), max_value=max_end.date()
#     )
#     start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
# --- Reporting period (no checkbox; defaults to full range) ---
# 

# --- Time window: always use full available range (no UI) ---
min_beg = pd.to_datetime(df.get("BEG_DATE"), errors="coerce").min()
max_end = pd.to_datetime(df.get("END_DATE"), errors="coerce").max()

# Fallbacks in case BEG/END are missing
start_ts = min_beg if pd.notna(min_beg) else pd.Timestamp("1900-01-01")
end_ts   = max_end if pd.notna(max_end) else pd.Timestamp("2100-01-01")

mask_window = (
    period_overlap_mask(df, start_ts, end_ts)
    if "BEG_DATE" in df.columns and "END_DATE" in df.columns
    else pd.Series(True, index=df.index)
)


county_exists = "COUNTY" in df.columns
county_universe = []
if county_exists:
    county_universe = (
        df["COUNTY"].astype("string")
        .replace(r"^\s*$", pd.NA, regex=True)
        .replace({"Unknown": pd.NA})
        .dropna()
        .unique()
        .tolist()
    )
    county_universe = sorted(county_universe, key=lambda x: x.lower())

county = st.sidebar.selectbox(
    "Select County",
    ["(All Counties)"] + (county_universe if county_exists else []),
    help=None if county_exists else "County column not found; showing all counties."
)

# Auto-neighbor suggestion (region-based)
# extra_counties = []
# if county_exists and county != "(All Counties)":
#     default_neighbors = suggested_neighbors_for(county, county_universe)
#     extra_counties = st.sidebar.multiselect(
#         "Include neighboring/additional counties",
#         options=[c for c in county_universe if c != county],
#         default=default_neighbors,
#         help="Suggested neighbors come from the same region; edit as needed."
#     )
# --- County neighbors picker (free-form) ---
# --- County neighbors picker (free-form) ---
extra_counties = []
if county_exists:
    # Region-based suggestions if a specific county is chosen
    suggested = suggested_neighbors_for(county, county_universe) if county != "(All Counties)" else []
    extra_counties = st.sidebar.multiselect(
        "Include counties (in addition to selected)",
        options=county_universe,              # allow ANY county to be added/removed
        default=suggested,                    # prefill with suggested neighbors when applicable
        help="Peers will include your selected county plus these. Add/remove freely."
    )



# Hospital list (optionally filtered by county)
if county_exists and county != "(All Counties)":
    hosp_opts = (
        df[df["COUNTY"] == county]["FAC_NAME"]
        .astype("string").dropna().unique().tolist()
    )
else:
    hosp_opts = df["FAC_NAME"].astype("string").dropna().unique().tolist()
hosp_opts = sorted(hosp_opts, key=lambda x: x.lower())
hospital = st.sidebar.selectbox("Select Hospital", hosp_opts if hosp_opts else ["(No hospitals found)"])

# Peer rules
st.sidebar.markdown("---")
st.sidebar.subheader("Peer Definition")
scope = st.sidebar.selectbox("Geographic scope", ["County","HSA","HFPA","System","Statewide"], index=0)
comparability = st.sidebar.selectbox("Comparability", ["Comparable only","Like with Like (by TYPE_HOSP)","All hospitals"], index=0)
same_care = st.sidebar.checkbox("Same Type of Care", value=True)
same_own = st.sidebar.checkbox("Same Ownership/Control", value=False)

# # ---------- Selected, peers, window ----------
# mask_window = period_overlap_mask(df, start_ts, end_ts) if "BEG_DATE" in df.columns and "END_DATE" in df.columns else pd.Series(True, index=df.index)
# selected = df[mask_window & (df["FAC_NAME"] == hospital)]

# peers = kpis.get_peer_frame(
#     df[mask_window],
#     selected,
#     scope,
#     require_same_care=same_care,
#     require_same_control=same_own,
#     comparability=comparability,
#     extra_counties=extra_counties if (county_exists and scope == "County") else None
# )

# # STRICT exclusion of the selected hospital from peer stats (no fallback)
# def peers_excluding_selected(peers_df: pd.DataFrame, hospital_name: str) -> pd.DataFrame:
#     if peers_df is None or peers_df.empty or "FAC_NAME" not in peers_df.columns:
#         return peers_df.iloc[0:0] if isinstance(peers_df, pd.DataFrame) else peers_df
#     return peers_df[peers_df["FAC_NAME"] != hospital_name]

# PEERS_NO_SELF = peers_excluding_selected(peers, hospital)
# peer_count = PEERS_NO_SELF["FAC_NAME"].nunique() if ("FAC_NAME" in PEERS_NO_SELF.columns and not PEERS_NO_SELF.empty) else 0

# st.title(f"🏥 {hospital}")
# st.caption(
#     f"Window: **{(start_ts.date() if pd.notna(min_beg) else 'All')} → {(end_ts.date() if pd.notna(max_end) else 'All')}** • "
#     f"Peers: **{scope}** • "
#     f"{'Same Care, ' if same_care else ''}{'Same Ownership, ' if same_own else ''}{comparability} • "
#     + (f"County: **{county}**; +{len(extra_counties)} neighbors" if county_exists and county != '(All Counties)' else "County: All")
# )
# if peer_count == 0:
#     st.info("No peer facilities found under the current filters (excluding the selected hospital). "
#             "Consider widening the scope, adding neighboring counties, or relaxing comparability filters.")
# # ---------- Selected, peers, window ----------
# mask_window = period_overlap_mask(df, start_ts, end_ts) if "BEG_DATE" in df.columns and "END_DATE" in df.columns else pd.Series(True, index=df.index)
# selected = df[mask_window & (df["FAC_NAME"] == hospital)]

# # Helper to build peers with optional extra counties
# def _build_peers(extra_c=None):
#     return kpis.get_peer_frame(
#         df[mask_window],
#         selected,
#         scope,
#         require_same_care=same_care,
#         require_same_control=same_own,
#         comparability=comparability,
#         extra_counties=extra_c if (county_exists and scope == "County") else None
#     )

# # Strictly exclude selected hospital from peer stats
# def _peers_excluding_selected(peers_df: pd.DataFrame, hospital_name: str) -> pd.DataFrame:
#     if isinstance(peers_df, pd.DataFrame) and ("FAC_NAME" in peers_df.columns):
#         return peers_df[peers_df["FAC_NAME"] != hospital_name]
#     return peers_df

# # 1) Initial peer set from current filters (may include user-chosen neighbors)
# peers = _build_peers(extra_counties)
# PEERS_NO_SELF = _peers_excluding_selected(peers, hospital)
# peer_count = PEERS_NO_SELF["FAC_NAME"].nunique() if ("FAC_NAME" in PEERS_NO_SELF.columns and not PEERS_NO_SELF.empty) else 0

# # 2) If NO peers in county, auto-include neighboring counties (region-based)
# auto_neighbors_used = []
# if peer_count == 0 and scope == "County" and county_exists and county != "(All Counties)":
#     # Suggested neighbors from region map
#     auto_neighbors = suggested_neighbors_for(county, county_universe)  # excludes the selected county
#     # Don’t duplicate any user-selected neighbors
#     base_set = set(extra_counties or [])
#     needed = [c for c in auto_neighbors if c not in base_set]
#     if needed:
#         # Rebuild peers with user-selected + auto neighbors
#         combined_neighbors = sorted(list(base_set.union(needed)))
#         peers_auto = _build_peers(combined_neighbors)
#         PEERS_NO_SELF = _peers_excluding_selected(peers_auto, hospital)
#         peer_count = PEERS_NO_SELF["FAC_NAME"].nunique() if ("FAC_NAME" in PEERS_NO_SELF.columns and not PEERS_NO_SELF.empty) else 0
#         if peer_count > 0:
#             auto_neighbors_used = needed
#             extra_counties = combined_neighbors  # reflect in the caption below

# # 3) Page header + context
# st.title(f"🏥 {hospital}")

# caption_bits = [
#     f"Window: **{(start_ts.date() if pd.notna(min_beg) else 'All')} → {(end_ts.date() if pd.notna(max_end) else 'All')}**",
#     f"Peers: **{scope}**",
#     ('Same Care' if same_care else None),
#     ('Same Ownership' if same_own else None),
#     comparability,
# ]
# caption_bits = [b for b in caption_bits if b]
# geo_bit = (
#     f"County: **{county}**"
#     if (county_exists and county != "(All Counties)")
#     else "County: All"
# )
# if county_exists and county != "(All Counties)":
#     # Show neighbors included (user-chosen and/or auto-added)
#     if extra_counties:
#         geo_bit += f"; Neighbors: {', '.join(extra_counties)}"
#     elif auto_neighbors_used:
#         geo_bit += f"; Auto neighbors: {', '.join(auto_neighbors_used)}"

# st.caption(" • ".join(caption_bits + [geo_bit]))

# # 4) Informational note only if still zero peers
# if peer_count == 0:
#     st.info(
#         "No peer facilities found under the current filters (excluding the selected hospital). "
#         "Try widening the scope (e.g., add neighbors or use HSA/HFPA/System/Statewide) or relax comparability."
#     )

# if selected.empty:
#     st.warning("No records for the selected hospital in the chosen period.")
#     st.stop()

# if selected.empty:
#     st.warning("No records for the selected hospital in the chosen period.")
#     st.stop()
# ---------- Selected, peers, window ----------
mask_window = (
    period_overlap_mask(df, start_ts, end_ts)
    if "BEG_DATE" in df.columns and "END_DATE" in df.columns
    else pd.Series(True, index=df.index)
)
selected = df[mask_window & (df["FAC_NAME"] == hospital)]

def _build_peers(scope_in: str, extra_c=None):
    return kpis.get_peer_frame(
        df[mask_window],
        selected,
        scope_in,
        require_same_care=same_care,
        require_same_control=same_own,
        comparability=comparability,
        extra_counties=extra_c if (county_exists and scope_in == "County") else None
    )

def _exclude_self(peers_df: pd.DataFrame, hospital_name: str) -> pd.DataFrame:
    if not isinstance(peers_df, pd.DataFrame) or peers_df.empty:
        return peers_df
    if "FAC_NAME" not in peers_df.columns:
        return peers_df
    return peers_df[peers_df["FAC_NAME"] != hospital_name]

def _unique_counties(series):
    return sorted(
        pd.Series(series).dropna().astype(str).unique().tolist(),
        key=lambda x: x.lower()
    )

# 1) Start with user picks
peers = _build_peers(scope, extra_counties)
PEERS_NO_SELF = _exclude_self(peers, hospital)

def _peer_fac_count(pdf):
    return pdf["FAC_NAME"].nunique() if (isinstance(pdf, pd.DataFrame) and "FAC_NAME" in pdf.columns and not pdf.empty) else 0

min_target = 3  # we aim for ≥3 peer facilities
peer_count = _peer_fac_count(PEERS_NO_SELF)

scope_used = scope
auto_neighbors_used = []
auto_level = None  # 'region', 'hsa', 'hfpa', 'state'

# 2) If County scope has too few peers, try region suggestions
if peer_count < min_target and scope == "County" and county_exists and county != "(All Counties)":
    region_suggestions = suggested_neighbors_for(county, county_universe)  # from our region map
    need = [c for c in region_suggestions if c not in (extra_counties or [])]
    if need:
        combined = sorted(list(set((extra_counties or []) + need)))
        peers2 = _build_peers("County", combined)
        no_self2 = _exclude_self(peers2, hospital)
        if _peer_fac_count(no_self2) > peer_count:
            peers, PEERS_NO_SELF = peers2, no_self2
            peer_count = _peer_fac_count(no_self2)
            auto_neighbors_used = need
            extra_counties = combined
            auto_level = "region"

# # 3) If still too few, include ALL counties in the SAME HSA as the selected hospital
sel_hsa_val = None
if "HSA" in selected.columns:
    # Always treat as a Series, coerce to numeric, take first non-null
    s_hsa = pd.to_numeric(selected["HSA"], errors="coerce")
    if s_hsa.notna().any():
        try:
            sel_hsa_val = int(s_hsa.dropna().iloc[0])
        except Exception:
            # If HSA is non-numeric codes in your data, fall back to string match
            sel_hsa_val = str(s_hsa.dropna().iloc[0])

if sel_hsa_val is not None and county_exists:
    # Build mask in df using the same dtype logic
    if isinstance(sel_hsa_val, int):
        hsa_mask = (pd.to_numeric(df.get("HSA"), errors="coerce") == sel_hsa_val)
    else:
        hsa_mask = (df.get("HSA").astype("string") == str(sel_hsa_val))

    hsa_mask = (mask_window & hsa_mask) if isinstance(mask_window, pd.Series) else hsa_mask
    hsa_counties = _unique_counties(df.loc[hsa_mask, "COUNTY"])
    hsa_neighbors = [c for c in hsa_counties if (county_exists and c != county)]
    need_hsa = [c for c in hsa_neighbors if c not in (extra_counties or [])]
    if need_hsa:
        combined_hsa = sorted(list(set((extra_counties or []) + need_hsa)))
        peers3 = _build_peers("County", combined_hsa)
        no_self3 = _exclude_self(peers3, hospital)
        if _peer_fac_count(no_self3) > peer_count:
            peers, PEERS_NO_SELF = peers3, no_self3
            peer_count = _peer_fac_count(no_self3)
            auto_neighbors_used = (
                need_hsa if not auto_neighbors_used
                else sorted(list(set(auto_neighbors_used + need_hsa)))
            )
            extra_counties = combined_hsa
            auto_level = "hsa"

# 4) If still too few, widen scope → HFPA (ignore county extras)
if peer_count < min_target:
    peers4 = _build_peers("HFPA", None)
    no_self4 = _exclude_self(peers4, hospital)
    if _peer_fac_count(no_self4) > peer_count:
        peers, PEERS_NO_SELF = peers4, no_self4
        peer_count = _peer_fac_count(no_self4)
        scope_used = "HFPA"
        auto_level = "hfpa"

# 5) If still too few, widen scope → Statewide
if peer_count < min_target:
    peers5 = _build_peers("Statewide", None)
    no_self5 = _exclude_self(peers5, hospital)
    if _peer_fac_count(no_self5) > peer_count:
        peers, PEERS_NO_SELF = peers5, no_self5
        peer_count = _peer_fac_count(no_self5)
        scope_used = "Statewide"
        auto_level = "state"

# ---------- Header / caption ----------
# st.title(f"🏥 {hospital}")

caption_bits = [
    # f"Window: **{(start_ts.date() if pd.notna(min_beg) else 'All')} → {(end_ts.date() if pd.notna(max_end) else 'All')}**",
    f"Peers: **{scope_used}**",
    ('Same Care' if same_care else None),
    ('Same Ownership' if same_own else None),
    comparability,
]
caption_bits = [b for b in caption_bits if b]

geo_bit = (
    f"County: **{county}**" if (county_exists and county != "(All Counties)") else "County: All"
)
if county_exists and county != "(All Counties)" and scope_used == "County":
    if extra_counties:
        geo_bit += f"; Neighbors: {', '.join(extra_counties)}"
    elif auto_neighbors_used:
        geo_bit += f"; Auto neighbors: {', '.join(auto_neighbors_used)}"

# If we auto-widened beyond County, note it.
if auto_level == "hfpa":
    geo_bit += " • Auto-expanded to **HFPA**"
elif auto_level == "state":
    geo_bit += " • Auto-expanded to **Statewide**"
elif auto_level == "hsa":
    geo_bit += " • Auto-included all **HSA** counties"

# st.caption(" • ".join(caption_bits + [geo_bit]))

# Informational note if STILL no peers
if peer_count == 0:
    st.info(
        "No peer facilities found under the current filters (excluding the selected hospital). "
        "Try widening the scope (e.g., add neighbors or use HSA/HFPA/System/Statewide) or relax comparability."
    )

if selected.empty:
    st.warning("No records for the selected hospital in the chosen period.")
    st.stop()

# ---------- Small utility for KPI blocks ----------
def kpi_block(title, func, units="", percent=False, expl=None):
    sel_val = func(selected)
    stats = kpis.peer_stats(PEERS_NO_SELF, func)  # <-- NEVER includes selected
    fig = charts.bar_selected_vs_peer(
        title, sel_val, stats["median"], units=units,
        p25=stats["p25"], p75=stats["p75"], percent=percent
    )
    st.plotly_chart(fig, use_container_width=True)
    if expl:
        for line in expl:
            st.caption(line)

# ---------- NAV + SECTIONS ----------
SECTIONS = [
    "Executive Overview",
    "Payer & Revenue Mix",
    "Cost & Productivity",
    "Access & Throughput",
    "Service Lines",
    "Liquidity & Capital",
    "Leaderboard & Compare",
]

def section_executive_overview():
    st.subheader("Executive Overview")
    col1, col2 = st.columns(2)
    with col1:
        kpi_block("Operating Margin", kpis.operating_margin, percent=True, expl=[
            "Operating Margin = NET_FRM_OP / (NET_PT_REV + OTH_OP_REV + TOT_CAP_REV)."
        ])
        kpi_block("Outpatient Revenue Share", kpis.outpatient_share, percent=True, expl=[
            "Outpatient Share = GR_OP_TOT / (GR_OP_TOT + GR_IP_TOT)."
        ])
        kpi_block("Occupancy (Licensed Beds)", kpis.occ_lic, units="ratio", expl=[
            "Occupancy = DAY_TOT / (BED_LIC × DAY_PER) or HCAI's OCC_LIC if present."
        ])
    with col2:
        kpi_block("Total Margin", kpis.total_margin, percent=True, expl=[
            "Total Margin = NET_INCOME / (Operating revenue + Non-operating revenue)."
        ])
        kpi_block("Net Patient Revenue per Discharge", kpis.npr_per_discharge, units="$ / discharge")
        kpi_block("Average Length of Stay (ALOS)", kpis.alos_all, units="days")
    definitions_expander(
        "Executive Overview",
        formulas=SECTION_DOCS["Executive Overview"]["formulas"],
        fields=SECTION_DOCS["Executive Overview"]["fields"],
    )

def section_payer_revenue_mix():
    st.subheader("Payer & Revenue Mix (Selected vs Peer Median)")
    sel_mix = kpis.net_revenue_mix(selected)
    if "FAC_NAME" in PEERS_NO_SELF.columns and not PEERS_NO_SELF.empty:
        pm = PEERS_NO_SELF.groupby("FAC_NAME").apply(kpis.net_revenue_mix).apply(pd.Series).median().to_dict()
    else:
        pm = {}  # no peer bar if no peers
    st.plotly_chart(charts.stacked_mix_chart("Net Revenue Mix by Payer", sel_mix, pm), use_container_width=True)
    definitions_expander(
        "Payer & Revenue Mix",
        formulas=SECTION_DOCS["Payer & Revenue Mix"]["formulas"],
        fields=SECTION_DOCS["Payer & Revenue Mix"]["fields"],
    )

def section_cost_productivity():
    st.subheader("Cost & Productivity")
    col1, col2 = st.columns(2)
    with col1:
        kpi_block("Expense per Adjusted Patient Day", kpis.expense_per_adjusted_pd, units="$ / adj day")
        kpi_block("Labor Cost % of Operating Revenue", kpis.labor_pct_of_op_revenue, percent=True)
    with col2:
        kpi_block("RN Hours per Patient Day", kpis.rn_hppd, units="hours / day")
        kpi_block("Total Paid Hours per Patient Day", kpis.paid_hppd, units="hours / day")
    kpi_block("Contract Labor Share of Productive Hours", kpis.contract_labor_share, percent=True)
    definitions_expander(
        "Cost & Productivity",
        formulas=SECTION_DOCS["Cost & Productivity"]["formulas"],
        fields=SECTION_DOCS["Cost & Productivity"]["fields"],
    )

def section_ownership_models():
    st.subheader("Ownership Models — Statewide medians by TYPE_CNTRL")

    # Toggle: limit to selected county + neighbors (optional)
    limit_geo = st.checkbox(
        "Limit to selected county + neighbors",
        value=False,
        help="Default is statewide; check to scope to your county + chosen neighbors."
    )

    # Base frame = entire window (statewide) or county+neighbors
    if limit_geo and "COUNTY" in df.columns and county_exists and county != "(All Counties)":
        allowed = {county, *extra_counties} if extra_counties else {county}
        df_scope = df[mask_window & df["COUNTY"].isin(allowed)]
    else:
        df_scope = df[mask_window]

    # Optional comparability shaping
    if "TYPE_HOSP" in df_scope.columns:
        if comparability == "Comparable only":
            df_scope = df_scope[df_scope["TYPE_HOSP"].astype("string").str.contains("Comparable", case=False, na=False)]
        elif comparability == "Like with Like (by TYPE_HOSP)":
            if "TYPE_HOSP" in selected.columns and selected["TYPE_HOSP"].notna().any():
                sel_th = selected["TYPE_HOSP"].astype("string").dropna().iloc[0]
                df_scope = df_scope[df_scope["TYPE_HOSP"].astype("string") == sel_th]
        # "All hospitals" → no filter

    # Per-facility KPIs (so group summaries are medians of facilities)
    rows = []
    if "FAC_NAME" in df_scope.columns:
        for name, g in df_scope.groupby("FAC_NAME"):
            type_cntrl = (
                g["TYPE_CNTRL"].astype("string").replace(r"^\s*$", pd.NA, regex=True).dropna().iloc[0]
                if "TYPE_CNTRL" in g.columns and g["TYPE_CNTRL"].notna().any() else "Unknown"
            )
            rows.append({
                "FAC_NAME": name,
                "TYPE_CNTRL": type_cntrl,
                "Op_Margin": kpis.operating_margin(g),
                "Total_Margin": kpis.total_margin(g),
                "Expense_per_APD": kpis.expense_per_adjusted_pd(g),
                "LaborPct_of_OpRev": kpis.labor_pct_of_op_revenue(g),
                "RN_HPPD": kpis.rn_hppd(g),
                "Paid_HPPD": kpis.paid_hppd(g),
                "Occ_Lic": kpis.occ_lic(g),
                "ALOS": kpis.alos_all(g),
                "Net_Income": kpis.sumcol(g, "NET_INCOME"),
            })
    fm = pd.DataFrame(rows).dropna(subset=["TYPE_CNTRL"])
    if fm.empty:
        st.info("No data available to compute ownership comparisons under current filters.")
        return

    # Summaries by ownership model
    summ = (fm
            .groupby("TYPE_CNTRL")
            .agg({
                "Op_Margin": "median",
                "Total_Margin": "median",
                "Expense_per_APD": "median",
                "LaborPct_of_OpRev": "median",
                "RN_HPPD": "median",
                "Paid_HPPD": "median",
                "Occ_Lic": "median",
                "ALOS": "median",
                "Net_Income": "sum"
            })
            .reset_index())

    # Helper for percent axis
    def _percent_axis(fig):
        fig.update_layout(yaxis_tickformat=".1%")
        return fig

    # Row 1: margins
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(summ, x="TYPE_CNTRL", y="Op_Margin", title="Median Operating Margin by Ownership")
        fig.update_traces(hovertemplate="Ownership: %{x}<br>Median Op Margin: %{y:.1%}<extra></extra>")
        st.plotly_chart(_percent_axis(fig), use_container_width=True)
    with c2:
        fig = px.bar(summ, x="TYPE_CNTRL", y="Total_Margin", title="Median Total Margin by Ownership")
        fig.update_traces(hovertemplate="Ownership: %{x}<br>Median Total Margin: %{y:.1%}<extra></extra>")
        st.plotly_chart(_percent_axis(fig), use_container_width=True)

    # Row 2: cost & labor intensity
    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(summ, x="TYPE_CNTRL", y="Expense_per_APD", title="Median Expense per Adjusted Patient Day")
        fig.update_traces(hovertemplate="Ownership: %{x}<br>Median $/Adj Day: $%{y:,.0f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        # rename just for plotting label clarity
        summ_lab = summ.rename(columns={"LaborPct_of_OpRev": "LaborPct"})
        fig = px.bar(summ_lab, x="TYPE_CNTRL", y="LaborPct", title="Labor Cost % of Operating Revenue (Median)")
        fig.update_traces(hovertemplate="Ownership: %{x}<br>Median Labor %: %{y:.1%}<extra></extra>")
        fig.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: staffing throughput
    c5, c6 = st.columns(2)
    with c5:
        fig = px.bar(summ, x="TYPE_CNTRL", y="RN_HPPD", title="RN Hours per Patient Day (Median)")
        fig.update_traces(hovertemplate="Ownership: %{x}<br>Median RN HPPD: %{y:.2f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        fig = px.bar(summ, x="TYPE_CNTRL", y="Paid_HPPD", title="Paid Hours per Patient Day (Median)")
        fig.update_traces(hovertemplate="Ownership: %{x}<br>Median Paid HPPD: %{y:.2f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    # Row 4: distributions
    st.markdown("#### Distribution of Occupancy & ALOS by Ownership (facility-level)")
    c7, c8 = st.columns(2)
    with c7:
        bx = px.box(fm.dropna(subset=["Occ_Lic"]), x="TYPE_CNTRL", y="Occ_Lic", points="outliers",
                    title="Occupancy (facility distribution)")
        bx.update_traces(hovertemplate="Ownership: %{x}<br>Occ (ratio): %{y:.2f}<extra></extra>")
        st.plotly_chart(bx, use_container_width=True)
    with c8:
        bx = px.box(fm.dropna(subset=["ALOS"]), x="TYPE_CNTRL", y="ALOS", points="outliers",
                    title="ALOS (facility distribution)")
        bx.update_traces(hovertemplate="Ownership: %{x}<br>ALOS (days): %{y:.2f}<extra></extra>")
        st.plotly_chart(bx, use_container_width=True)

    # Row 5: scale of model (sum of Net Income)
    st.markdown("#### Total Net Income by Ownership (sum across facilities)")
    fig = px.bar(summ.sort_values("Net_Income", ascending=False), x="TYPE_CNTRL", y="Net_Income",
                 title="Total Net Income by Ownership")
    fig.update_traces(hovertemplate="Ownership: %{x}<br>Total Net Income: $%{y:,.0f}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

    definitions_expander(
        "Ownership Models",
        formulas=SECTION_DOCS["Ownership Models"]["formulas"],
        fields=SECTION_DOCS["Ownership Models"]["fields"],
    )

    # # Definitions
    # definitions_expander(
    #     "Ownership Models",
    #     formulas=SECTION_DOCS["Ownership Models"]["formulas"],
    #     fields=SECTION_DOCS["Ownership Models"]["fields"],
    # )


# def section_access_throughput():
#     st.subheader("Access & Throughput")
#     kpi_block("ED Visits per 1,000 Discharges", kpis.ed_visits_per_1k_dis, units="per 1,000")

#     # ALOS vs OCC scatter (peers only)
#     perf = []
#     if "FAC_NAME" in PEERS_NO_SELF.columns and not PEERS_NO_SELF.empty:
#         for name, g in PEERS_NO_SELF.groupby("FAC_NAME"):
#             perf.append({
#                 "FAC_NAME": name,
#                 "COUNTY": g["COUNTY"].iloc[0] if "COUNTY" in g.columns else None,
#                 "OWNER": g["OWNER"].iloc[0] if "OWNER" in g.columns else None,
#                 "OCC_LIC": kpis.occ_lic(g),
#                 "ALOS_ALL": kpis.alos_all(g)
#             })
#     pfm = pd.DataFrame(perf).dropna(subset=["OCC_LIC","ALOS_ALL"])
#     if not pfm.empty:
#         st.plotly_chart(charts.scatter_alos_vs_occ(pfm, highlight_name=hospital), use_container_width=True)

#     # OR bubble (peers only)
#     if not PEERS_NO_SELF.empty:
#         st.plotly_chart(charts.bubble_or_util(PEERS_NO_SELF, highlight_name=hospital), use_container_width=True)
def section_access_throughput():
    st.subheader("Access & Throughput")
    kpi_block("ED Visits per 1,000 Discharges", kpis.ed_visits_per_1k_dis, units="per 1,000")

    # --- Build plotting frame that includes peers + selected (for *visuals only*) ---
    plot_base = PEERS_NO_SELF
    if isinstance(selected, pd.DataFrame) and not selected.empty:
        # Append selected hospital so it can be highlighted with a star on charts
        plot_base = pd.concat([PEERS_NO_SELF, selected], ignore_index=True)

    # ALOS vs OCC scatter (group by facility; compute snapshot metrics)
    perf_rows = []
    if isinstance(plot_base, pd.DataFrame) and not plot_base.empty and "FAC_NAME" in plot_base.columns:
        for name, g in plot_base.groupby("FAC_NAME"):
            perf_rows.append({
                "FAC_NAME": name,
                "COUNTY": g["COUNTY"].iloc[0] if "COUNTY" in g.columns else None,
                "OWNER": g["OWNER"].iloc[0] if "OWNER" in g.columns else None,
                "OCC_LIC": kpis.occ_lic(g),
                "ALOS_ALL": kpis.alos_all(g),
            })
    pfm = pd.DataFrame(perf_rows).dropna(subset=["OCC_LIC", "ALOS_ALL"])
    if not pfm.empty:
        st.plotly_chart(
            charts.scatter_alos_vs_occ(pfm, highlight_name=hospital),  # ⭐ now present
            use_container_width=True
        )

    # OR bubble (use peers + selected so selected shows as a star)
    if isinstance(plot_base, pd.DataFrame) and not plot_base.empty:
        st.plotly_chart(
            charts.bubble_or_util(plot_base, highlight_name=hospital),  # ⭐ now present
            use_container_width=True
        )

    definitions_expander(
        "Access & Throughput",
        formulas=SECTION_DOCS["Access & Throughput"]["formulas"],
        fields=SECTION_DOCS["Access & Throughput"]["fields"],
    )

    # definitions_expander(
    #     "Access & Throughput",
    #     formulas=SECTION_DOCS["Access & Throughput"]["formulas"],
    #     fields=SECTION_DOCS["Access & Throughput"]["fields"],
    # )

def section_service_lines():
    st.subheader("Service Lines")
    kpi_block("C-Section Rate (OB)", kpis.c_section_rate, percent=True)

    vm_sel = kpis.visit_mix(selected)
    if "FAC_NAME" in PEERS_NO_SELF.columns and not PEERS_NO_SELF.empty:
        vm_peer = PEERS_NO_SELF.groupby("FAC_NAME").apply(kpis.visit_mix).apply(pd.Series).median().to_dict()
    else:
        vm_peer = {}
    st.plotly_chart(charts.stacked_visit_mix("Visit Mix (Clinic vs Home vs ED)", vm_sel, vm_peer), use_container_width=True)
    definitions_expander(
        "Service Lines",
        formulas=SECTION_DOCS["Service Lines"]["formulas"],
        fields=SECTION_DOCS["Service Lines"]["fields"],
    )

def section_liquidity_capital():
    st.subheader("Liquidity & Capital (board-level)")

    def _get_day_per(df_):
        if "DAY_PER" in df_.columns:
            v = pd.to_numeric(df_["DAY_PER"], errors="coerce").dropna()
            if not v.empty:
                return float(v.iloc[0])
        return 365.0

    def _days_cash(df_):
        cash = kpis.sumcol(df_, "CASH")
        opex = kpis.sumcol(df_, "TOT_OP_EXP")
        day_per = _get_day_per(df_)
        daily_opex = kpis.safe_div(opex, day_per)
        return float(kpis.safe_div(cash, daily_opex))

    def _current_ratio(df_):
        cur_asst = kpis.sumcol(df_, "CUR_ASST")
        cur_liab = kpis.sumcol(df_, "CUR_LIAB")
        return float(kpis.safe_div(cur_asst, cur_liab))

    def _debt_to_cap(df_):
        ltd = kpis.sumcol(df_, "TOT_LTDEBT")
        eq  = kpis.sumcol(df_, "EQUITY")
        return float(kpis.safe_div(ltd, ltd + eq))

    def _avg_age_plant(df_):
        acc = kpis.sumcol(df_, "ACC_DEPRE")
        dep = kpis.sumcol(df_, "EXP_DEPRE")
        return float(kpis.safe_div(acc, dep))

    def _days_in_ar(df_):
        ar   = kpis.sumcol(df_, "ACCTS_REC")
        npr  = kpis.sumcol(df_, "NET_PT_REV")
        day_per = _get_day_per(df_)
        daily_npr = kpis.safe_div(npr, day_per)
        return float(kpis.safe_div(ar, daily_npr))

    metrics = [
        ("Current Ratio", _current_ratio(selected), "", False),
        ("Days Cash on Hand (simple)", _days_cash(selected), "days", False),
        ("Debt to Capital", _debt_to_cap(selected), "", True),
        ("Average Age of Plant (proxy)", _avg_age_plant(selected), "years", False),
        ("Days in A/R (simple)", _days_in_ar(selected), "days", False),
    ]
    table_rows = []
    for title, val, units, is_pct in metrics:
        def _f(df_):
            if title.startswith("Current Ratio"): return _current_ratio(df_)
            if title.startswith("Days Cash"): return _days_cash(df_)
            if title.startswith("Debt to Capital"): return _debt_to_cap(df_)
            if title.startswith("Average Age"): return _avg_age_plant(df_)
            if title.startswith("Days in A/R"): return _days_in_ar(df_)
        stats = kpis.peer_stats(PEERS_NO_SELF, _f)
        table_rows.append({
            "Metric": title,
            "Selected": f"{val*100:.1f}%" if (is_pct and pd.notna(val)) else (f"{val:,.2f}" if pd.notna(val) else ""),
            "Peer Median": f"{stats['median']*100:.1f}%" if (is_pct and pd.notna(stats['median'])) else (f"{stats['median']:,.2f}" if pd.notna(stats['median']) else ""),
            "Units": "%" if is_pct else units
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    definitions_expander(
        "Liquidity & Capital",
        formulas=SECTION_DOCS["Liquidity & Capital"]["formulas"],
        fields=SECTION_DOCS["Liquidity & Capital"]["fields"],
    )

def section_leaderboard_compare():
    st.subheader("Leaderboard (within selected window & peer rules)")
    base = PEERS_NO_SELF if ("FAC_NAME" in PEERS_NO_SELF.columns and not PEERS_NO_SELF.empty) else peers
    rows = []
    for name, g in base.groupby("FAC_NAME"):
        rows.append({
            "FAC_NAME": name,
            "COUNTY": (
                g["COUNTY"].astype("string").replace(r"^\s*$", pd.NA, regex=True)
                  .dropna().iloc[0]
                  if ("COUNTY" in g.columns and g["COUNTY"].notna().any())
                  else "Unknown"
            ),
            "OWNER": g["OWNER"].iloc[0] if "OWNER" in g.columns else None,
            "Op_Margin": kpis.operating_margin(g),
            "Total_Margin": kpis.total_margin(g),
            "Occ_Lic": kpis.occ_lic(g),
            "ALOS": kpis.alos_all(g),
            "NPR_per_Dis": kpis.npr_per_discharge(g),
            "Outpatient_Share": kpis.outpatient_share(g),
            "Expense_per_APD": kpis.expense_per_adjusted_pd(g),
            "LaborPct_of_OpRev": kpis.labor_pct_of_op_revenue(g),
            "RN_HPPD": kpis.rn_hppd(g),
            "Paid_HPPD": kpis.paid_hppd(g),
            "Contract_Share": kpis.contract_labor_share(g),
            "ED_per_1k_Dis": kpis.ed_visits_per_1k_dis(g),
            "CSection_Rate": kpis.c_section_rate(g),
        })
    m = pd.DataFrame(rows)

    kpi_choices = {
        "Operating Margin (%)": ("Op_Margin", True),
        "Total Margin (%)": ("Total_Margin", True),
        "Occupancy (ratio)": ("Occ_Lic", True),
        "ALOS (days)": ("ALOS", False),
        "NPR per Discharge ($/dis)": ("NPR_per_Dis", True),
        "Outpatient Share (%)": ("Outpatient_Share", True),
        "Expense per Adjusted Patient Day ($)": ("Expense_per_APD", False),
        "Labor Cost % of Op Revenue": ("LaborPct_of_OpRev", False),
        "RN HPPD": ("RN_HPPD", None),
        "Paid HPPD": ("Paid_HPPD", None),
        "Contract Labor Share (%)": ("Contract_Share", False),
        "ED Visits per 1,000 Discharges": ("ED_per_1k_Dis", None),
        "C-Section Rate (%)": ("CSection_Rate", None),
    }
    klabel = st.selectbox("Rank by KPI", list(kpi_choices.keys()))
    kcol, higher_is_better = kpi_choices[klabel]
    sort_desc = st.checkbox("Sort high → low", value=(higher_is_better in (True, None)))

    money_cols = {"NPR_per_Dis","Expense_per_APD"}
    def _fmt(v, c):
        if pd.isna(v): return ""
        if c in {"Op_Margin","Total_Margin","Outpatient_Share","LaborPct_of_OpRev","Contract_Share","CSection_Rate"}:
            return f"{v*100:.1f}%"
        if c in money_cols: return f"${v:,.0f}"
        if c in {"Occ_Lic"}: return f"{v:.2f}"
        if c in {"ALOS","RN_HPPD","Paid_HPPD"}: return f"{v:.2f}"
        if c == "ED_per_1k_Dis": return f"{v:,.0f}"
        return f"{v:,.2f}"

    ms = m.assign(_sort=m[kcol]).sort_values("_sort", ascending=not sort_desc, na_position="last").drop(columns="_sort")
    md = ms.copy()
    for c in md.columns:
        if c in {"FAC_NAME","COUNTY","OWNER"}: continue
        md[c] = md[c].apply(lambda x: _fmt(x, c))

    st.dataframe(md, use_container_width=True, height=520)
    st.download_button(
        "⬇️ Download leaderboard (CSV)",
        data=ms.to_csv(index=False),
        file_name="leaderboard.csv",
        mime="text/csv"
    )
    definitions_expander(
        "Leaderboard & Compare",
        formulas=SECTION_DOCS["Leaderboard & Compare"]["formulas"],
        fields=SECTION_DOCS["Leaderboard & Compare"]["fields"],
    )

# ---- Top-level navigation (pills if available; else sidebar radio) ----
def _rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

SECTIONS = [
    "Executive Overview",
    "Payer & Revenue Mix",
    "Cost & Productivity",
    "Access & Throughput",
    "Service Lines",
    "Liquidity & Capital",
    "Ownership Models",          # <-- add this
    "Leaderboard & Compare",
]


# if hasattr(st, "segmented_control"):
#     active = st.segmented_control(
#         "Navigate",
#         SECTIONS,
#         selection=st.session_state.get("active_section", SECTIONS[0]),
#         help="Use these pills to jump between sections."
#     )
# else:
#     active = st.sidebar.radio(
#         "Navigate",
#         SECTIONS,
#         index=SECTIONS.index(st.session_state.get("active_section", SECTIONS[0]))
#     )
# st.session_state["active_section"] = active

# --- Robust nav control: works across Streamlit versions ---
def nav_control(options):
    """
    Returns the selected item from `options`.
    Uses segmented_control if available; otherwise falls back to radio.
    Tries multiple known signatures to avoid TypeError on older builds.
    """
    default = st.session_state.get("active_section", options[0])
    seg = getattr(st, "segmented_control", None)

    if callable(seg):
        # Try progressively simpler signatures (no 'help' arg — older builds error on it)
        trials = (
            dict(selection=default),  # newer API
            dict(default=default),    # mid API
            dict(index=options.index(default) if default in options else 0),  # older API
            dict(),                   # last resort
        )
        for kwargs in trials:
            try:
                choice = seg("Navigate", options, key="nav_seg", **kwargs)
                st.session_state["active_section"] = choice
                return choice
            except TypeError:
                continue
            except Exception:
                break  # if seg itself is flaky, drop to radio

    # Fallback: always works
    choice = st.radio(
        "Navigate",
        options,
        index=options.index(default) if default in options else 0,
        key="nav_radio",
    )
    st.session_state["active_section"] = choice
    return choice

# ---- Define your sections (use your existing list if you already have one)
SECTIONS = [
    "Executive Overview",
    "Payer & Revenue Mix",
    "Cost & Productivity",
    "Access & Throughput",
    "Service Lines",
    "Liquidity & Capital",
    "Ownership Models",
    "Leaderboard & Compare",
]

# ---- Simple, robust navigation (works on all Streamlit versions) ----
# (Keep your existing SECTIONS list; don't redefine it if it's already above)
default = st.session_state.get("active_section", SECTIONS[0])
active = st.radio(
    "Navigate",
    SECTIONS,
    index=SECTIONS.index(default) if default in SECTIONS else 0,
    key="nav_radio",
)
st.session_state["active_section"] = active



cprev, cnext = st.columns([1, 1])
if cprev.button("◀ Prev"):
    idx = (SECTIONS.index(active) - 1) % len(SECTIONS)
    st.session_state["active_section"] = SECTIONS[idx]
    _rerun()
if cnext.button("Next ▶"):
    idx = (SECTIONS.index(active) + 1) % len(SECTIONS)
    st.session_state["active_section"] = SECTIONS[idx]
    _rerun()

show_all = st.checkbox("Show all sections on one page", value=False)

# if show_all:
#     for title in SECTIONS:
#         with st.expander(title, expanded=True):
#             if title == "Executive Overview": section_executive_overview()
#             elif title == "Payer & Revenue Mix": section_payer_revenue_mix()
#             elif title == "Cost & Productivity": section_cost_productivity()
#             elif title == "Access & Throughput": section_access_throughput()
#             elif title == "Service Lines": section_service_lines()
#             elif title == "Liquidity & Capital": section_liquidity_capital()
#             elif title == "Ownership Models": section_ownership_models()
#             elif title == "Leaderboard & Compare": section_leaderboard_compare()
# else:
#     if active == "Executive Overview": section_executive_overview()
#     elif active == "Payer & Revenue Mix": section_payer_revenue_mix()
#     elif active == "Cost & Productivity": section_cost_productivity()
#     elif active == "Access & Throughput": section_access_throughput()
#     elif active == "Service Lines": section_service_lines()
#     elif active == "Liquidity & Capital": section_liquidity_capital()
#     elif active == "Ownership Models": section_ownership_models()
#     elif active == "Leaderboard & Compare": section_leaderboard_compare()
if show_all:
    # (keep your existing expanders, no top title spam in show-all mode)
    for title in SECTIONS:
        with st.expander(title, expanded=True):
            if title == "Executive Overview": section_executive_overview()
            elif title == "Payer & Revenue Mix": section_payer_revenue_mix()
            elif title == "Cost & Productivity": section_cost_productivity()
            elif title == "Access & Throughput": section_access_throughput()
            elif title == "Service Lines": section_service_lines()
            elif title == "Liquidity & Capital": section_liquidity_capital()
            elif title == "Ownership Models": section_ownership_models()
            elif title == "Leaderboard & Compare": section_leaderboard_compare()
else:
    # 🔹 Render the new header just once per page/tab
    render_page_header(active)

    if active == "Executive Overview": section_executive_overview()
    elif active == "Payer & Revenue Mix": section_payer_revenue_mix()
    elif active == "Cost & Productivity": section_cost_productivity()
    elif active == "Access & Throughput": section_access_throughput()
    elif active == "Service Lines": section_service_lines()
    elif active == "Liquidity & Capital": section_liquidity_capital()
    elif active == "Ownership Models": section_ownership_models()
    elif active == "Leaderboard & Compare": section_leaderboard_compare()
