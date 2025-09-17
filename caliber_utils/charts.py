# caliber_utils/charts.py
from typing import Optional, Dict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- bar: selected vs peer (with IQR band) ---
def bar_selected_vs_peer(
    title: str,
    selected: Optional[float],
    peer_median: Optional[float],
    units: str = "",
    p25: Optional[float] = None,
    p75: Optional[float] = None,
    percent: bool = False
):
    s = selected * 100 if (percent and selected is not None) else selected
    m = peer_median * 100 if (percent and peer_median is not None) else peer_median
    ylab = "%" if percent else units

    fig = go.Figure()
    fig.add_bar(name="Selected Hospital", x=[title], y=[s])
    fig.add_bar(name="Peer Median", x=[title], y=[m])
    fig.update_layout(barmode="group", legend_title=None, yaxis_title=ylab, title=title)

    if p25 is not None and p75 is not None and pd.notna(p25) and pd.notna(p75):
        p25p = p25 * 100 if percent else p25
        p75p = p75 * 100 if percent else p75
        fig.add_hrect(y0=min(p25p, p75p), y1=max(p25p, p75p),
                      line_width=0, fillcolor="lightgray", opacity=0.25)
        fig.add_annotation(x=0, y=max(p25p, p75p), text="Peer IQR", showarrow=False, yshift=10)
    return fig

# --- stacked mix charts ---
def stacked_mix_chart(title: str, selected_mix: Dict[str, float], peer_mix: Dict[str, float]):
    labels = list(selected_mix.keys())
    sel_vals = [v * 100 if v == v else None for v in selected_mix.values()]
    peer_vals = [peer_mix.get(k, np.nan) * 100 for k in labels]
    df = pd.DataFrame({"Category": labels, "Selected %": sel_vals, "Peer Median %": peer_vals})
    fig = px.bar(
        df.melt(id_vars="Category", var_name="Series", value_name="Percent"),
        x="Category", y="Percent", color="Series", barmode="group", title=title
    )
    fig.update_layout(yaxis_title="%")
    return fig

def stacked_visit_mix(title: str, selected_mix: Dict[str, float], peer_mix: Dict[str, float]):
    labels = list(selected_mix.keys())
    sel_vals = [v * 100 if v == v else None for v in selected_mix.values()]
    peer_vals = [peer_mix.get(k, np.nan) * 100 for k in labels]
    df = pd.DataFrame({"Setting": labels, "Selected %": sel_vals, "Peer Median %": peer_vals})
    fig = px.bar(
        df.melt(id_vars="Setting", var_name="Series", value_name="Percent"),
        x="Setting", y="Percent", color="Series", barmode="group", title=title
    )
    fig.update_layout(yaxis_title="%")
    return fig

# --- scatters/bubbles ---
def scatter_alos_vs_occ(pfm: pd.DataFrame, highlight_name: Optional[str] = None):
    d = pfm.copy()
    d["OCC_LIC"] = pd.to_numeric(d["OCC_LIC"], errors="coerce")
    d["ALOS_ALL"] = pd.to_numeric(d["ALOS_ALL"], errors="coerce")
    d = d[(d["OCC_LIC"] > 0) & (d["ALOS_ALL"] > 0)]
    fig = px.scatter(
        d, x="OCC_LIC", y="ALOS_ALL",
        hover_data=[c for c in ["FAC_NAME", "COUNTY", "OWNER"] if c in d.columns],
        title="ALOS vs Occupancy (peers)"
    )
    if highlight_name and "FAC_NAME" in d.columns:
        h = d[d["FAC_NAME"] == highlight_name]
        if not h.empty:
            fig.add_scatter(
                x=h["OCC_LIC"], y=h["ALOS_ALL"], mode="markers+text",
                text=h["FAC_NAME"], textposition="top center",
                marker=dict(size=14, symbol="star"), name="Selected"
            )
    fig.update_layout(xaxis_title="Occupancy (ratio)", yaxis_title="Average Length of Stay (days)")
    return fig

def bubble_or_util(peers: pd.DataFrame, highlight_name: Optional[str] = None):
    df = peers.copy()
    for c in ["SURG_OP", "SURG_IP", "OP_MIN_IP", "OP_MIN_OP"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["OP_MIN_TOTAL"] = df["OP_MIN_IP"] + df["OP_MIN_OP"]
    fig = px.scatter(
        df, x="SURG_OP", y="SURG_IP", size="OP_MIN_TOTAL",
        hover_data=[c for c in ["FAC_NAME","COUNTY","OWNER"] if c in df.columns],
        title="Surgical Volume & OR Minutes (peers)"
    )
    if highlight_name and "FAC_NAME" in df.columns:
        h = df[df["FAC_NAME"] == highlight_name]
        if not h.empty:
            fig.add_scatter(
                x=h["SURG_OP"], y=h["SURG_IP"], mode="markers+text",
                text=h["FAC_NAME"], textposition="top center",
                marker=dict(size=16, symbol="star"), name="Selected"
            )
    fig.update_layout(xaxis_title="Outpatient Surgeries", yaxis_title="Inpatient Surgeries")
    return fig
