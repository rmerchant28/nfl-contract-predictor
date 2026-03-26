"""
Model Diagnostics — dark mode
Quantile calibration, bias by APY tier, and residual analysis.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Model Diagnostics — NFL Contract Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BG      = "#0d1117"
SURFACE = "#161b22"
SURFACE2= "#21262d"
BORDER  = "#30363d"
TEXT    = "#c9d1d9"
SUBTEXT = "#8b949e"
ACCENT  = "#f0b429"

POS_COLORS = {"QB": "#ff6b6b", "WR": "#4fc3f7", "RB": "#69f0ae", "TE": "#ce93d8"}

st.markdown(f"""
<style>
  .stApp {{ background-color: {BG}; }}
  [data-testid="stAppViewContainer"] > .main {{ background-color: {BG}; }}
  section[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}
  [data-testid="stMetric"] {{
      background: {SURFACE}; border: 1px solid {BORDER};
      border-radius: 12px; padding: 0.8rem 1.1rem !important;
  }}
  [data-testid="stMetricValue"] {{ color: {TEXT} !important; }}
  [data-testid="stMetricLabel"] {{ color: {SUBTEXT} !important; font-size:0.72rem !important; text-transform:uppercase; letter-spacing:0.05em; }}
  .diag-card {{
      background: {SURFACE}; border: 1px solid {BORDER};
      border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 0.5rem;
  }}
  .diag-label {{ font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:{SUBTEXT}; margin-bottom:4px; }}
  .diag-value {{ font-size:1.6rem; font-weight:800; color:{TEXT}; }}
  .diag-sub {{ font-size:0.82rem; color:{SUBTEXT}; margin-top:2px; }}
  .section-hdr {{ font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:{SUBTEXT}; margin:1.5rem 0 0.6rem; }}
  .cal-badge-good  {{ background:#1c3a1c; border:1px solid #3fb950; border-radius:20px; padding:3px 12px; color:#3fb950; font-size:0.8rem; font-weight:600; display:inline-block; }}
  .cal-badge-warn  {{ background:#3a2a00; border:1px solid {ACCENT}; border-radius:20px; padding:3px 12px; color:{ACCENT}; font-size:0.8rem; font-weight:600; display:inline-block; }}
  .cal-badge-bad   {{ background:#3a1010; border:1px solid #ff6b6b; border-radius:20px; padding:3px 12px; color:#ff6b6b; font-size:0.8rem; font-weight:600; display:inline-block; }}
  #MainMenu {{ visibility:hidden; }} footer {{ visibility:hidden; }} .stDeployButton {{ display:none; }}
</style>
""", unsafe_allow_html=True)

_DARK = dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(color=TEXT, size=11),
)


def hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


@st.cache_data(show_spinner=False)
def load_eval():
    p = ROOT / "models" / "evaluation.json"
    if not p.exists(): return {}
    with open(p) as f: return json.load(f)


@st.cache_data(show_spinner=False)
def load_oof():
    p = ROOT / "models" / "oof_predictions.csv"
    if not p.exists(): return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_summary():
    p = ROOT / "models" / "results_summary.json"
    if not p.exists(): return {}
    with open(p) as f: return json.load(f)


# ── Chart builders ────────────────────────────────────────────────────────────
def calibration_gauge(nominal, actual, position, color):
    """Bullet gauge: actual coverage vs 80% nominal target."""
    delta = actual - nominal

    fig = go.Figure()
    # Background track
    fig.add_shape(type="rect", x0=0, x1=1, y0=0.3, y1=0.7, fillcolor=SURFACE2, line_width=0)
    # Target line
    fig.add_shape(type="line", x0=nominal, x1=nominal, y0=0.1, y1=0.9,
                  line=dict(color=SUBTEXT, width=2, dash="dot"))
    # Actual fill
    fill_color = "#3fb950" if abs(delta) <= 0.05 else (ACCENT if abs(delta) <= 0.10 else "#ff6b6b")
    fig.add_shape(type="rect", x0=0, x1=actual, y0=0.32, y1=0.68,
                  fillcolor=hex_to_rgba(fill_color, 0.53), line=dict(color=fill_color, width=1))
    # Labels
    fig.add_annotation(x=actual, y=1.1, text=f"<b>{actual*100:.1f}%</b>",
                       showarrow=False, font=dict(size=13, color=fill_color))
    fig.add_annotation(x=nominal, y=-0.2, text=f"Target {nominal*100:.0f}%",
                       showarrow=False, font=dict(size=10, color=SUBTEXT))

    fig.update_layout(
        **_DARK,
        height=90, margin=dict(l=20, r=20, t=28, b=28),
        xaxis=dict(range=[0.4, 1.0], tickformat=".0%",
                   tickfont=dict(size=9, color=SUBTEXT), showgrid=False,
                   gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(visible=False, range=[-0.5, 1.5]),
        title=dict(text=f"{position} — Actual interval coverage",
                   font=dict(size=10, color=SUBTEXT), x=0),
    )
    return fig


def tier_bias_chart(tiers, mae_vals, bias_vals, counts, color):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MAE (cap %)", x=tiers, y=mae_vals,
        marker=dict(color=color, opacity=0.75,
                    line=dict(color=hex_to_rgba(color, 0.67), width=1)),
        text=[f"{v:.2f}%" for v in mae_vals], textposition="outside",
        textfont=dict(color=TEXT, size=10),
        hovertemplate="%{x}<br>MAE: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        name="Bias (over/under)", x=tiers, y=bias_vals,
        mode="markers+lines",
        marker=dict(color=ACCENT, size=9, symbol="diamond"),
        line=dict(color=ACCENT, width=1.5, dash="dot"),
        hovertemplate="%{x}<br>Bias: %{y:+.2f}%<extra></extra>",
        yaxis="y2",
    ))
    fig.add_shape(type="line", x0=-0.5, x1=len(tiers)-0.5, y0=0, y1=0,
                  line=dict(color=SUBTEXT, width=1, dash="dot"),
                  xref="x", yref="y2")

    fig.update_layout(
        **_DARK,
        height=300,
        yaxis=dict(title="MAE (cap % pts)", gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis2=dict(title=dict(text="Bias (cap % pts)", font=dict(color=ACCENT)),
                    overlaying="y", side="right",
                    showgrid=False, zerolinecolor=BORDER,
                    tickcolor=ACCENT),
        legend=dict(orientation="h", y=1.12, font=dict(color=TEXT)),
        xaxis=dict(gridcolor=BORDER),
        title=dict(text="MAE and systematic bias by contract tier",
                   font=dict(size=11, color=SUBTEXT), x=0),
        bargap=0.3,
    )
    return fig


def residual_histogram(errors, color):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors, nbinsx=30,
        marker=dict(color=color, opacity=0.75,
                    line=dict(color=hex_to_rgba(color, 0.67), width=0.5)),
        hovertemplate="Error %{x:.2f}%: %{y} contracts<extra></extra>",
        name="",
    ))
    # Zero line
    fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1,
                  xref="x", yref="paper",
                  line=dict(color=ACCENT, width=2, dash="dash"))
    # Mean line
    mean_err = float(np.mean(errors))
    if abs(mean_err) > 0.05:
        fig.add_shape(type="line", x0=mean_err, x1=mean_err, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(color="#ff6b6b", width=1.5, dash="dot"))
        fig.add_annotation(x=mean_err, y=0.95, xref="x", yref="paper",
                           text=f"bias {mean_err:+.2f}%",
                           showarrow=False, font=dict(size=9, color="#ff6b6b"),
                           xanchor="left" if mean_err >= 0 else "right")

    fig.update_layout(
        **_DARK,
        height=260,
        xaxis=dict(title="Prediction error (predicted − actual, cap % pts)",
                   gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(title="Count", gridcolor=BORDER, zerolinecolor=BORDER),
        showlegend=False,
        title=dict(text="Residual distribution", font=dict(size=11, color=SUBTEXT), x=0),
    )
    return fig


def residual_scatter(y_true, y_pred, errors, player_names, years, color):
    abs_err = [abs(e) for e in errors]

    fig = go.Figure()
    fig.add_shape(type="line", x0=min(y_pred)-0.5, x1=max(y_pred)+0.5, y0=0, y1=0,
                  line=dict(color=SUBTEXT, width=1, dash="dash"))
    fig.add_trace(go.Scatter(
        x=y_pred, y=errors,
        mode="markers",
        marker=dict(
            color=abs_err,
            colorscale=[[0, hex_to_rgba(color, 0.27)], [0.5, color], [1.0, "#ff6b6b"]],
            size=7, opacity=0.8,
            showscale=True,
            colorbar=dict(title=dict(text="Abs error", font=dict(color=SUBTEXT, size=9)),
                          tickfont=dict(color=SUBTEXT, size=9), thickness=10),
            line=dict(color=BG, width=0.5),
        ),
        text=[f"{n} ({y})<br>Pred: {p:.1f}%  Actual: {t:.1f}%"
              for n, y, p, t in zip(player_names, years, y_pred, y_true)],
        hovertemplate="%{text}<br>Error: %{y:+.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **_DARK,
        height=280,
        xaxis=dict(title="Predicted cap %", ticksuffix="%",
                   gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(title="Error (predicted − actual, cap %)",
                   gridcolor=BORDER, zerolinecolor=BORDER),
        title=dict(text="Errors vs. predicted value (heteroskedasticity check)",
                   font=dict(size=11, color=SUBTEXT), x=0),
        showlegend=False,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    eval_data = load_eval()
    oof_df    = load_oof()
    summary   = load_summary()

    st.markdown("## 🔬 Model Diagnostics")
    st.markdown(f"<span style='color:{SUBTEXT};font-size:0.93rem'>"
                "Quantile calibration, tier bias, and residual analysis for each position model.</span>",
                unsafe_allow_html=True)

    if not eval_data and oof_df.empty:
        st.warning("No evaluation data found. Run `make model` to generate diagnostics.")
        st.stop()

    positions = [p for p in ["QB", "WR", "RB", "TE"]
                 if p in eval_data or (not oof_df.empty and p in oof_df.get("position", pd.Series()).values)]

    if not positions:
        st.warning("No evaluation data found for any position.")
        st.stop()

    pos_tabs = st.tabs([f"{p}" for p in positions])

    for tab, pos in zip(pos_tabs, positions):
        with tab:
            color = POS_COLORS.get(pos, ACCENT)
            pos_eval = eval_data.get(pos, {})
            pos_oof  = oof_df[oof_df["position"] == pos].copy() if not oof_df.empty else pd.DataFrame()

            best_model = summary.get("best_models", {}).get(pos, "—")
            mae        = summary.get(pos, {}).get(best_model, {}).get("mae", None)
            rmse       = summary.get(pos, {}).get(best_model, {}).get("rmse", None)
            n_contracts = len(pos_oof) if not pos_oof.empty else "—"

            # ── Summary metrics ────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Best Model", best_model)
            with m2:
                st.metric("LOYO MAE", f"{mae:.2f}%" if mae else "—",
                          help="Mean absolute error across all left-out years")
            with m3:
                st.metric("LOYO RMSE", f"{rmse:.2f}%" if rmse else "—")
            with m4:
                st.metric("Contracts evaluated", str(n_contracts))

            st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

            # ── Three columns: calibration | tier bias | residuals summary ─────
            left, mid, right = st.columns(3)

            # Calibration
            with left:
                st.markdown('<p class="section-hdr">Quantile Calibration (p10–p90)</p>',
                            unsafe_allow_html=True)
                cal = pos_eval.get("calibration", {})
                if cal:
                    actual  = cal.get("actual_coverage", 0)
                    nominal = cal.get("nominal_coverage", 0.80)
                    delta   = actual - nominal
                    badge_cls = "cal-badge-good" if abs(delta) <= 0.05 else (
                                "cal-badge-warn" if abs(delta) <= 0.10 else "cal-badge-bad")
                    label = "Well calibrated" if abs(delta) <= 0.05 else (
                            "Slightly off" if abs(delta) <= 0.10 else
                            "Overconfident" if delta < 0 else "Conservative")
                    st.markdown(f'<span class="{badge_cls}">{label}</span>',
                                unsafe_allow_html=True)
                    st.markdown("")
                    cal_fig = calibration_gauge(nominal, actual, pos, color)
                    st.plotly_chart(cal_fig, use_container_width=True,
                                    config={"displayModeBar": False})
                    st.markdown(
                        f"<div style='color:{SUBTEXT};font-size:0.8rem'>"
                        f"{actual*100:.1f}% of contracts fall within the predicted p10–p90 range "
                        f"(target 80%). Evaluated on {cal.get('n','?')} LOYO contracts.</div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color:{SUBTEXT};font-size:0.85rem'>"
                                "Retrain models (requires XGBoost) to generate calibration data.</div>",
                                unsafe_allow_html=True)

            # Residual summary stats
            with mid:
                st.markdown('<p class="section-hdr">Residual Summary</p>', unsafe_allow_html=True)
                res = pos_eval.get("residuals", {})
                if not res and not pos_oof.empty:
                    errs = pos_oof["error_pct"]
                    res = {
                        "mean_bias": round(float(errs.mean()), 3),
                        "std":       round(float(errs.std()), 3),
                        "skewness":  round(float(errs.skew()), 3),
                        "pct_overpredict": round(float((errs > 0).mean() * 100), 1),
                        "p10": round(float(errs.quantile(0.10)), 3),
                        "p90": round(float(errs.quantile(0.90)), 3),
                    }
                if res:
                    bias = res.get("mean_bias", 0)
                    bias_color = "#ff6b6b" if abs(bias) > 0.5 else "#3fb950"
                    pct_over   = res.get("pct_overpredict", 50)
                    items = [
                        ("Mean Bias",    f"{bias:+.2f}%",       bias_color),
                        ("Std Dev",      f"{res.get('std',0):.2f}%", TEXT),
                        ("Skewness",     f"{res.get('skewness',0):.2f}", TEXT),
                        ("Overpredicts", f"{pct_over:.0f}% of contracts", TEXT),
                        ("p10 error",    f"{res.get('p10',0):+.2f}%", TEXT),
                        ("p90 error",    f"{res.get('p90',0):+.2f}%", TEXT),
                    ]
                    for label, val, clr in items:
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;"
                            f"padding:5px 0;border-bottom:1px solid {BORDER}'>"
                            f"<span style='color:{SUBTEXT};font-size:0.82rem'>{label}</span>"
                            f"<span style='color:{clr};font-weight:600;font-size:0.85rem'>{val}</span>"
                            f"</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color:{SUBTEXT}'>No residual data available.</div>",
                                unsafe_allow_html=True)

            # Contract years MAE
            with right:
                st.markdown('<p class="section-hdr">Contract Length Model</p>',
                            unsafe_allow_html=True)
                yrs_mae = pos_eval.get("contract_years_mae")
                if yrs_mae:
                    badge_cls = ("cal-badge-good" if yrs_mae < 0.7
                                 else "cal-badge-warn" if yrs_mae < 1.2 else "cal-badge-bad")
                    st.markdown(f'<span class="{badge_cls}">MAE {yrs_mae:.2f} yrs</span>',
                                unsafe_allow_html=True)
                    st.markdown("")
                    st.markdown(
                        f"<div style='color:{SUBTEXT};font-size:0.82rem'>"
                        f"The contract length model predicts the number of years in a new deal. "
                        f"An MAE of {yrs_mae:.2f} years means predictions are typically within "
                        f"{'less than 1 year' if yrs_mae < 1.0 else 'about 1–1.5 years'} "
                        f"of the actual contract length.</div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color:{SUBTEXT};font-size:0.85rem'>"
                                "Contract length model not yet trained. Run `make model`.</div>",
                                unsafe_allow_html=True)

            if pos_oof.empty:
                st.info("Run `make model` to generate out-of-fold predictions for detailed charts.")
                continue

            st.markdown("---")

            # ── Tier bias chart ────────────────────────────────────────────────
            st.markdown('<p class="section-hdr">Error by Contract Tier</p>',
                        unsafe_allow_html=True)

            tier_data = pos_eval.get("tier_bias", {})
            if tier_data:
                tiers  = tier_data.get("tiers", [])
                maes   = tier_data.get("mae", [])
                biases = tier_data.get("bias", [])
                counts = tier_data.get("counts", [])
            else:
                # Compute from OOF on the fly
                try:
                    pos_oof_t = pos_oof.copy()
                    pos_oof_t["tier"] = pd.qcut(
                        pos_oof_t["y_true_pct"],
                        q=[0, 0.25, 0.5, 0.75, 0.9, 1.0],
                        labels=["Bottom 25%", "Q25–50%", "Q50–75%", "Q75–90%", "Top 10%"],
                    )
                    ts = pos_oof_t.groupby("tier", observed=True).agg(
                        mae=("abs_error_pct", "mean"),
                        bias=("error_pct", "mean"),
                        n=("error_pct", "count"),
                    )
                    tiers  = ts.index.tolist()
                    maes   = ts["mae"].tolist()
                    biases = ts["bias"].tolist()
                    counts = ts["n"].tolist()
                except Exception:
                    tiers = maes = biases = counts = []

            if tiers:
                tier_fig = tier_bias_chart(tiers, maes, biases, counts, color)
                st.plotly_chart(tier_fig, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown(
                    f"<div style='color:{SUBTEXT};font-size:0.8rem'>"
                    f"<strong style='color:{TEXT}'>How to read:</strong> Bars show MAE per tier "
                    f"(smaller = more accurate). The diamond line shows systematic bias — "
                    f"positive = model overpredicts that tier, negative = underpredicts. "
                    f"Ideal: low, flat bars and bias near zero across all tiers.</div>",
                    unsafe_allow_html=True)

            st.markdown('<p class="section-hdr">Residual Analysis</p>', unsafe_allow_html=True)
            r_left, r_right = st.columns(2)

            with r_left:
                hist_fig = residual_histogram(pos_oof["error_pct"].tolist(), color)
                st.plotly_chart(hist_fig, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown(
                    f"<div style='color:{SUBTEXT};font-size:0.8rem'>"
                    f"Distribution of prediction errors. Centered at zero = unbiased. "
                    f"Skewed right = systematically overpredicts.</div>",
                    unsafe_allow_html=True)

            with r_right:
                scat_fig = residual_scatter(
                    pos_oof["y_true_pct"].tolist(),
                    pos_oof["y_pred_pct"].tolist(),
                    pos_oof["error_pct"].tolist(),
                    pos_oof["player_name"].tolist(),
                    pos_oof["signing_year"].tolist(),
                    color,
                )
                st.plotly_chart(scat_fig, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown(
                    f"<div style='color:{SUBTEXT};font-size:0.8rem'>"
                    f"Errors vs. predicted value. If errors fan out at higher values, "
                    f"the model is less reliable for elite contracts (heteroskedasticity).</div>",
                    unsafe_allow_html=True)

            # ── Worst misses ───────────────────────────────────────────────────
            st.markdown('<p class="section-hdr">Largest Prediction Errors</p>',
                        unsafe_allow_html=True)
            worst = pos_oof.nlargest(8, "abs_error_pct")[
                ["player_name", "signing_year", "y_true_pct", "y_pred_pct", "error_pct"]
            ].copy()
            worst.columns = ["Player", "Year", "Actual %", "Predicted %", "Error %"]
            worst["Actual %"]    = worst["Actual %"].round(1)
            worst["Predicted %"] = worst["Predicted %"].round(1)
            worst["Error %"]     = worst["Error %"].round(2)

            def highlight_error(row):
                v = row["Error %"]
                clr = f"background-color:#3a1010" if v > 3 else (
                      f"background-color:#3a2a00" if v > 1.5 else "")
                return [clr] * len(row)

            st.dataframe(worst.style.apply(highlight_error, axis=1),
                         use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        "Diagnostics generated during `make model` · "
        "LOYO = Leave-One-Year-Out cross-validation · "
        "Quantile calibration requires XGBoost"
    )


if __name__ == "__main__":
    main()
