"""
NFL Contract Predictor — Streamlit App  (dark mode)
Run: streamlit run app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from predict import predict_contract, find_comps, list_available_positions, CAP_HISTORY

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NFL Contract Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Dark palette ───────────────────────────────────────────────────────────────
BG        = "#0d1117"
SURFACE   = "#161b22"
SURFACE2  = "#21262d"
BORDER    = "#30363d"
TEXT      = "#c9d1d9"
SUBTEXT   = "#8b949e"
ACCENT    = "#f0b429"

POSITION_COLORS = {
    "QB": ("#ff6b6b", "#2d1515"),
    "WR": ("#4fc3f7", "#0d2233"),
    "RB": ("#69f0ae", "#0d2618"),
    "TE": ("#ce93d8", "#1e1030"),
}

TEAM_COLORS = {
    "Cardinals": ("#97233F", "#FFB612"), "Falcons":    ("#A71930", "#000000"),
    "Ravens":    ("#241773", "#9E7C0C"), "Bills":      ("#00338D", "#C60C30"),
    "Panthers":  ("#0085CA", "#101820"), "Bears":      ("#0B162A", "#C83803"),
    "Bengals":   ("#FB4F14", "#000000"), "Browns":     ("#311D00", "#FF3C00"),
    "Cowboys":   ("#041E42", "#869397"), "Broncos":    ("#FB4F14", "#002244"),
    "Lions":     ("#0076B6", "#B0B7BC"), "Packers":    ("#203731", "#FFB612"),
    "Texans":    ("#03202F", "#A71930"), "Colts":      ("#002C5F", "#A2AAAD"),
    "Jaguars":   ("#101820", "#D7A22A"), "Chiefs":     ("#E31837", "#FFB81C"),
    "Raiders":   ("#000000", "#A5ACAF"), "Chargers":   ("#0080C6", "#FFC20E"),
    "Rams":      ("#003594", "#FFA300"), "Dolphins":   ("#008E97", "#FC4C02"),
    "Vikings":   ("#4F2683", "#FFC62F"), "Patriots":   ("#002244", "#C60C30"),
    "Saints":    ("#D3BC8D", "#101820"), "Giants":     ("#0B2265", "#A71930"),
    "Jets":      ("#125740", "#000000"), "Eagles":     ("#004C54", "#A5ACAF"),
    "Steelers":  ("#101820", "#FFB612"), "49ers":      ("#AA0000", "#B3995D"),
    "Seahawks":  ("#002244", "#69BE28"), "Buccaneers": ("#D50A0A", "#FF7900"),
    "Titans":    ("#0C2340", "#4B92DB"), "Commanders": ("#5A1414", "#FFB612"),
}


def get_team_colors(team: str) -> tuple[str, str]:
    if not team:
        return "#1f6feb", ACCENT
    for key, colors in TEAM_COLORS.items():
        if key.lower() in str(team).lower():
            return colors
    return "#1f6feb", ACCENT


# ── CSS ────────────────────────────────────────────────────────────────────────
def inject_css(primary: str, secondary: str):
    st.markdown(f"""
    <style>
      /* ── Base ── */
      .stApp {{ background-color: {BG}; }}
      [data-testid="stAppViewContainer"] > .main {{ background-color: {BG}; }}
      section[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}

      /* ── Streamlit metric cards ── */
      [data-testid="stMetric"] {{
          background: {SURFACE};
          border: 1px solid {BORDER};
          border-radius: 12px;
          padding: 0.8rem 1.1rem !important;
      }}
      [data-testid="stMetricValue"] {{ color: {TEXT} !important; font-size: 1.3rem !important; }}
      [data-testid="stMetricLabel"] {{ color: {SUBTEXT} !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.05em; }}

      /* ── Hero banner ── */
      .hero {{
          background: linear-gradient(135deg, {primary}28 0%, {SURFACE} 60%);
          border: 1px solid {primary}45;
          border-radius: 16px;
          padding: 2rem 2.5rem;
          margin-bottom: 1.5rem;
          position: relative;
          overflow: hidden;
      }}
      .hero::after {{
          content: '';
          position: absolute;
          top: -60px; right: -60px;
          width: 260px; height: 260px;
          background: radial-gradient({primary}18, transparent 70%);
          border-radius: 50%;
          pointer-events: none;
      }}
      .hero-name {{
          font-size: 2.2rem; font-weight: 800;
          color: {TEXT}; margin: 0; letter-spacing: -0.5px;
      }}
      .hero-sub {{ font-size: 0.95rem; color: {SUBTEXT}; margin: 0.25rem 0 0; }}
      .hero-apy {{
          font-size: 2.8rem; font-weight: 800;
          color: {secondary}; margin: 1rem 0 0; line-height: 1;
      }}
      .hero-cap {{ font-size: 1rem; color: {SUBTEXT}; margin: 0.3rem 0 0; }}
      .hero-years {{
          display: inline-block;
          background: {primary}22; border: 1px solid {primary}44;
          border-radius: 20px; padding: 3px 12px;
          font-size: 0.82rem; color: {TEXT}; margin-top: 0.5rem;
      }}

      /* ── Accent bar ── */
      .accent-bar {{
          height: 3px;
          background: linear-gradient(90deg, {primary}, {secondary}88);
          border-radius: 2px; margin: 1.2rem 0;
      }}

      /* ── Section headers ── */
      .section-header {{
          font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
          letter-spacing: 0.1em; color: {SUBTEXT}; margin: 1.5rem 0 0.6rem;
      }}

      /* ── Comparable contract rows ── */
      .comp-row {{
          display: flex; justify-content: space-between; align-items: center;
          padding: 10px 14px; border-radius: 10px; margin: 5px 0;
          background: {SURFACE}; border: 1px solid {BORDER};
          transition: border-color 0.15s, background 0.15s;
      }}
      .comp-row:hover {{ background: {SURFACE2}; border-color: {primary}55; }}
      .comp-name {{ font-weight: 600; color: {TEXT}; font-size: 0.9rem; }}
      .comp-meta {{ color: {SUBTEXT}; font-size: 0.78rem; margin-top: 2px; }}
      .comp-pct {{ font-weight: 800; font-size: 1rem; color: {primary}; }}

      /* ── Stat pills ── */
      .stat-pill {{
          display: inline-block; background: {SURFACE2}; border: 1px solid {BORDER};
          border-radius: 20px; padding: 4px 12px; font-size: 0.78rem;
          margin: 3px 2px; color: {TEXT};
      }}
      .stat-pill strong {{ color: {primary}; }}

      /* ── Warning/info boxes ── */
      [data-testid="stAlert"] {{ background: {SURFACE2} !important; border-radius: 10px !important; }}

      /* ── Tables ── */
      [data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}

      /* ── Hide Streamlit chrome ── */
      #MainMenu {{ visibility: hidden; }}
      footer {{ visibility: hidden; }}
      .stDeployButton {{ display: none; }}

      /* ── Input labels ── */
      .stSelectbox label {{
          font-size: 0.72rem !important; font-weight: 700 !important;
          text-transform: uppercase !important; letter-spacing: 0.05em !important;
          color: {SUBTEXT} !important;
      }}
    </style>
    """, unsafe_allow_html=True)


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_contracts():
    path = ROOT / "data" / "raw" / "contracts_with_cap_pct.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["signing_year"] = pd.to_numeric(df["signing_year"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_oof():
    path = ROOT / "models" / "oof_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_results_summary():
    path = ROOT / "models" / "results_summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _get_active_player_names() -> set:
    active = set()
    for stat_file in ("pfr_passing", "pfr_receiving", "pfr_rushing"):
        path = ROOT / "data" / "raw" / f"{stat_file}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["player_name", "season"])
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
        active.update(df[df["season"] >= 2023]["player_name"].dropna().tolist())
    return active


@st.cache_data(show_spinner=False)
def get_player_list():
    contracts = load_contracts()
    if contracts.empty:
        return []
    active      = _get_active_player_names()
    all_players = contracts["player_name"].dropna().unique().tolist()
    filtered    = [p for p in all_players if p in active]
    return sorted(filtered) if filtered else sorted(all_players)


@st.cache_data(show_spinner=False)
def get_player_info(player_name: str):
    contracts = load_contracts()
    if contracts.empty:
        return {}
    rows = contracts[contracts["player_name"] == player_name].sort_values("signing_year", ascending=False)
    return rows.iloc[0].to_dict() if not rows.empty else {}


@st.cache_data(show_spinner=False)
def cached_predict(player_name: str, position: str, signing_year: int):
    return predict_contract(player_name, position, signing_year)


@st.cache_data(show_spinner=False)
def cached_comps(position: str, predicted_cap_pct: float, signing_year: int):
    return find_comps(position, predicted_cap_pct, signing_year, n=6)


# ── Chart helpers ──────────────────────────────────────────────────────────────
def hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


_DARK_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(color=TEXT, size=11),
)


def make_confidence_chart(predicted, low, high, position, primary) -> go.Figure:
    pos_range = {"QB": (8, 26), "WR": (2, 16), "RB": (1, 11), "TE": (1, 10)}
    x_min, x_max = pos_range.get(position, (1, 20))

    fig = go.Figure()
    fig.add_shape(type="rect", x0=x_min, x1=x_max, y0=0.3, y1=0.7,
                  fillcolor=SURFACE2, line_width=0)
    fig.add_shape(type="rect", x0=low, x1=high, y0=0.22, y1=0.78,
                  fillcolor=hex_to_rgba(primary, 0.18),
                  line=dict(color=hex_to_rgba(primary, 0.5), width=1))
    fig.add_shape(type="line", x0=predicted, x1=predicted, y0=0.05, y1=0.95,
                  line=dict(color=primary, width=3))
    fig.add_annotation(x=predicted, y=1.15, text=f"<b>{predicted:.1f}%</b>",
                       showarrow=False, font=dict(size=14, color=primary))
    fig.add_annotation(x=(low + high) / 2, y=-0.25,
                       text=f"{low:.1f}% – {high:.1f}%",
                       showarrow=False, font=dict(size=11, color=SUBTEXT))

    fig.update_layout(
        **_DARK_LAYOUT,
        height=100,
        margin=dict(l=20, r=20, t=32, b=32),
        xaxis=dict(range=[x_min, x_max], showgrid=False,
                   ticksuffix="%", tickfont=dict(size=10, color=SUBTEXT),
                   gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(visible=False, range=[-0.6, 1.6]),
        showlegend=False,
    )
    return fig


def make_oof_chart(position: str, primary: str) -> go.Figure | None:
    oof = load_oof()
    if oof.empty:
        return None
    pos_oof = oof[oof["position"] == position].copy()
    if pos_oof.empty:
        return None

    mae = pos_oof["abs_error_pct"].mean()
    axis_max = max(pos_oof["y_true_pct"].max(), pos_oof["y_pred_pct"].max()) + 1
    axis_min = max(0, min(pos_oof["y_true_pct"].min(), pos_oof["y_pred_pct"].min()) - 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[axis_min, axis_max], y=[axis_min, axis_max],
        mode="lines", line=dict(color=BORDER, width=1, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=pos_oof["y_true_pct"], y=pos_oof["y_pred_pct"],
        mode="markers",
        marker=dict(color=primary, size=7, opacity=0.7,
                    line=dict(color=BG, width=1)),
        text=pos_oof["player_name"] + " (" + pos_oof["signing_year"].astype(str) + ")",
        hovertemplate="<b>%{text}</b><br>Actual: %{x:.1f}%<br>Predicted: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        **_DARK_LAYOUT,
        height=260,
        xaxis=dict(title="Actual cap %", ticksuffix="%",
                   range=[axis_min, axis_max], gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(title="Predicted cap %", ticksuffix="%",
                   range=[axis_min, axis_max], gridcolor=BORDER, zerolinecolor=BORDER),
        title=dict(text=f"{position} accuracy (LOYO CV) — MAE {mae:.2f}%",
                   font=dict(size=11, color=SUBTEXT), x=0),
    )
    return fig


def make_importance_chart(result: dict, primary: str, top_n: int = 12) -> go.Figure | None:
    features = result.get("features_used", {})
    if not features:
        return None

    mean_feats = {k: v for k, v in features.items()
                  if k.endswith("_mean") and not np.isnan(v)}
    if not mean_feats:
        mean_feats = {k: v for k, v in features.items() if not np.isnan(float(v))}

    def fmt_name(n):
        return n.replace("_mean", "").replace("_last", "").replace("_trend", "").replace("_", " ").title()

    items  = sorted(mean_feats.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    labels = [fmt_name(k) for k, _ in items]
    values = [v for _, v in items]
    colors = [primary if v >= 0 else "#ff6b6b" for v in values]

    fig = go.Figure(go.Bar(
        x=values[::-1], y=labels[::-1],
        orientation="h",
        marker=dict(color=colors[::-1], opacity=0.85),
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_DARK_LAYOUT,
        height=max(200, top_n * 24),
        xaxis=dict(title="3-yr mean value", gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(showgrid=False),
        title=dict(text="Stats driving the prediction", font=dict(size=11, color=SUBTEXT), x=0),
    )
    return fig


# ── Formatters ─────────────────────────────────────────────────────────────────
def fmt_money(v: float) -> str:
    return f"${v / 1_000_000:.1f}M" if v >= 1_000_000 else f"${v:,.0f}"


def fmt_stat(key: str, value: float) -> str:
    if key.endswith("_games"):
        return str(int(value))
    if any(x in key for x in ("epa", "pacr", "racr", "wopr")):
        return f"{value:.2f}"
    if any(x in key for x in ("target_share", "air_yards_share", "catch_rate", "completion_pct")):
        return f"{value * 100:.1f}%" if abs(value) <= 1.5 else f"{value:.1f}%"
    if "pct" in key:
        return f"{value * 100:.1f}%" if abs(value) <= 1.5 else f"{value:.1f}%"
    if value > 100:
        return f"{value:,.0f}"
    return f"{value:.1f}"


def fmt_key(key: str) -> str:
    return (key.replace("_mean", "").replace("_last", " (last yr)")
               .replace("_trend", " (trend)").replace("_", " ").strip().title())


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    available_positions = list_available_positions()
    if not available_positions:
        st.error("No trained models found. Run `python notebooks/model.py` first.")
        st.stop()

    # ── Header + search ────────────────────────────────────────────────────────
    st.markdown("## 🏈 NFL Contract Predictor")
    st.markdown(
        "<span style='color:" + SUBTEXT + ";font-size:0.93rem'>"
        "Predict what a player would earn on a new free agent contract "
        "as a percentage of the salary cap — normalized across eras.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    col_search, col_pos, col_year = st.columns([3, 1, 1])
    with col_search:
        player_list = get_player_list()
        player_name = st.selectbox("Player", options=[""] + player_list,
                                   index=0, placeholder="Search player...")
    with col_pos:
        position = st.selectbox("Position", options=available_positions, index=0)
    with col_year:
        signing_year = st.selectbox("Contract year", options=list(range(2026, 2013, -1)), index=0)

    if not player_name:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div style='background:{SURFACE};border:1px solid {BORDER};"
                        f"border-radius:12px;padding:1.2rem'>"
                        f"<div style='font-size:1.4rem'>🔍</div>"
                        f"<div style='font-weight:700;color:{TEXT};margin:0.5rem 0 0.3rem'>Search a player</div>"
                        f"<div style='color:{SUBTEXT};font-size:0.85rem'>Select any QB, WR, RB, or TE "
                        f"who has signed at least one veteran contract.</div></div>",
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div style='background:{SURFACE};border:1px solid {BORDER};"
                        f"border-radius:12px;padding:1.2rem'>"
                        f"<div style='font-size:1.4rem'>📅</div>"
                        f"<div style='font-weight:700;color:{TEXT};margin:0.5rem 0 0.3rem'>Set contract year</div>"
                        f"<div style='color:{SUBTEXT};font-size:0.85rem'>Defaults to 2026. Uses stats from "
                        f"the 3 seasons before the contract year.</div></div>",
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div style='background:{SURFACE};border:1px solid {BORDER};"
                        f"border-radius:12px;padding:1.2rem'>"
                        f"<div style='font-size:1.4rem'>📊</div>"
                        f"<div style='font-weight:700;color:{TEXT};margin:0.5rem 0 0.3rem'>See the prediction</div>"
                        f"<div style='color:{SUBTEXT};font-size:0.85rem'>Predicted APY, cap %, guaranteed money, "
                        f"and contract length — with comparable historical deals.</div></div>",
                        unsafe_allow_html=True)
        st.stop()

    # ── Run prediction ─────────────────────────────────────────────────────────
    with st.spinner(f"Loading stats for {player_name}..."):
        try:
            result = cached_predict(player_name, position, signing_year)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # ── Colors ─────────────────────────────────────────────────────────────────
    player_info = get_player_info(player_name)
    team        = player_info.get("team", "")
    primary, secondary = get_team_colors(team)
    inject_css(primary, secondary)

    predicted_cap_pct   = result["predicted_cap_pct"]
    predicted_apy       = result["predicted_apy"]
    cap                 = result["cap_that_year"]
    conf_low, conf_high = result["confidence_range"]
    predicted_gtd       = result.get("predicted_guaranteed")
    predicted_gtd_pct   = result.get("predicted_gtd_pct")
    predicted_years     = result.get("predicted_years")

    # ── Low activity warning ───────────────────────────────────────────────────
    features     = result.get("features_used", {})
    attempts_mean = features.get("attempts_mean")
    games_mean    = features.get("games_mean")

    if position == "QB" and attempts_mean is not None and float(attempts_mean) < 150:
        st.warning(f"⚠️ Limited starter data — {player_name} averaged <150 pass attempts/season. "
                   "Prediction may overestimate market value.")
    elif position != "QB" and games_mean is not None and float(games_mean) < 10:
        st.warning(f"⚠️ Limited playing time — {player_name} averaged <10 games/season. "
                   "Prediction may be less reliable.")

    # ── Hero banner ────────────────────────────────────────────────────────────
    team_display  = f" · {team}" if team else ""
    years_display = (f'<span class="hero-years">~{predicted_years:.0f}-year deal</span>'
                     if predicted_years else "")
    st.markdown(f"""
    <div class="hero">
      <p class="hero-name">{player_name}</p>
      <p class="hero-sub">{position}{team_display} · {signing_year} projection</p>
      <p class="hero-apy">{fmt_money(predicted_apy)}<span style="font-size:1.2rem;font-weight:400;color:{SUBTEXT}"> / yr</span></p>
      <p class="hero-cap">{predicted_cap_pct:.1f}% of the salary cap · {fmt_money(cap)} cap</p>
      {years_display}
    </div>
    """, unsafe_allow_html=True)

    # ── Metric row ─────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Predicted APY", fmt_money(predicted_apy))
    with m2:
        st.metric("Cap %", f"{predicted_cap_pct:.2f}%")
    with m3:
        if predicted_gtd is not None:
            st.metric("Est. Guaranteed", fmt_money(predicted_gtd),
                      help=f"{predicted_gtd_pct:.1f}% of cap")
        else:
            st.metric("Est. Guaranteed", "—",
                      help="Retrain models to enable guaranteed money prediction")
    with m4:
        ci_method = result.get("ci_method", "mae")
        if ci_method == "cqr":
            ci_help = "Conformalized Quantile Regression — statistically calibrated 80% coverage interval."
        elif ci_method == "quantile":
            ci_help = "10th–90th percentile from quantile regression (uncalibrated)."
        else:
            ci_help = "Symmetric ±MAE fallback. Retrain models to enable calibrated intervals."
        st.metric("Confidence range", f"{conf_low:.1f}% – {conf_high:.1f}%", help=ci_help)
    with m5:
        results_summary = load_results_summary()
        model_name = result.get("model_used", "—")
        mae = results_summary.get(position, {}).get(model_name, {}).get("mae")
        st.metric("Model MAE", f"{mae:.2f}%" if mae else "—",
                  help="Mean absolute error from leave-one-year-out CV")

    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    # ── Two-column layout ──────────────────────────────────────────────────────
    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        st.markdown('<p class="section-header">Confidence range</p>', unsafe_allow_html=True)
        conf_fig = make_confidence_chart(predicted_cap_pct, conf_low, conf_high, position, primary)
        st.plotly_chart(conf_fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<p class="section-header">Comparable historical contracts</p>',
                    unsafe_allow_html=True)
        comps = cached_comps(position, predicted_cap_pct, signing_year)
        if not comps.empty:
            for _, row in comps.iterrows():
                st.markdown(f"""
                <div class="comp-row">
                  <div>
                    <div class="comp-name">{row['Player']}</div>
                    <div class="comp-meta">{row['Team']} · {int(row['Year'])} · {row['Years']} yrs</div>
                  </div>
                  <div style="text-align:right">
                    <div class="comp-pct">{row['Cap %']}%</div>
                    <div class="comp-meta">{row['APY']}/yr</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No comparable contracts found in historical data.")

        st.markdown(f'<p class="section-header">Model accuracy — {position}</p>',
                    unsafe_allow_html=True)
        oof_fig = make_oof_chart(position, primary)
        if oof_fig:
            st.plotly_chart(oof_fig, use_container_width=True, config={"displayModeBar": False})

    with right_col:
        st.markdown('<p class="section-header">Stats used in prediction</p>',
                    unsafe_allow_html=True)
        missing = result.get("missing_features", [])

        if features:
            mean_feats = {k: v for k, v in features.items()
                          if k.endswith("_mean") and not np.isnan(float(v))
                          and not k.startswith("games_")}
            other_feats = {k: v for k, v in features.items()
                           if not any(k.endswith(s) for s in ("_mean", "_last", "_trend", "_games"))
                           and not np.isnan(float(v))}

            if mean_feats:
                st.caption(f"3-year averages ({signing_year - 3}–{signing_year - 1})")
                pills = "".join(
                    f'<span class="stat-pill"><strong>{fmt_key(k)}:</strong> {fmt_stat(k, v)}</span>'
                    for k, v in list(mean_feats.items())[:12]
                )
                st.markdown(pills, unsafe_allow_html=True)

            if other_feats:
                st.caption("Market context & signals")
                pills = "".join(
                    f'<span class="stat-pill"><strong>{fmt_key(k)}:</strong> {fmt_stat(k, v)}</span>'
                    for k, v in other_feats.items()
                )
                st.markdown(pills, unsafe_allow_html=True)

            if missing:
                with st.expander(f"{len(missing)} features missing (imputed with median)"):
                    st.caption(", ".join(missing[:20]))
        else:
            st.warning(f"No stats found for {player_name} in seasons "
                       f"{signing_year - 3}–{signing_year - 1}. "
                       "All stat features imputed with median.")

        st.markdown('<p class="section-header">What drove the prediction</p>',
                    unsafe_allow_html=True)
        imp_fig = make_importance_chart(result, primary)
        if imp_fig:
            st.plotly_chart(imp_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No stats found — cannot display feature chart.")

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"Model: {result.get('model_used', '—')} · "
        f"Training data: OTC contract history + nflreadpy · "
        f"Validated via leave-one-year-out CV · "
        f"Not affiliated with the NFL or any team."
    )


if __name__ == "__main__":
    main()
