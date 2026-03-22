"""
NFL Contract Predictor — Streamlit App
=======================================
Run: streamlit run app.py

Requires trained models in models/ directory.
Run notebooks/model.py first if models don't exist.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

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

# ── NFL team colors ────────────────────────────────────────────────────────────
TEAM_COLORS = {
    "Cardinals":    ("#97233F", "#FFB612"), "Falcons":     ("#A71930", "#000000"),
    "Ravens":       ("#241773", "#9E7C0C"), "Bills":       ("#00338D", "#C60C30"),
    "Panthers":     ("#0085CA", "#101820"), "Bears":       ("#0B162A", "#C83803"),
    "Bengals":      ("#FB4F14", "#000000"), "Browns":      ("#311D00", "#FF3C00"),
    "Cowboys":      ("#041E42", "#869397"), "Broncos":     ("#FB4F14", "#002244"),
    "Lions":        ("#0076B6", "#B0B7BC"), "Packers":     ("#203731", "#FFB612"),
    "Texans":       ("#03202F", "#A71930"), "Colts":       ("#002C5F", "#A2AAAD"),
    "Jaguars":      ("#101820", "#D7A22A"), "Chiefs":      ("#E31837", "#FFB81C"),
    "Raiders":      ("#000000", "#A5ACAF"), "Chargers":    ("#0080C6", "#FFC20E"),
    "Rams":         ("#003594", "#FFA300"), "Dolphins":    ("#008E97", "#FC4C02"),
    "Vikings":      ("#4F2683", "#FFC62F"), "Patriots":    ("#002244", "#C60C30"),
    "Saints":       ("#D3BC8D", "#101820"), "Giants":      ("#0B2265", "#A71930"),
    "Jets":         ("#125740", "#000000"), "Eagles":      ("#004C54", "#A5ACAF"),
    "Steelers":     ("#101820", "#FFB612"), "49ers":       ("#AA0000", "#B3995D"),
    "Seahawks":     ("#002244", "#69BE28"), "Buccaneers":  ("#D50A0A", "#FF7900"),
    "Titans":       ("#0C2340", "#4B92DB"), "Commanders":  ("#5A1414", "#FFB612"),
}

DEFAULT_PRIMARY   = "#1a1a2e"
DEFAULT_SECONDARY = "#e8b500"

POSITION_COLORS = {
    "QB": ("#c0392b", "#fadbd8"),
    "WR": ("#1a5276", "#d6eaf8"),
    "RB": ("#1e8449", "#d5f5e3"),
    "TE": ("#7d3c98", "#e8daef"),
}


def get_team_colors(team_name: str) -> tuple[str, str]:
    """Return (primary, secondary) hex colors for a team."""
    if not team_name:
        return DEFAULT_PRIMARY, DEFAULT_SECONDARY
    for key, colors in TEAM_COLORS.items():
        if key.lower() in str(team_name).lower():
            return colors
    return DEFAULT_PRIMARY, DEFAULT_SECONDARY


# ── CSS injection ──────────────────────────────────────────────────────────────
def inject_css(primary: str, secondary: str):
    st.markdown(f"""
    <style>
        /* Hero banner */
        .hero {{
            background: linear-gradient(135deg, {primary} 0%, {primary}dd 100%);
            border-radius: 12px;
            padding: 2rem 2.5rem;
            margin-bottom: 1.5rem;
            color: white;
        }}
        .hero-name {{
            font-size: 2.4rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.5px;
        }}
        .hero-sub {{
            font-size: 1rem;
            opacity: 0.75;
            margin: 0.2rem 0 0;
        }}
        .hero-apy {{
            font-size: 3rem;
            font-weight: 800;
            color: {secondary};
            margin: 1rem 0 0;
            line-height: 1;
        }}
        .hero-cap {{
            font-size: 1.1rem;
            opacity: 0.8;
            margin: 0.3rem 0 0;
        }}

        /* Metric cards */
        .metric-row {{
            display: flex;
            gap: 12px;
            margin: 1rem 0;
        }}
        .metric-card {{
            flex: 1;
            background: white;
            border: 1px solid #e8ecef;
            border-radius: 10px;
            padding: 1rem 1.2rem;
        }}
        .metric-label {{
            font-size: 0.75rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }}
        .metric-value {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #1a1a1a;
        }}
        .metric-sub {{
            font-size: 0.75rem;
            color: #999;
            margin-top: 2px;
        }}

        /* Accent bar */
        .accent-bar {{
            height: 4px;
            background: linear-gradient(90deg, {primary}, {secondary});
            border-radius: 2px;
            margin: 1rem 0;
        }}

        /* Section headers */
        .section-header {{
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #888;
            margin: 1.5rem 0 0.75rem;
        }}

        /* Confidence range bar */
        .conf-track {{
            background: #f0f2f5;
            border-radius: 6px;
            height: 12px;
            position: relative;
            margin: 0.5rem 0;
        }}
        .conf-fill {{
            background: linear-gradient(90deg, {primary}88, {primary});
            border-radius: 6px;
            height: 12px;
        }}

        /* Stat pill */
        .stat-pill {{
            display: inline-block;
            background: #f5f7fa;
            border: 1px solid #e8ecef;
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.8rem;
            margin: 3px;
            color: #333;
        }}
        .stat-pill strong {{
            color: {primary};
        }}

        /* Comp table row */
        .comp-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            border-radius: 8px;
            margin: 4px 0;
            background: #f8f9fa;
            font-size: 0.88rem;
        }}
        .comp-row:hover {{ background: #f0f2f5; }}
        .comp-name {{ font-weight: 600; color: #1a1a1a; }}
        .comp-meta {{ color: #888; font-size: 0.78rem; }}
        .comp-pct {{
            font-weight: 700;
            font-size: 1rem;
            color: {primary};
        }}

        /* Hide streamlit chrome */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stDeployButton {{display: none;}}

        /* Input styling */
        .stSelectbox label, .stTextInput label {{
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            color: #666 !important;
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
def get_player_list():
    """Return sorted list of all players in the contracts data."""
    contracts = load_contracts()
    if contracts.empty:
        return []
    return sorted(contracts["player_name"].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def get_player_info(player_name: str):
    """Return most recent contract info for a player."""
    contracts = load_contracts()
    if contracts.empty:
        return {}
    rows = contracts[contracts["player_name"] == player_name].sort_values(
        "signing_year", ascending=False
    )
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


@st.cache_data(show_spinner=False)
def cached_predict(player_name: str, position: str, signing_year: int):
    """Cache predictions so re-selecting the same player is instant."""
    return predict_contract(player_name, position, signing_year)


@st.cache_data(show_spinner=False)
def cached_comps(position: str, predicted_cap_pct: float, signing_year: int):
    return find_comps(position, predicted_cap_pct, signing_year, n=6)


# ── Chart builders ─────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string for Plotly compatibility."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def make_confidence_chart(
    predicted: float,
    low: float,
    high: float,
    position: str,
    primary: str,
) -> go.Figure:
    """Horizontal gauge showing prediction and confidence range."""
    pos_range = {"QB": (8, 26), "WR": (2, 16), "RB": (1, 11), "TE": (1, 10)}
    x_min, x_max = pos_range.get(position, (1, 20))

    fig = go.Figure()

    # Background track
    fig.add_shape(type="rect", x0=x_min, x1=x_max, y0=0.3, y1=0.7,
                  fillcolor="#f0f2f5", line_width=0)

    # Confidence band
    fig.add_shape(type="rect", x0=low, x1=high, y0=0.25, y1=0.75,
                  fillcolor=hex_to_rgba(primary, 0.2),
                  line=dict(color=hex_to_rgba(primary, 0.4), width=1))

    # Prediction line
    fig.add_shape(type="line", x0=predicted, x1=predicted, y0=0.1, y1=0.9,
                  line=dict(color=primary, width=3))

    # Annotation
    fig.add_annotation(x=predicted, y=1.1, text=f"{predicted:.1f}%",
                       showarrow=False, font=dict(size=14, color=primary, family="Arial Black"))
    fig.add_annotation(x=(low + high) / 2, y=-0.2,
                       text=f"Range: {low:.1f}% – {high:.1f}%",
                       showarrow=False, font=dict(size=11, color="#888"))

    fig.update_layout(
        height=100, margin=dict(l=20, r=20, t=30, b=30),
        xaxis=dict(range=[x_min, x_max], showgrid=False,
                   ticksuffix="%", tickfont=dict(size=10, color="#aaa")),
        yaxis=dict(visible=False, range=[-0.5, 1.5]),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def make_oof_chart(position: str, primary: str) -> go.Figure:
    """Scatter plot of predicted vs actual for this position's OOF predictions."""
    oof = load_oof()
    if oof.empty:
        return None

    pos_oof = oof[oof["position"] == position].copy()
    if pos_oof.empty:
        return None

    mae = pos_oof["abs_error_pct"].mean()

    fig = go.Figure()

    # Perfect prediction line
    axis_max = max(pos_oof["y_true_pct"].max(), pos_oof["y_pred_pct"].max()) + 1
    axis_min = max(0, min(pos_oof["y_true_pct"].min(), pos_oof["y_pred_pct"].min()) - 1)
    fig.add_trace(go.Scatter(
        x=[axis_min, axis_max], y=[axis_min, axis_max],
        mode="lines", line=dict(color="#ddd", width=1, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))

    # OOF scatter
    fig.add_trace(go.Scatter(
        x=pos_oof["y_true_pct"],
        y=pos_oof["y_pred_pct"],
        mode="markers",
        marker=dict(color=primary, size=7, opacity=0.6,
                    line=dict(color="white", width=1)),
        text=pos_oof["player_name"] + " (" + pos_oof["signing_year"].astype(str) + ")",
        hovertemplate="<b>%{text}</b><br>Actual: %{x:.1f}%<br>Predicted: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(title="Actual cap %", ticksuffix="%",
                   range=[axis_min, axis_max], showgrid=True,
                   gridcolor="#f0f2f5", zeroline=False),
        yaxis=dict(title="Predicted cap %", ticksuffix="%",
                   range=[axis_min, axis_max], showgrid=True,
                   gridcolor="#f0f2f5", zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text=f"{position} model accuracy (LOYO CV) — MAE {mae:.2f}%",
                   font=dict(size=12, color="#666"), x=0),
    )
    return fig


def make_importance_chart(
    result: dict,
    primary: str,
    top_n: int = 12,
) -> go.Figure | None:
    """Bar chart of feature values used, sorted by magnitude."""
    features = result.get("features_used", {})
    if not features:
        return None

    # Show only _mean features for cleanliness (the core signal)
    mean_feats = {k: v for k, v in features.items()
                  if k.endswith("_mean") and not np.isnan(v)}
    if not mean_feats:
        mean_feats = {k: v for k, v in features.items() if not np.isnan(float(v))}

    # Normalise feature names for display
    def fmt_name(name: str) -> str:
        name = name.replace("_mean", "").replace("_last", "").replace("_trend", "")
        name = name.replace("_", " ")
        return name.title()

    items = sorted(mean_feats.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    labels = [fmt_name(k) for k, _ in items]
    values = [v for _, v in items]

    colors = [primary if v >= 0 else "#e74c3c" for v in values]

    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=labels[::-1],
        orientation="h",
        marker_color=colors[::-1],
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(200, top_n * 24),
        margin=dict(l=10, r=20, t=30, b=20),
        xaxis=dict(showgrid=True, gridcolor="#f0f2f5", zeroline=True,
                   zerolinecolor="#ddd", title="Value (3-yr mean)"),
        yaxis=dict(showgrid=False),
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text="Stats used in prediction (3-yr averages)",
                   font=dict(size=12, color="#666"), x=0),
    )
    return fig


# ── Formatting helpers ─────────────────────────────────────────────────────────
def fmt_money(value: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def fmt_stat(key: str, value: float) -> str:
    """Format a stat value sensibly based on its name."""
    if key.endswith("_games"):
        return str(int(value))
    # Ratio metrics — show as raw number
    if any(x in key for x in ("epa", "pacr", "racr", "dakota", "wopr")):
        return f"{value:.2f}"
    # Share/rate metrics stored as decimals (0.47 = 47%)
    if any(x in key for x in ("target_share", "air_yards_share", "catch_rate", "completion_pct")):
        if abs(value) <= 1.5:
            return f"{value * 100:.1f}%"
        return f"{value:.1f}%"
    # Other pct columns
    if "pct" in key:
        if abs(value) <= 1.5:
            return f"{value * 100:.1f}%"
        return f"{value:.1f}%"
    if value > 100:
        return f"{value:,.0f}"
    return f"{value:.1f}"


def fmt_key(key: str) -> str:
    """Human-readable stat label."""
    key = key.replace("_mean", "").replace("_last", " (last yr)").replace("_trend", " (trend)")
    key = key.replace("_", " ")
    return key.strip().title()


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    available_positions = list_available_positions()

    if not available_positions:
        st.error("No trained models found. Run `python notebooks/model.py` first.")
        st.stop()

    # ── Search bar ─────────────────────────────────────────────────────────────
    st.markdown("## 🏈 NFL Contract Predictor")
    st.markdown("Search for any NFL player to predict their next contract value as a % of the salary cap.")

    col_search, col_pos, col_year = st.columns([3, 1, 1])

    with col_search:
        player_list = get_player_list()
        player_name = st.selectbox(
            "Player name",
            options=[""] + player_list,
            index=0,
            placeholder="Search player...",
        )

    with col_pos:
        position = st.selectbox(
            "Position",
            options=available_positions,
            index=0,
        )

    with col_year:
        signing_year = st.selectbox(
            "Contract year",
            options=list(range(2026, 2013, -1)),
            index=0,
        )

    if not player_name:
        st.markdown("---")
        st.markdown("#### How it works")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**1. Search a player**\nSelect any QB, WR, RB, or TE from the dropdown.")
        with c2:
            st.markdown("**2. Set contract year**\nChoose the year the contract would be signed.")
        with c3:
            st.markdown("**3. See the prediction**\nGet predicted APY, cap %, comparable contracts, and model accuracy.")
        st.stop()

    # ── Run prediction ─────────────────────────────────────────────────────────
    with st.spinner(f"Fetching stats for {player_name}..."):
        try:
            result = cached_predict(player_name, position, signing_year)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # ── Team colors ────────────────────────────────────────────────────────────
    player_info = get_player_info(player_name)
    team = player_info.get("team", "")
    primary, secondary = get_team_colors(team)
    inject_css(primary, secondary)

    predicted_cap_pct = result["predicted_cap_pct"]
    predicted_apy     = result["predicted_apy"]
    cap               = result["cap_that_year"]
    conf_low, conf_high = result["confidence_range"]

    # ── Hero banner ────────────────────────────────────────────────────────────
    team_display = f" · {team}" if team else ""
    st.markdown(f"""
    <div class="hero">
        <p class="hero-name">{player_name}</p>
        <p class="hero-sub">{position}{team_display} · {signing_year} contract projection</p>
        <p class="hero-apy">{fmt_money(predicted_apy)} / yr</p>
        <p class="hero-cap">{predicted_cap_pct:.1f}% of the salary cap · {fmt_money(cap)} cap</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric row ─────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Predicted APY", fmt_money(predicted_apy))
    with m2:
        st.metric("Cap %", f"{predicted_cap_pct:.2f}%")
    with m3:
        st.metric("Confidence range", f"{conf_low:.1f}% – {conf_high:.1f}%")
    with m4:
        results = load_results_summary()
        model_name = result.get("model_used", "—")
        mae = results.get(position, {}).get(model_name, {}).get("mae", None)
        st.metric("Model MAE", f"{mae:.2f}%" if mae else "—", help="Mean absolute error from leave-one-year-out cross-validation")

    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    # ── Two column layout ──────────────────────────────────────────────────────
    left_col, right_col = st.columns([1.1, 1])

    with left_col:

        # Confidence range chart
        st.markdown('<p class="section-header">Confidence range</p>', unsafe_allow_html=True)
        conf_fig = make_confidence_chart(predicted_cap_pct, conf_low, conf_high, position, primary)
        st.plotly_chart(conf_fig, use_container_width=True, config={"displayModeBar": False})

        # Comparable contracts
        st.markdown('<p class="section-header">Comparable historical contracts</p>', unsafe_allow_html=True)
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

        # Model accuracy chart
        st.markdown('<p class="section-header">Model accuracy — {position}</p>'.replace("{position}", position),
                    unsafe_allow_html=True)
        oof_fig = make_oof_chart(position, primary)
        if oof_fig:
            st.plotly_chart(oof_fig, use_container_width=True, config={"displayModeBar": False})

    with right_col:

        # Recent stats used
        st.markdown('<p class="section-header">Stats used in prediction</p>', unsafe_allow_html=True)
        features = result.get("features_used", {})
        missing  = result.get("missing_features", [])

        if features:
            # Separate into meaningful groups — exclude _games (season count, not useful to show)
            mean_feats  = {k: v for k, v in features.items()
                           if k.endswith("_mean")
                           and not np.isnan(float(v))
                           and not k.startswith("games_")}
            other_feats = {k: v for k, v in features.items()
                           if not any(k.endswith(s) for s in ("_mean", "_last", "_trend", "_games"))
                           and not np.isnan(float(v))}

            if mean_feats:
                st.caption(f"3-year averages ({signing_year - 3}–{signing_year - 1})")
                pills_html = ""
                for k, v in list(mean_feats.items())[:12]:
                    label = fmt_key(k)
                    val   = fmt_stat(k, v)
                    pills_html += f'<span class="stat-pill"><strong>{label}:</strong> {val}</span>'
                st.markdown(pills_html, unsafe_allow_html=True)

            if other_feats:
                st.caption("Market context")
                pills_html = ""
                for k, v in other_feats.items():
                    label = fmt_key(k)
                    val   = fmt_stat(k, v)
                    pills_html += f'<span class="stat-pill"><strong>{label}:</strong> {val}</span>'
                st.markdown(pills_html, unsafe_allow_html=True)

            if missing:
                with st.expander(f"{len(missing)} features missing (imputed with median)"):
                    st.caption(", ".join(missing[:20]))

        else:
            st.warning(
                f"No stats found for {player_name} in seasons "
                f"{signing_year - 3}–{signing_year - 1}. "
                "Prediction uses median imputation for all stat features."
            )

        # Feature importance chart
        st.markdown('<p class="section-header">What drove the prediction</p>', unsafe_allow_html=True)
        imp_fig = make_importance_chart(result, primary)
        if imp_fig:
            st.plotly_chart(imp_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Feature importance not available — no stats found for this player.")

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"Model: {result.get('model_used', '—')} · "
        f"Training data: OTC contract history + nfl_data_py stats · "
        f"Validated via leave-one-year-out CV · "
        f"Not affiliated with the NFL or any team."
    )


if __name__ == "__main__":
    main()
