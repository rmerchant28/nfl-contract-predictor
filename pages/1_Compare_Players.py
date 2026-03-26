"""
Player Comparison — dark mode
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predict import predict_contract, list_available_positions

st.set_page_config(
    page_title="Compare Players — NFL Contract Predictor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Dark palette (mirrors app.py) ──────────────────────────────────────────────
BG      = "#0d1117"
SURFACE = "#161b22"
SURFACE2= "#21262d"
BORDER  = "#30363d"
TEXT    = "#c9d1d9"
SUBTEXT = "#8b949e"
ACCENT  = "#f0b429"

PLAYER_COLORS = ["#f0b429", "#4fc3f7", "#69f0ae", "#ce93d8"]

DISPLAY_STATS = {
    "QB": [
        ("pass_yards","Pass Yards"),("pass_tds","TDs"),("interceptions","INTs"),
        ("completion_pct","Comp %"),("attempts_per_game","Att/G"),
        ("sacks_taken","Sacks"),("passing_epa","EPA"),
        ("games","Games"),("games_missed","Missed"),
    ],
    "WR": [
        ("targets","Targets"),("receptions","Rec"),("rec_yards","Yards"),
        ("rec_tds","TDs"),("catch_rate","Catch %"),("yards_per_reception","YPR"),
        ("targets_per_game","Tgts/G"),("receiving_epa","EPA"),
        ("games","Games"),("games_missed","Missed"),
    ],
    "RB": [
        ("rush_attempts","Carries"),("rush_yards","Rush Yds"),("rush_tds","Rush TDs"),
        ("yards_per_carry","YPC"),("receptions","Rec"),("rec_yards","Rec Yds"),
        ("rec_tds","Rec TDs"),("games","Games"),("games_missed","Missed"),
    ],
    "TE": [
        ("targets","Targets"),("receptions","Rec"),("rec_yards","Yards"),
        ("rec_tds","TDs"),("catch_rate","Catch %"),("yards_per_reception","YPR"),
        ("receiving_epa","EPA"),("games","Games"),("games_missed","Missed"),
    ],
}

RADAR_STATS = {
    "QB": [("pass_yards","Yds"),("pass_tds","TDs"),("completion_pct","Comp%"),
           ("passing_epa","EPA"),("attempts_per_game","Att/G")],
    "WR": [("rec_yards","Yds"),("rec_tds","TDs"),("catch_rate","Catch%"),
           ("targets_per_game","Tgts/G"),("receiving_epa","EPA")],
    "RB": [("rush_yards","Rush Yds"),("rush_tds","Rush TDs"),("yards_per_carry","YPC"),
           ("receptions","Rec"),("rec_yards","Rec Yds")],
    "TE": [("rec_yards","Yds"),("rec_tds","TDs"),("catch_rate","Catch%"),
           ("targets_per_game","Tgts/G"),("receiving_epa","EPA")],
}


def inject_css():
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
      [data-testid="stMetricLabel"] {{ color: {SUBTEXT} !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.05em; }}
      #MainMenu {{ visibility: hidden; }} footer {{ visibility: hidden; }} .stDeployButton {{ display: none; }}
      [data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
      .stSelectbox label, .stNumberInput label {{
          font-size: 0.72rem !important; font-weight: 700 !important;
          text-transform: uppercase !important; color: {SUBTEXT} !important;
      }}
    </style>
    """, unsafe_allow_html=True)


inject_css()

_DARK = dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(color=TEXT, size=11),
)


def hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def fmt_money(v):
    return f"${v/1e6:.1f}M" if v >= 1e6 else f"${v:,.0f}"


def fmt_val(col, val):
    if pd.isna(val): return "—"
    v = float(val)
    if col in ("completion_pct","catch_rate"): return f"{v*100:.1f}%" if v<=1.5 else f"{v:.1f}%"
    if col in ("passing_epa","receiving_epa","rush_epa"): return f"{v:+.1f}"
    if col in ("yards_per_carry","yards_per_reception","attempts_per_game","targets_per_game"): return f"{v:.1f}"
    return f"{v:.0f}"


@st.cache_data(show_spinner=False)
def load_stat_csv(stat_type):
    path = ROOT / "data" / "raw" / f"pfr_{stat_type}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def get_seasons(player_name, position, signing_year):
    import unicodedata
    WINDOW = 3
    seasons = list(range(signing_year - WINDOW, signing_year))

    def norm(n):
        n = unicodedata.normalize("NFKD", str(n)).encode("ascii","ignore").decode("ascii")
        return n.strip().lower()

    def key(n):
        parts = norm(n).replace(".", " ").split()
        return f"{parts[0][0]}.{parts[-1]}" if len(parts) >= 2 else norm(n)

    pn, pk = norm(player_name), key(player_name)

    def filt(df):
        if df.empty or "player_name_norm" not in df.columns: return pd.DataFrame()
        # Try exact normalised name first; only fall back to name_key if nothing found.
        # Using OR caused false positives (e.g. "a.brown" matching A.J. Brown AND Amon-Ra St. Brown).
        exact = df[(df["player_name_norm"] == pn) & df["season"].isin(seasons)]
        if not exact.empty:
            return exact.copy()
        if "name_key" in df.columns:
            return df[(df["name_key"] == pk) & df["season"].isin(seasons)].copy()
        return pd.DataFrame()

    if position == "QB":
        raw = filt(load_stat_csv("passing"))
    elif position in ("WR","TE"):
        raw = filt(load_stat_csv("receiving"))
    elif position == "RB":
        rush = filt(load_stat_csv("rushing"))
        recv = filt(load_stat_csv("receiving"))
        if rush.empty: raw = recv
        elif recv.empty: raw = rush
        else:
            rcols = [c for c in ["season","targets","receptions","rec_yards","rec_tds","catch_rate","yards_per_reception"] if c in recv.columns]
            raw = rush.merge(recv[rcols], on="season", how="left", suffixes=("","_r"))
    else:
        raw = pd.DataFrame()

    return raw.sort_values("season").reset_index(drop=True) if not raw.empty else pd.DataFrame()


@st.cache_data(show_spinner=False)
def active_players():
    out = set()
    for f in ("pfr_passing","pfr_receiving","pfr_rushing"):
        p = ROOT/"data"/"raw"/f"{f}.csv"
        if not p.exists(): continue
        df = pd.read_csv(p, usecols=["player_name","season"])
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
        out.update(df[df["season"]>=2023]["player_name"].dropna())
    return sorted(out)


@st.cache_data(show_spinner=False)
def run_predict(name, pos, yr):
    try:
        return predict_contract(name, pos, yr)
    except Exception as e:
        return {"error": str(e)}


# ── Charts ────────────────────────────────────────────────────────────────────
def radar_chart(players_data, signing_year):
    valid = [p for p in players_data if not p["df"].empty]
    if len(valid) < 2: return None
    from collections import Counter
    pos = Counter(p["pos"] for p in valid).most_common(1)[0][0]
    rcols = RADAR_STATS.get(pos, [])
    if not rcols: return None

    avgs = []
    for p in valid:
        a = {}
        for col, _ in rcols:
            if col in p["df"].columns:
                vals = pd.to_numeric(p["df"][col], errors="coerce").dropna()
                a[col] = float(vals.mean()) if len(vals) else 0.0
            else:
                a[col] = 0.0
        avgs.append({**p, "avgs": a})

    cols   = [c for c,_ in rcols]
    labels = [l for _,l in rcols]
    cmx    = {c: max(a["avgs"].get(c,0) for a in avgs) for c in cols}

    fig = go.Figure()
    for a in avgs:
        vals = [round(100*a["avgs"].get(c,0)/cmx[c],1) if cmx[c]>0 else 0 for c in cols]
        vc = vals + [vals[0]]
        lc = labels + [labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=vc, theta=lc, fill="toself",
            fillcolor=hex_to_rgba(a["color"], 0.16),
            line=dict(color=a["color"], width=2),
            name=a["name"],
            hovertemplate="%{theta}: %{r:.0f}<extra>"+a["name"]+"</extra>",
        ))

    fig.update_layout(
        **_DARK,
        polar=dict(
            bgcolor=SURFACE2,
            radialaxis=dict(visible=True, range=[0,105], showticklabels=False,
                            gridcolor=BORDER, linecolor=BORDER),
            angularaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                             tickfont=dict(color=TEXT, size=11)),
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, font=dict(color=TEXT)),
        height=380,
        title=dict(text=f"3-yr profile ({signing_year-3}–{signing_year-1}, normalized)",
                   font=dict(size=11,color=SUBTEXT), x=0),
    )
    return fig


def contract_bar(players_data):
    valid = [p for p in players_data if p["result"] and "error" not in p["result"]]
    if len(valid) < 2: return None

    names = [p["name"] for p in valid]
    apys  = [p["result"]["predicted_apy"]/1e6 for p in valid]
    caps  = [p["result"]["predicted_cap_pct"] for p in valid]
    clrs  = [p["color"] for p in valid]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="APY ($M)", x=names, y=apys,
        marker=dict(color=clrs, opacity=0.85,
                    line=dict(color=[hex_to_rgba(c, 0.53) for c in clrs], width=1)),
        text=[f"${v:.1f}M" for v in apys], textposition="outside",
        textfont=dict(color=TEXT, size=12),
        yaxis="y",
        hovertemplate="%{x}: $%{y:.1f}M/yr<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        name="Cap %", x=names, y=caps,
        mode="markers+text",
        marker=dict(color=clrs, size=14, symbol="diamond",
                    line=dict(color=BG, width=2)),
        text=[f"{v:.1f}%" for v in caps],
        textposition="top center",
        textfont=dict(color=TEXT, size=11),
        yaxis="y2",
        hovertemplate="%{x}: %{y:.1f}% cap<extra></extra>",
    ))
    fig.update_layout(
        **_DARK,
        height=320,
        margin=dict(l=40, r=60, t=50, b=40),
        yaxis=dict(title="APY ($M)", tickprefix="$", ticksuffix="M",
                   gridcolor=BORDER, zerolinecolor=BORDER, showgrid=True),
        yaxis2=dict(title="Cap %", ticksuffix="%", overlaying="y", side="right",
                    showgrid=False, zerolinecolor=BORDER),
        legend=dict(orientation="h", y=1.12, font=dict(color=TEXT)),
        bargap=0.35,
        title=dict(text="Predicted contract value", font=dict(size=11,color=SUBTEXT), x=0),
    )
    return fig


def stats_table(df, position, signing_year):
    if df.empty: return pd.DataFrame()
    dcols = DISPLAY_STATS.get(position, [])
    rows = []
    for _, row in df.iterrows():
        r = {"Season": int(row["season"])}
        for col, lbl in dcols:
            r[lbl] = fmt_val(col, row.get(col))
        rows.append(r)
    seasons_present = sorted(df["season"].dropna().astype(int).unique())
    if len(seasons_present) >= 2:
        avg_label = f"{seasons_present[0]}–{seasons_present[-1]} avg"
    elif len(seasons_present) == 1:
        avg_label = f"{seasons_present[0]} avg"
    else:
        avg_label = "avg"
    avg = {"Season": avg_label}
    for col, lbl in dcols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            avg[lbl] = fmt_val(col, vals.mean()) if len(vals) else "—"
        else:
            avg[lbl] = "—"
    rows.append(avg)
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    avail_pos = list_available_positions()
    all_names = active_players()

    if not avail_pos:
        st.error("No trained models found. Run `python notebooks/model.py` first.")
        st.stop()

    st.markdown("## ⚖️ Compare Players")
    st.markdown(f"<span style='color:{SUBTEXT};font-size:0.93rem'>"
                "Side-by-side stats and predicted contract values for up to 4 players.</span>",
                unsafe_allow_html=True)
    st.markdown("")

    with st.form("compare_form"):
        col_yr, _ = st.columns([1, 3])
        with col_yr:
            signing_year = st.selectbox("Contract year", options=list(range(2026,2013,-1)), index=0)

        st.markdown(f"<span style='font-size:0.8rem;font-weight:700;text-transform:uppercase;"
                    f"letter-spacing:0.08em;color:{SUBTEXT}'>Select 2–4 players</span>",
                    unsafe_allow_html=True)
        cols = st.columns(4)
        sel = []
        for i, col in enumerate(cols):
            with col:
                n = st.selectbox(f"Player {i+1}", [""] + all_names, index=0, key=f"n{i}")
                p = st.selectbox(f"Position {i+1}", [""] + avail_pos, index=0, key=f"p{i}")
                sel.append((n, p))

        submitted = st.form_submit_button("Compare Players", use_container_width=True, type="primary")

    active = [(n, p) for n, p in sel if n and p]
    if not submitted or len(active) < 2:
        st.markdown(f"<div style='background:{SURFACE};border:1px solid {BORDER};"
                    f"border-radius:12px;padding:1.5rem;text-align:center;color:{SUBTEXT}'>"
                    f"Select at least 2 players above and click <strong>Compare Players</strong>.</div>",
                    unsafe_allow_html=True)
        st.stop()

    # Load data
    players = []
    with st.spinner("Loading stats and running predictions..."):
        for i, (name, pos) in enumerate(active):
            players.append({
                "name":   name,
                "pos":    pos,
                "df":     get_seasons(name, pos, signing_year),
                "result": run_predict(name, pos, signing_year),
                "color":  PLAYER_COLORS[i % len(PLAYER_COLORS)],
            })

    # ── Contract value cards ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Predicted Contract Value")

    card_cols = st.columns(len(players))
    for col, p in zip(card_cols, players):
        r = p["result"]
        with col:
            if "error" in r:
                st.markdown(
                    f"<div style='background:{SURFACE};border:1px solid #ff6b6b44;"
                    f"border-radius:12px;padding:1.2rem'>"
                    f"<div style='color:#ff6b6b;font-weight:700'>{p['name']}</div>"
                    f"<div style='color:{SUBTEXT};font-size:0.82rem;margin-top:0.4rem'>{r['error']}</div>"
                    f"</div>", unsafe_allow_html=True)
            else:
                gtd   = r.get("predicted_guaranteed")
                yrs   = r.get("predicted_years")
                st.markdown(
                    f"<div style='background:{SURFACE};border:1px solid {p['color']}44;"
                    f"border-left:4px solid {p['color']};border-radius:12px;padding:1.2rem 1.4rem'>"
                    f"<div style='font-size:1rem;font-weight:700;color:{TEXT}'>{p['name']}</div>"
                    f"<div style='font-size:0.78rem;color:{SUBTEXT};margin-bottom:0.6rem'>{p['pos']} · {signing_year}</div>"
                    f"<div style='font-size:2rem;font-weight:800;color:{p['color']};line-height:1.1'>"
                    f"{fmt_money(r['predicted_apy'])}/yr</div>"
                    f"<div style='font-size:0.85rem;color:{SUBTEXT};margin-top:2px'>"
                    f"{r['predicted_cap_pct']:.1f}% of cap</div>"
                    + (f"<div style='font-size:0.82rem;color:{TEXT};margin-top:4px'>"
                       f"Gtd: {fmt_money(gtd)}</div>" if gtd else "")
                    + (f"<div style='font-size:0.82rem;color:{SUBTEXT};margin-top:2px'>"
                       f"~{yrs:.0f}-yr deal</div>" if yrs else "")
                    + f"<div style='font-size:0.76rem;color:{SUBTEXT};margin-top:4px'>"
                    f"{r['confidence_range'][0]:.1f}%–{r['confidence_range'][1]:.1f}% range</div>"
                    f"</div>", unsafe_allow_html=True)

    bar_fig = contract_bar(players)
    if bar_fig:
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Radar chart ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Stat Profile")

    rad_fig = radar_chart(players, signing_year)
    if rad_fig:
        left, right = st.columns([1.3, 1])
        with left:
            st.plotly_chart(rad_fig, use_container_width=True, config={"displayModeBar": False})
        with right:
            st.markdown(f"<div style='color:{SUBTEXT};font-size:0.82rem;margin-top:1rem'>"
                        f"Each metric normalized to 100 = best among selected players. "
                        f"Larger area = better overall profile.</div>", unsafe_allow_html=True)
            st.markdown("")
            for p in players:
                r = p["result"]
                if "error" in r or p["df"].empty: continue
                st.markdown(
                    f"<span style='color:{p['color']};font-size:1.1rem'>●</span> "
                    f"**{p['name']}** ({p['pos']}) — "
                    f"{fmt_money(r['predicted_apy'])}/yr · {r['predicted_cap_pct']:.1f}%",
                    unsafe_allow_html=True)
    else:
        st.info("Radar comparison requires players of the same position with stats available.")

    # ── Season-by-season tabs ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Season-by-Season Stats")

    tab_labels = [f"{p['name']} ({p['pos']})" for p in players]
    tabs = st.tabs(tab_labels)

    for tab, p in zip(tabs, players):
        with tab:
            if p["df"].empty:
                st.warning(f"No stats for {p['name']} in {signing_year-3}–{signing_year-1}.")
            else:
                tbl = stats_table(p["df"], p["pos"], signing_year)
                if not tbl.empty:
                    def hl(row):
                        if str(row["Season"]).endswith("avg"):
                            return [f"background-color:{p['color']}18;font-weight:600"] * len(row)
                        return [""] * len(row)
                    st.dataframe(tbl.style.apply(hl, axis=1),
                                 use_container_width=True, hide_index=True)

    # ── Side-by-side averages (same position only) ─────────────────────────────
    valid = [p for p in players if not p["df"].empty]
    if len(valid) >= 2 and len(set(p["pos"] for p in valid)) == 1:
        pos = valid[0]["pos"]
        st.markdown("---")
        st.markdown("### 3-Year Average Side by Side")
        rows = {}
        for col, lbl in DISPLAY_STATS.get(pos, []):
            rows[lbl] = {}
            for p in valid:
                if col in p["df"].columns:
                    vals = pd.to_numeric(p["df"][col], errors="coerce").dropna()
                    rows[lbl][p["name"]] = fmt_val(col, vals.mean()) if len(vals) else "—"
                else:
                    rows[lbl][p["name"]] = "—"
        cdf = pd.DataFrame(rows).T
        cdf.index.name = "Stat"
        st.dataframe(cdf, use_container_width=True)

    st.markdown("---")
    st.caption("Stats: nflreadpy · Contracts: OverTheCap · Not affiliated with the NFL or any team.")


if __name__ == "__main__":
    main()
