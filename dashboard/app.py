"""
dashboard/app.py
================
Streamlit dashboard for Ukraine air raid alert analysis.
Run: streamlit run dashboard/app.py
"""

import os, sys, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR    = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FEATURES_F  = DATA_DIR / "features.parquet"
MODELS_F    = DATA_DIR / "models.pkl"

st.set_page_config(
    page_title="Ukraine Air Raid Analysis",
    page_icon=None, layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ────────────────────────────────────────────────────────────
ACCENT = {
    "blue":   "#5B6FD8",
    "green":  "#34A270",
    "coral":  "#E0697A",
    "amber":  "#D8A23B",
    "purple": "#7C6FD8",
}
TEXT_DARK = "#1A1A2E"
TEXT_MUTE = "#6B7280"
GRID      = "#EEEEF2"

st.markdown(f"""<style>
.stApp {{ background-color: #F4F5FA; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; }}
h1, h2, h3, h4 {{ color: {TEXT_DARK}; font-weight: 700; }}
p, span, label, div {{ color: {TEXT_DARK}; }}

[data-testid="stSidebar"] {{
    background-color: #FFFFFF;
    border-right: 1px solid {GRID};
}}
[data-testid="stSidebar"] * {{ color: {TEXT_DARK}; }}

[data-testid="stSidebar"] div[role="radiogroup"] {{
    margin-left: -1rem;
    margin-right: -1rem;
    width: calc(100% + 2rem);
}}
div[role="radiogroup"] label {{
    padding: 12px 24px;
    border-radius: 0;
    margin-bottom: 2px;
    width: 100%;
}}
div[role="radiogroup"] label:hover {{ background-color: #F8F8FB; }}
div[role="radiogroup"] label:has(input:checked) {{
    background-color: #F0F1F8; font-weight: 600;
    border-left: 3px solid {ACCENT["blue"]};
}}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: #FFFFFF;
    border-radius: 16px !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"] > div {{
    border-radius: 16px !important;
    border-color: {GRID} !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}}

.metric-card, .reflection-card {{
    background:#ffffff; border-radius:16px; padding:22px 24px;
    box-shadow:0 1px 4px rgba(0,0,0,0.05); margin-bottom:14px;
}}
.section-label {{
    font-size: 12px; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; color: {TEXT_MUTE}; margin-bottom: 4px;
}}
.kpi-value {{ font-size: 26px; font-weight: 700; color: {TEXT_DARK}; line-height:1.2; }}
.kpi-label {{ font-size: 13px; color: {TEXT_MUTE}; margin-top: 2px; }}
.kpi-dot {{ width:10px; height:10px; border-radius:50%; margin-bottom:10px; }}

.stTextArea textarea {{ background-color: #FBFBFD; border-radius: 10px; }}
</style>""", unsafe_allow_html=True)


def style_fig(fig, height: int | None = None) -> go.Figure:
    """Apply consistent, theme-safe styling to any plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_DARK, family="Helvetica, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


def kpi_card(col, value: str, label: str, color: str):
    with col.container(border=True):
        st.markdown(f"<div class='kpi-dot' style='background:{color};'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-value'>{value}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-label'>{label}</div>", unsafe_allow_html=True)


def notes_card(col, text: str):
    with col.container(border=True):
        st.markdown("<div class='section-label'>Notes</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:14px;color:{TEXT_DARK};line-height:1.6;'>{text}</div>",
                    unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_panel():
    return pd.read_parquet(FEATURES_F) if FEATURES_F.exists() else None

@st.cache_resource
def load_models():
    if MODELS_F.exists():
        with open(MODELS_F, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data(ttl=300)
def get_live_alerts():
    from maps import fetch_live_alerts
    return fetch_live_alerts(os.getenv("ALERTS_IN_UA_TOKEN",""))


PAGES = ["Overview", "Forecast", "Live Map", "Patterns", "Summary"]

with st.sidebar:
    kse_logo = ROOT / "dashboard" / "assets" / "kse_logo.png"
    hec_logo = ROOT / "dashboard" / "assets" / "hec_logo.png"
    logo_col1, logo_col2 = st.columns([1, 1], gap="small")
    if kse_logo.exists():
        logo_col1.image(str(kse_logo), width=90)
    if hec_logo.exists():
        logo_col2.image(str(hec_logo), width=81)
    st.markdown(
        f"<div style='font-size:13px;color:{TEXT_MUTE};margin-top:6px;margin-bottom:18px;'>Andrii Mokliak</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<hr style='margin:0 0 14px 0;border-color:{GRID};'>", unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")


# ── OVERVIEW ─────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Ukraine Air Raid Alert Analysis")
    st.markdown(f"<p style='color:{TEXT_MUTE};margin-top:-10px;'>Time series forecasting and pattern analysis</p>",
                unsafe_allow_html=True)

    col_main, col_side = st.columns([2, 1])

    with col_main:
        components.html("""
    <div style="
        background:white;
        border-radius:14px;
        padding:22px 24px;
        box-shadow:0 2px 10px rgba(0,0,0,0.06);
        border:1px solid #eee;
        font-family:Arial, sans-serif;
        box-sizing:border-box;
    ">
        <h4 style="margin-top:0;margin-bottom:14px;color:#222;">
            Project Overview
        </h4>

        <p style="color:#444;font-size:14px;line-height:1.6;">
            <b>Project name:</b> Air Raid Alert Forecasting and Pattern Analysis Dashboard
        </p>

        <p style="color:#444;font-size:14px;line-height:1.6;">
            This project analyzes air raid alert patterns across Ukraine and builds machine learning models
            to forecast the probability of an alert occurring within the next 1 day and 7 days. It combines
            historical alert data, live alert visualization, regional pattern analysis, and additional contextual features.
        </p>

        <p style="color:#444;font-size:14px;line-height:1.6;">
            <b>Main dataset:</b>
            <a href="https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset" target="_blank">
                Ukrainian Air Raid Sirens Dataset
            </a>
        </p>

        <p style="color:#444;font-size:14px;line-height:1.6;">
            <b>Additional resources:</b> live alert status data from the alerts.in.ua API, weather data for selected
            Russian launch-site regions, and news-based context features.
        </p>

        <p style="color:#444;font-size:14px;line-height:1.6;">
            <b>Methodology:</b> the project constructs a daily oblast-level panel, engineers lag, rolling, calendar,
            geographic, weather, and news-context features, and compares Logistic Regression and Random Forest models
            using time-aware validation.
        </p>

        <p style="color:#444;font-size:14px;line-height:1.6;margin-bottom:0;">
            The dashboard is structured around four main parts: forecasting results, live alert maps, pattern analysis,
            and a final summary of findings. For readability, <b>Air Raid Alert</b> is shortened to <b>ARA</b>
            throughout the rest of the dashboard.
        </p>
    </div>
    """, height=620)

    with col_side:
        st.markdown("""
        <div class="reflection-card">
        <h4 style="margin-top:0;">Reflection</h4>
       <p style="color:#444;font-size:14px;white-space:pre-line;">
Dear Admission Team,

First of all, I wanted to thank you for the opportunity. I am very motivated to join this program.

This was an interesting task. Throughout the project, I better understood how to instruct LLMs not only to fix bugs in code, but also to maintain consistent logic across the entire structure, especially when interconnecting different scripts.

While fine-tuning the project, I encountered several interesting mistakes, both logical and technical, that turned into valuable lessons. But more on that later.

Hope you enjoy it.

By Andrii Mokliak :)
</p>
        </div>
        """, unsafe_allow_html=True)

    st.caption("KSE x HEC Paris — Andrii Mokliak")


# ── FORECAST ─────────────────────────────────────────────────────────────────
elif page == "Forecast":
    st.title("Alert Probability Forecast")
    panel  = load_panel()
    models = load_models()
    if panel is None:
        st.warning("Run `python main.py data/alerts.csv` first.")
        st.stop()

    oblasts = sorted(panel["oblast"].unique())
    c1, c2 = st.columns([2, 2])
    with c1:
        oblast = st.selectbox("Oblast", oblasts)
    with c2:
        hlabel = st.selectbox("Horizon", ["1 day", "7 days"])
    h = 1 if "1" in hlabel else 7

    sub = panel[panel["oblast"] == oblast].sort_values("date")
    latest = sub.dropna(subset=["lag_1d_count"]).iloc[-1] if len(sub) > 1 else None

    if models and latest is not None and f"{h}d" in models:
        key = f"{h}d"
        gc1, gc2 = st.columns(2)
        for gcol, name, color in [(gc1, "logistic", ACCENT["purple"]), (gc2, "random_forest", ACCENT["coral"])]:
            if name in models[key]:
                with gcol.container(border=True):
                    r = models[key][name]
                    avail = [f for f in r["features"] if f in latest.index]
                    x = latest[avail].fillna(0).values.reshape(1, -1)
                    prob = r["pipeline"].predict_proba(x)[0, 1]
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=round(prob * 100, 1),
                        number={"suffix": "%", "font": {"size": 28, "color": TEXT_DARK}},
                        title={"text": f"{name.replace('_',' ').title()}<br>"
                                       f"<span style='font-size:11px;color:{TEXT_MUTE}'>Forecast probability</span>",
                               "font": {"size": 14, "color": TEXT_DARK}},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                               "steps": [{"range": [0, 40], "color": "#EAF3DE"},
                                         {"range": [40, 70], "color": "#FAEEDA"},
                                         {"range": [70, 100], "color": "#FCEBEB"}],
                               "threshold": {"line": {"color": TEXT_DARK, "width": 2}, "value": 50}},
                    ))
                    style_fig(gauge, height=230)
                    st.plotly_chart(gauge, use_container_width=True)
                    st.caption(f"Model accuracy (CV AUC-ROC): **{r['cv_auc']:.3f}**")
    else:
        st.info("Models not trained yet. Run `python main.py data/alerts.csv` first.")

    st.markdown("<br>", unsafe_allow_html=True)
    chart_col, note_col = st.columns([2, 1])

    with chart_col:
        with st.container(border=True):
            st.markdown("<div class='section-label'>Probability over time</div>", unsafe_allow_html=True)
            if models and f"{h}d" in models and "random_forest" in models[f"{h}d"]:
                r = models[f"{h}d"]["random_forest"]
                avail = [f for f in r["features"] if f in sub.columns]
                proba = r["pipeline"].predict_proba(sub[avail].fillna(0).values)[:, 1]
                sub2 = sub.assign(prob=proba)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sub2["date"], y=sub2["prob"], fill="tozeroy",
                    mode="lines", line=dict(color=ACCENT["blue"], width=1.5),
                    fillcolor="rgba(91,111,216,0.12)", name="Alert probability"))
                fig.add_trace(go.Bar(x=sub2["date"], y=(sub2["alert_count"] > 0).astype(int),
                    name="Actual alert", marker_color="rgba(224,105,122,0.35)", yaxis="y2"))
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis=dict(title="Probability", range=[0, 1]),
                    yaxis2=dict(title="Actual", overlaying="y", side="right", range=[0, 3], showgrid=False),
                    legend=dict(x=0, y=1.12, orientation="h"),
                )
                style_fig(fig, height=320)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trained model available for this horizon.")

    notes_card(note_col,
           "I developed 2 models to forecast ARA: logistic probability and random forest.\n\n"
           "I used the following dataset: https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset. "
           "It contains ARA records since the beginning of the full-scale invasion, but I only used data from 2024 onward.\n\n"
           "I made the assumption that, with the rapid development of drone warfare, the war has entered a different phase. "
           "Because of this, earlier data may be less relevant for training the model.\n\n"
           "The dataset also had an inconsistency related to Kyiv, so I combined Kyiv and Kyiv Oblast.\n\n"
           "At first, I faced a feature selection problem: I had 62 features, creating a feature matrix of approximately "
           "62 x 38,000, which was not optimal for a small project. To solve this, I used a tree-based model to select "
           "the 12 features with the highest explanatory power. You can see them in the summary.\n\n"
           "P.S. I did not set a seed, so the results may slightly differ between runs. Also, across all runs of the code, "
           "weather and news features never made it into the top 12 :(")

    if models and "monthly" in models:
        r_m = models["monthly"].get("rf_regressor")
        if r_m and latest is not None:
            avail = [f for f in r_m["features"] if f in latest.index]
            pred = r_m["pipeline"].predict(latest[avail].fillna(0).values.reshape(1, -1))[0]
            st.markdown("<br>", unsafe_allow_html=True)
            mc1, mc2 = st.columns(2)
            kpi_card(mc1, str(int(round(pred))), "Predicted alerts (next 30 days)", ACCENT["green"])
            kpi_card(mc2, f"±{r_m['cv_mae']:.1f}", "Cross-validated MAE", ACCENT["amber"])


# ── LIVE MAP ─────────────────────────────────────────────────────────────────
elif page == "Live Map":
    st.title("Map View")
    mode = st.radio("Mode", ["Live alerts", "Historical heat map"], horizontal=True)

    try:
        import folium
        from streamlit_folium import st_folium
        HAS_F = True
    except Exception:
        HAS_F = False

    if mode == "Live alerts":
        st.caption("Refreshed every 5 minutes.")
        if HAS_F:
            from maps import live_alert_map
            active = get_live_alerts()  # cached 5 min, shared across all viewers
            m = live_alert_map(active=active)
            with st.container(border=True):
                st_folium(m, width=None, height=520)
        else:
            st.error("pip install streamlit-folium")
    else:
        panel = load_panel()
        if panel is None:
            st.warning("No panel data.")
        elif HAS_F:
            metric = st.selectbox("Metric", ["alert_rate", "avg_duration", "alert_count_total"])
            from maps import risk_heatmap_folium
            m = risk_heatmap_folium(panel, metric=metric)
            with st.container(border=True):
                st_folium(m, width=None, height=520)


# ── PATTERNS ─────────────────────────────────────────────────────────────────
elif page == "Patterns":
    st.title("Pattern Analysis")
    panel  = load_panel()
    models = load_models()
    if panel is None:
        st.warning("No panel data.")
        st.stop()

    intro_col, note_col = st.columns([2, 1])
    with intro_col:
        with st.container(border=True):
            st.markdown("<div class='section-label'>What to look for</div>", unsafe_allow_html=True)
            st.write("These tabs break alert activity down by time of day, weather context, "
                     "persistence (autocorrelation), regional co-occurrence, and alert level. "
                     "Use the Weather tab to see whether launch-site weather or news context "
                     "meaningfully improves the forecast beyond alert history alone.")
    notes_card(note_col,
           "This is the most interesting part: finding patterns.\n\n"
           "Since there was some room for creativity, I decided to test 2 additional factors: weather and news, "
           "and check whether they would improve the model's performance.\n\n"
           "I started with weather because, intuitively, if there is heavy wind or rain, the ARA probability may decrease. "
           "The first logical mistake I encountered was considering weather conditions in Ukraine alone. After some time, "
           "I realized that weather conditions in Lviv would have little correlation with launch conditions in Russia, "
           "and therefore would not help explain ARA variability.\n\n"
           "Instead, I started considering the 5 most common oblasts in Russia used to launch drones and missiles. "
           "I took the average of their wind, temperature, cloud cover, and precipitation, engineered them as features, "
           "and added them to the fundamental 12 features from the base model.\n\n"
           "I \"forced\" the weather features into the model because they would not be ranked in the top 12 by the base model. "
           "To preserve the base model's accuracy and make comparison possible, I created 3 new models: "
           "base + weather, base + news, and base + weather + news.\n\n"
           "This page also provides additional fundamental analysis and other interesting pattern insights.")

    st.markdown("<br>", unsafe_allow_html=True)
    tabs = st.tabs(["Time", "Weather", "Autocorrelation", "Co-occurrence", "Levels"])
    figs = REPORTS_DIR / "figures"

    with tabs[0]:
        for fname, title in [("hour_distribution.png", "Hour of day"),
                              ("dow_pattern.png", "Day of week"),
                              ("monthly_seasonality.png", "Monthly seasonality")]:
            p = figs / fname
            with st.container(border=True):
                if p.exists():
                    st.image(str(p), caption=title, use_container_width=True)
                else:
                    st.info(f"Run the pattern analysis step to generate {fname}")

    with tabs[1]:
        WEATHER_FEATURES = ["wind_speed_max", "precipitation", "cloud_cover", "temperature", "high_wind"]

        if models and "comparison" in models and "base_plus_weather" in models["comparison"]:
            fi_w = models["comparison"]["base_plus_weather"]["feature_importances"]
            weather_share = fi_w[fi_w.index.isin(WEATHER_FEATURES)].sum()
            total_w = fi_w.sum()
            weather_pct = (weather_share / total_w * 100) if total_w > 0 else 0.0

            with st.container(border=True):
                st.metric("Weather's share of importance (base 12 + weather model)", f"{weather_pct:.1f}%")
                st.caption("Computed from the base-12-plus-weather comparison model, not the "
                           "original base model, which excludes weather from its top 12.")

                cat_map = {
                    "weather (all 5)": WEATHER_FEATURES,
                    "lags/rolling": [c for c in fi_w.index if c.startswith(("lag_", "roll_", "nat_"))],
                    "geography": ["zone_east", "zone_central", "zone_west"],
                    "news": ["news_severity", "nato_visit_bin", "peace_talks_bin"],
                }
                cat_scores = {cat: fi_w[fi_w.index.isin(cols)].sum() for cat, cols in cat_map.items()}
                cat_scores["other"] = total_w - sum(cat_scores.values())
                cat_df = pd.DataFrame({"category": cat_scores.keys(), "importance": cat_scores.values()})
                fig_cat = px.bar(cat_df, x="category", y="importance",
                    color_discrete_sequence=[ACCENT["blue"]], title="Model importance by feature category")
                st.plotly_chart(style_fig(fig_cat, height=320), use_container_width=True)

            with st.container(border=True):
                st.markdown("**Individual weather feature importance**")
                weather_fi = fi_w[fi_w.index.isin(WEATHER_FEATURES)].sort_values(ascending=False)
                fig_weather = px.bar(weather_fi.reset_index(), x="index", y=0,
                    color_discrete_sequence=[ACCENT["coral"]],
                    labels={"index": "Feature", "0": "Importance"},
                    title="Importance by individual weather feature")
                st.plotly_chart(style_fig(fig_weather, height=320), use_container_width=True)
        else:
            st.info("Train models first to see weather's variance-explained share.")

        p = figs / "weather_correlation.png"
        if p.exists():
            with st.container(border=True):
                st.image(str(p), use_container_width=True)

        available_weather = [c for c in WEATHER_FEATURES if c in panel.columns and c != "high_wind"]
        if available_weather:
            with st.container(border=True):
                sc1, sc2 = st.columns(2)
                with sc1:
                    ob = st.selectbox("Oblast (scatter)", sorted(panel["oblast"].unique()))
                with sc2:
                    feat = st.selectbox("Weather feature", available_weather)
                sub = panel[panel["oblast"] == ob].dropna(subset=[feat])
                fig = px.scatter(sub, x=feat, y="alert_count", trendline="lowess",
                    color_discrete_sequence=[ACCENT["blue"]],
                    labels={feat: feat.replace("_", " ").title(), "alert_count": "Alerts"},
                    title=f"{feat.replace('_',' ').title()} vs alerts — {ob}")
                st.plotly_chart(style_fig(fig, height=340), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("Does weather or news actually improve predictions?")
            if models and "comparison" in models:
                comp = models["comparison"]
                base_auc = models["1d"]["random_forest"]["cv_auc"]
                rows = [{"model": "Base (12 features)", "cv_auc": base_auc}]
                labels = {
                    "base_plus_weather": "Base + Weather",
                    "base_plus_news":    "Base + News",
                    "base_plus_both":    "Base + Weather + News",
                }
                for k, lbl in labels.items():
                    if k in comp:
                        rows.append({"model": lbl, "cv_auc": comp[k]["cv_auc"]})
                comp_df = pd.DataFrame(rows)
                fig_comp = px.bar(comp_df, x="model", y="cv_auc",
                    color_discrete_sequence=[ACCENT["blue"]],
                    labels={"cv_auc": "CV AUC-ROC", "model": "Model variant"},
                    title="1-day forecast accuracy by feature set")
                fig_comp.update_yaxes(range=[0.5, 1.0])
                fig_comp.add_hline(y=base_auc, line_dash="dash", line_color=ACCENT["coral"],
                    annotation_text="Base model", annotation_position="bottom right")
                st.plotly_chart(style_fig(fig_comp, height=340), use_container_width=True)

                best = comp_df.loc[comp_df["cv_auc"].idxmax()]
                improvement = best["cv_auc"] - base_auc
                if improvement > 0.01:
                    st.success(f"**{best['model']}** improves AUC by {improvement:.3f} over the base model.")
                else:
                    st.info("None of the additions meaningfully improve on the base model.")
            else:
                st.info("Comparison models not found. Retrain to generate them.")

    with tabs[2]:
        ob_ac = st.selectbox("Oblast", sorted(panel["oblast"].unique()), key="ac")
        p = figs / f"autocorrelation_{ob_ac.split()[0]}.png"
        if not p.exists():
            from patterns import plot_autocorrelation
            plot_autocorrelation(panel, ob_ac, figs)
        with st.container(border=True):
            if p.exists():
                st.image(str(p), use_container_width=True)

    with tabs[3]:
        p = figs / "region_cooccurrence.png"
        if p.exists():
            with st.container(border=True):
                st.image(str(p), use_container_width=True)
                st.caption("Pearson r = 1 means two regions are always targeted on the same day.")

    with tabs[4]:
        p = figs / "level_analysis.png"
        with st.container(border=True):
            if p.exists():
                st.image(str(p), use_container_width=True)
        if "raion_frag" in panel.columns:
            with st.container(border=True):
                fig = px.histogram(panel, x="raion_frag", nbins=30,
                    color_discrete_sequence=[ACCENT["blue"]],
                    title="Distribution of raion-alert fraction (0 = oblast-wide, 1 = raion-only)",
                    labels={"raion_frag": "Raion alert fraction"})
                st.plotly_chart(style_fig(fig, height=320), use_container_width=True)


# ── SUMMARY ──────────────────────────────────────────────────────────────────
elif page == "Summary":
    st.title("Key Findings")
    panel  = load_panel()
    models = load_models()
    if panel is None:
        st.warning("No panel data.")
        st.stop()

    total = int(panel["alert_count"].sum())
    days  = panel["date"].nunique()
    obs   = panel["oblast"].nunique()
    top   = panel.groupby("oblast")["alert_count"].sum().idxmax()

    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, f"{total:,}", "Total alerts", ACCENT["blue"])
    kpi_card(k2, str(days), "Days of data", ACCENT["green"])
    kpi_card(k3, str(obs), "Oblasts tracked", ACCENT["coral"])
    kpi_card(k4, top.split()[0], "Most targeted", ACCENT["amber"])

    st.markdown("<br>", unsafe_allow_html=True)
    chart_col, note_col = st.columns([2, 1])

    with chart_col:
        with st.container(border=True):
            if models and "1d" in models and "random_forest" in models["1d"]:
                st.markdown("<div class='section-label'>What drives the forecast (RF importance, 1-day)</div>",
                            unsafe_allow_html=True)
                fi = models["1d"]["random_forest"]["feature_importances"].head(15)
                fig = px.bar(fi[::-1].reset_index(), x=0, y="index", orientation="h",
                    labels={"index": "Feature", 0: "Importance"},
                    color=fi[::-1].values, color_continuous_scale=[[0, "#EEEDFE"], [1, ACCENT["blue"]]])
                fig.update_layout(showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig, height=380), use_container_width=True)
            else:
                st.info("Train models to see feature importance.")

    notes_card(note_col,
               "[Replace this with your key takeaways — which features matter most, "
               "which regions stand out, and anything worth highlighting.]")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<div class='section-label'>Oblast risk ranking</div>", unsafe_allow_html=True)
        risk = (
            panel.groupby("oblast")
            .agg(alert_rate=("had_alert", "mean"), total_alerts=("alert_count", "sum"))
            .sort_values("alert_rate", ascending=False).reset_index()
        )
        fig2 = px.bar(risk, x="oblast", y="alert_rate", color="alert_rate",
            color_continuous_scale="YlOrRd",
            labels={"alert_rate": "Alert day fraction", "oblast": "Oblast"})
        fig2.update_layout(xaxis_tickangle=-45, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig2, height=380), use_container_width=True)

    with st.container(border=True):
        st.markdown("<div class='section-label'>Cumulative national alerts over time</div>", unsafe_allow_html=True)
        nat = panel.groupby("date")["alert_count"].sum().reset_index().sort_values("date")
        nat["cumulative"] = nat["alert_count"].cumsum()
        fig3 = px.area(nat, x="date", y="cumulative", color_discrete_sequence=[ACCENT["blue"]],
            labels={"cumulative": "Cumulative alerts", "date": "Date"})
        st.plotly_chart(style_fig(fig3, height=280), use_container_width=True)

    with st.container(border=True):
        st.markdown("<div class='section-label'>Alert intensity calendar (latest 90 days)</div>", unsafe_allow_html=True)
        recent = nat.tail(90).copy()
        recent["date_dt"] = pd.to_datetime(recent["date"])
        recent["week"] = recent["date_dt"].dt.isocalendar().week.astype(int)
        recent["dow"]  = recent["date_dt"].dt.dayofweek
        pivot_cal = recent.pivot_table(index="dow", columns="week", values="alert_count", aggfunc="sum").fillna(0)
        fig4 = px.imshow(pivot_cal, labels=dict(x="Week", y="Day", color="Alerts"),
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            color_continuous_scale="YlOrRd", aspect="auto")
        st.plotly_chart(style_fig(fig4, height=220), use_container_width=True)

    if "unique_raions_hit" in panel.columns:
        with st.container(border=True):
            st.markdown("<div class='section-label'>Attack fragmentation over time</div>", unsafe_allow_html=True)
            frag = panel.groupby("date")["unique_raions_hit"].sum().reset_index()
            frag["roll14"] = frag["unique_raions_hit"].rolling(14, min_periods=1).mean()
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(x=frag["date"], y=frag["unique_raions_hit"],
                marker_color="rgba(91,111,216,0.35)", name="Daily raions hit"))
            fig5.add_trace(go.Scatter(x=frag["date"], y=frag["roll14"],
                line=dict(color=ACCENT["coral"], width=2), name="14-day average"))
            fig5.update_layout(legend=dict(x=0, y=1.12, orientation="h"),
                xaxis_title="Date", yaxis_title="Unique raions hit")
            st.plotly_chart(style_fig(fig5, height=280), use_container_width=True)
