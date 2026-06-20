"""
dashboard/app.py
================
4-page Streamlit dashboard.
Run: streamlit run dashboard/app.py
"""

import os, sys, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR    = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FEATURES_F  = DATA_DIR / "features.parquet"
MODELS_F    = DATA_DIR / "models.pkl"

st.set_page_config(
    page_title="Ukraine Air Raid Analysis",
    page_icon="🇺🇦", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""<style>
.metric-card{background:#f8f8ff;border-radius:12px;padding:16px 20px;
    border-left:4px solid #534AB7;margin-bottom:10px;}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_panel():
    return pd.read_parquet(FEATURES_F) if FEATURES_F.exists() else None

@st.cache_resource
def load_models():
    if MODELS_F.exists():
        with open(MODELS_F,"rb") as f: return pickle.load(f)
    return None

@st.cache_data(ttl=300)
def get_live_alerts():
    from maps import fetch_live_alerts
    return fetch_live_alerts(os.getenv("UKRAINE_ALARM_API_KEY",""))


PAGES = ["🎯 Forecast","🗺️ Live Map","🔍 Patterns","📊 Summary"]
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/49/Flag_of_Ukraine.svg", width=80)
    st.title("Ukraine Air Raid\nAnalysis")
    st.markdown("---")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("Open-Meteo · NewsAPI · ukrainealarm.com")


# ── PAGE 1: FORECAST ──────────────────────────────────────────────────────
if page == PAGES[0]:
    st.title("🎯 Alert Probability Forecast")
    panel  = load_panel()
    models = load_models()
    if panel is None:
        st.warning("Run `python main.py data/alerts.csv` first."); st.stop()

    oblasts = sorted(panel["oblast"].unique())
    c1, c2 = st.columns([2,2])
    with c1: oblast = st.selectbox("Oblast", oblasts)
    with c2: hlabel = st.selectbox("Horizon", ["1 day","7 days"])
    h = 1 if "1" in hlabel else 7

    sub = panel[panel["oblast"]==oblast].sort_values("date")
    latest = sub.dropna(subset=["lag_1d_count"]).iloc[-1] if len(sub) > 1 else None

    if models and latest is not None:
        key = f"{h}d"
        if key in models:
            c1, c2 = st.columns(2)
            for col, name, color in [(c1,"logistic","#534AB7"),(c2,"random_forest","#D85A30")]:
                if name in models[key]:
                    r   = models[key][name]
                    avail = [f for f in r["features"] if f in latest.index]
                    x   = latest[avail].fillna(0).values.reshape(1,-1)
                    prob = r["pipeline"].predict_proba(x)[0,1]
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=round(prob*100,1),
                        number={"suffix":"%","font":{"size":28}},
                        title={"text":name.replace("_"," ").title(),"font":{"size":14}},
                        gauge={"axis":{"range":[0,100]},"bar":{"color":color},
                               "steps":[{"range":[0,40],"color":"#EAF3DE"},
                                        {"range":[40,70],"color":"#FAEEDA"},
                                        {"range":[70,100],"color":"#FCEBEB"}],
                               "threshold":{"line":{"color":"black","width":2},"value":50}},
                    ))
                    gauge.update_layout(height=220, margin=dict(l=20,r=20,t=40,b=10))
                    col.plotly_chart(gauge, use_container_width=True)
    else:
        st.info("Models not trained yet. Run `python src/models.py`.")

    # Probability timeline
    if models and f"{h}d" in models and "random_forest" in models[f"{h}d"]:
        r     = models[f"{h}d"]["random_forest"]
        avail = [f for f in r["features"] if f in sub.columns]
        proba = r["pipeline"].predict_proba(sub[avail].fillna(0).values)[:,1]
        sub2  = sub.assign(prob=proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub2["date"],y=sub2["prob"],fill="tozeroy",
            mode="lines",line=dict(color="#534AB7",width=1.5),
            fillcolor="rgba(83,74,183,0.15)",name="Alert prob"))
        fig.add_trace(go.Bar(x=sub2["date"],y=(sub2["alert_count"]>0).astype(int),
            name="Actual alert",marker_color="rgba(216,90,48,0.3)",yaxis="y2"))
        fig.update_layout(
            xaxis_title="Date",
            yaxis=dict(title="Probability",range=[0,1]),
            yaxis2=dict(title="Actual",overlaying="y",side="right",range=[0,3],showgrid=False),
            legend=dict(x=0,y=1.1,orientation="h"),height=300,
            margin=dict(l=0,r=0,t=20,b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Monthly forecast
    if models and "monthly" in models:
        r_m   = models["monthly"].get("rf_regressor")
        if r_m and latest is not None:
            avail = [f for f in r_m["features"] if f in latest.index]
            pred  = r_m["pipeline"].predict(latest[avail].fillna(0).values.reshape(1,-1))[0]
            st.markdown("**Monthly total forecast (next 30 days)**")
            mc1, mc2 = st.columns(2)
            mc1.metric("Predicted alerts", int(round(pred)))
            mc2.metric("CV MAE", f"±{r_m['cv_mae']:.1f}")


# ── PAGE 2: LIVE MAP ──────────────────────────────────────────────────────
elif page == PAGES[1]:
    st.title("🗺️ Map View")
    mode = st.radio("Mode", ["Live alerts","Historical heat map"], horizontal=True)

    try:
        import folium
        from streamlit_folium import st_folium
        HAS_F = True
    except:
        HAS_F = False

    if mode == "Live alerts":
        st.caption("Refreshed every 5 min from ukrainealarm.com")
        if HAS_F:
            from maps import live_alert_map
            m = live_alert_map(os.getenv("UKRAINE_ALARM_API_KEY",""))
            st_folium(m, width=None, height=520)
        else:
            st.error("pip install streamlit-folium")
    else:
        panel = load_panel()
        if panel is None:
            st.warning("No panel data.")
        elif HAS_F:
            metric = st.selectbox("Metric",["alert_rate","avg_duration","alert_count_total"])
            from maps import risk_heatmap_folium
            m = risk_heatmap_folium(panel, metric=metric)
            st_folium(m, width=None, height=520)


# ── PAGE 3: PATTERNS ──────────────────────────────────────────────────────
elif page == PAGES[2]:
    st.title("🔍 Pattern Analysis")
    panel = load_panel()
    if panel is None:
        st.warning("No panel data."); st.stop()

    tabs = st.tabs(["Time","Weather","Autocorrelation","Co-occurrence","Levels"])
    figs = REPORTS_DIR / "figures"

    with tabs[0]:
        for fname, title in [("hour_distribution.png","Hour of day"),
                              ("dow_pattern.png","Day of week"),
                              ("monthly_seasonality.png","Monthly seasonality")]:
            p = figs / fname
            if p.exists(): st.image(str(p), caption=title, use_container_width=True)
            else: st.info(f"Run `python src/patterns.py` to generate {fname}")

    with tabs[1]:
        p = figs / "weather_correlation.png"
        if p.exists(): st.image(str(p), use_container_width=True)
        if "wind_speed_max" in panel.columns:
            ob = st.selectbox("Oblast (scatter)", sorted(panel["oblast"].unique()))
            sub = panel[panel["oblast"]==ob].dropna(subset=["wind_speed_max"])
            fig = px.scatter(sub, x="wind_speed_max", y="alert_count",
                trendline="lowess", color_discrete_sequence=["#534AB7"],
                labels={"wind_speed_max":"Max wind (km/h)","alert_count":"Alerts"},
                title=f"Wind vs alerts — {ob}")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        ob_ac = st.selectbox("Oblast", sorted(panel["oblast"].unique()), key="ac")
        p = figs / f"autocorrelation_{ob_ac.split()[0]}.png"
        if not p.exists():
            from patterns import plot_autocorrelation
            plot_autocorrelation(panel, ob_ac, figs)
        if p.exists(): st.image(str(p), use_container_width=True)

    with tabs[3]:
        p = figs / "region_cooccurrence.png"
        if p.exists():
            st.image(str(p), use_container_width=True)
            st.caption("Pearson r=1 → always targeted on same day")

    with tabs[4]:
        p = figs / "level_analysis.png"
        if p.exists(): st.image(str(p), use_container_width=True)
        # Interactive: raion breakdown
        if "raion_frag" in panel.columns:
            fig = px.histogram(panel, x="raion_frag", nbins=30,
                color_discrete_sequence=["#534AB7"],
                title="Distribution of raion-alert fraction (0=oblast-wide, 1=raion-only)",
                labels={"raion_frag":"Raion alert fraction"})
            st.plotly_chart(fig, use_container_width=True)


# ── PAGE 4: SUMMARY ───────────────────────────────────────────────────────
elif page == PAGES[3]:
    st.title("📊 Key Findings & Creative Visuals")
    panel  = load_panel()
    models = load_models()
    if panel is None:
        st.warning("No panel data."); st.stop()

    total = int(panel["alert_count"].sum())
    days  = panel["date"].nunique()
    obs   = panel["oblast"].nunique()
    top   = panel.groupby("oblast")["alert_count"].sum().idxmax()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total alerts",f"{total:,}")
    c2.metric("Days of data", days)
    c3.metric("Oblasts",obs)
    c4.metric("Most targeted",top.split()[0])

    st.markdown("---")

    # Feature importance
    if models and "1d" in models and "random_forest" in models["1d"]:
        st.subheader("What drives the forecast? (RF importance, 1-day)")
        fi = models["1d"]["random_forest"]["feature_importances"].head(15)
        fig = px.bar(fi[::-1].reset_index(), x=0, y="index", orientation="h",
            labels={"index":"Feature",0:"Importance"},
            color=fi[::-1].values, color_continuous_scale=[[0,"#EEEDFE"],[1,"#534AB7"]],
            title="Random Forest feature importance")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Risk ranking
    st.subheader("Oblast risk ranking")
    risk = (
        panel.groupby("oblast")
        .agg(alert_rate=("had_alert","mean"), total_alerts=("alert_count","sum"))
        .sort_values("alert_rate", ascending=False).reset_index()
    )
    fig2 = px.bar(risk, x="oblast", y="alert_rate",
        color="alert_rate", color_continuous_scale="YlOrRd",
        labels={"alert_rate":"Alert day %","oblast":"Oblast"},
        title="Fraction of days with ≥1 alert")
    fig2.update_layout(xaxis_tickangle=-45, height=380, showlegend=False,
                       coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Cumulative timeline
    st.subheader("Cumulative national alerts over time")
    nat = panel.groupby("date")["alert_count"].sum().reset_index().sort_values("date")
    nat["cumulative"] = nat["alert_count"].cumsum()
    fig3 = px.area(nat, x="date", y="cumulative",
        color_discrete_sequence=["#534AB7"],
        labels={"cumulative":"Cumulative alerts","date":"Date"})
    fig3.update_layout(height=280)
    st.plotly_chart(fig3, use_container_width=True)

    # Calendar heatmap
    st.subheader("Alert intensity calendar (latest 90 days)")
    recent = nat.tail(90).copy()
    recent["date_dt"] = pd.to_datetime(recent["date"])
    recent["week"]    = recent["date_dt"].dt.isocalendar().week.astype(int)
    recent["dow"]     = recent["date_dt"].dt.dayofweek
    pivot_cal = recent.pivot_table(index="dow",columns="week",
                                    values="alert_count",aggfunc="sum").fillna(0)
    fig4 = px.imshow(pivot_cal,
        labels=dict(x="Week",y="Day",color="Alerts"),
        y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        color_continuous_scale="YlOrRd",aspect="auto",
        title="Alert intensity heatmap")
    fig4.update_layout(height=220)
    st.plotly_chart(fig4, use_container_width=True)

    # Raion fragmentation over time
    if "unique_raions_hit" in panel.columns:
        st.subheader("Attack fragmentation over time")
        frag = panel.groupby("date")["unique_raions_hit"].sum().reset_index()
        frag["roll14"] = frag["unique_raions_hit"].rolling(14,min_periods=1).mean()
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=frag["date"],y=frag["unique_raions_hit"],
            marker_color="rgba(83,74,183,0.35)",name="Daily raions hit"))
        fig5.add_trace(go.Scatter(x=frag["date"],y=frag["roll14"],
            line=dict(color="#D85A30",width=2),name="14d average"))
        fig5.update_layout(height=280,legend=dict(x=0,y=1.1,orientation="h"),
            xaxis_title="Date",yaxis_title="Unique raions hit")
        st.plotly_chart(fig5, use_container_width=True)
