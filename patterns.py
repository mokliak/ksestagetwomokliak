"""
src/patterns.py
===============
Module 3: Pattern analysis — temporal, weather, geographic, and creative.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")
P_MAIN  = "#534AB7"
P_WARM  = "#D85A30"
P_LIGHT = "#EEEDFE"


def _save(fig, name: str, d: Path):
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {d / name}")


def plot_hour_distribution(df_raw: pd.DataFrame, output_dir: Path):
    df = df_raw.copy()
    df["hour"] = pd.to_datetime(df["started_at"], utc=True).dt.tz_convert("Europe/Kyiv").dt.hour
    overall = df.groupby("hour").size().reindex(range(24), fill_value=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax = axes[0]
    ax.bar(range(24), overall.values, color=P_MAIN, alpha=0.85, width=0.85)
    ax.axvspan(0, 6, alpha=0.07, color=P_WARM, label="00–06 h (night window)")
    ax.axvspan(22, 24, alpha=0.07, color=P_WARM)
    ax.set_xticks(range(24)); ax.set_xlabel("Hour (Kyiv time)"); ax.set_ylabel("Alerts")
    ax.set_title("Alert start hour – all Ukraine"); ax.legend()

    top_ob = df["oblast"].value_counts().head(6).index
    heat = (
        df[df["oblast"].isin(top_ob)]
        .groupby(["oblast","hour"]).size()
        .unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
    )
    sns.heatmap(heat, ax=axes[1], cmap="YlOrRd", linewidths=0.3,
                cbar_kws={"label":"Alert count"})
    axes[1].set_title("Alert hour by oblast (top 6)"); axes[1].set_xlabel("Hour (Kyiv time)")
    fig.tight_layout()
    _save(fig, "hour_distribution.png", output_dir)


def plot_dow_pattern(df_raw: pd.DataFrame, output_dir: Path):
    df = df_raw.copy()
    df["dow"] = pd.to_datetime(df["started_at"], utc=True).dt.dayofweek
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    cnt  = df["dow"].value_counts().reindex(range(7), fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(days, cnt.values, color=P_MAIN, alpha=0.85)
    axes[0].set_title("Alerts by day of week"); axes[0].set_ylabel("Count")
    norm = cnt / cnt.sum()
    axes[1].bar(days, norm.values, color=P_WARM, alpha=0.85)
    axes[1].axhline(1/7, ls="--", color="gray", label="Uniform (1/7)")
    axes[1].set_title("Relative frequency vs uniform"); axes[1].legend()
    fig.tight_layout()
    _save(fig, "dow_pattern.png", output_dir)


def plot_monthly_seasonality(panel: pd.DataFrame, output_dir: Path):
    monthly = (
        panel.groupby(["year","month"])["alert_count"].sum().reset_index()
    )
    pivot = monthly.pivot(index="month", columns="year", values="alert_count").fillna(0)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for yr in pivot.columns:
        axes[0].plot(range(1,13), pivot[yr], marker="o", label=str(yr), alpha=0.8, lw=1.5)
    axes[0].set_xticks(range(1,13)); axes[0].set_xticklabels(months, rotation=45)
    axes[0].set_ylabel("Total alerts"); axes[0].set_title("Monthly totals by year"); axes[0].legend(fontsize=9)

    avg = pivot.mean(axis=1)
    axes[1].bar(range(1,13), avg.values, color=P_MAIN, alpha=0.85)
    axes[1].set_xticks(range(1,13)); axes[1].set_xticklabels(months, rotation=45)
    axes[1].set_title("Average seasonal profile"); axes[1].set_ylabel("Avg alerts")
    fig.tight_layout()
    _save(fig, "monthly_seasonality.png", output_dir)


def plot_weather_correlation(panel: pd.DataFrame, output_dir: Path):
    weather_cols = [c for c in ["wind_speed_max","precipitation","cloud_cover"] if c in panel.columns]
    if not weather_cols:
        print("  No weather columns – skipping")
        return

    labels = {"wind_speed_max":"Max wind (km/h)","precipitation":"Precipitation (mm)",
              "cloud_cover":"Cloud cover (%)"}
    fig, axes = plt.subplots(1, len(weather_cols), figsize=(5*len(weather_cols), 5))
    if len(weather_cols) == 1:
        axes = [axes]

    for ax, feat in zip(axes, weather_cols):
        q_col = f"{feat}_q"
        panel[q_col] = pd.qcut(panel[feat].fillna(0), q=4, labels=False, duplicates="drop")
        grp = panel.groupby(q_col)["alert_count"].agg(["mean","sem"]).reset_index()
        ax.bar(grp.index, grp["mean"], yerr=grp["sem"], color=P_MAIN, alpha=0.85, capsize=4)
        ax.set_xticks([0,1,2,3]); ax.set_xticklabels(["Q1\n(low)","Q2","Q3","Q4\n(high)"])
        ax.set_xlabel(f"{labels[feat]} quartile"); ax.set_ylabel("Mean daily alerts")
        ax.set_title(labels[feat])

        g1 = panel[panel[q_col]==0]["alert_count"].dropna()
        g4 = panel[panel[q_col]==3]["alert_count"].dropna()
        if len(g1) > 5 and len(g4) > 5:
            _, p = stats.mannwhitneyu(g1, g4, alternative="two-sided")
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            ax.text(1.5, grp["mean"].max()*0.95, f"Q1 vs Q4: {sig}", ha="center", fontsize=9, color="gray")

    fig.suptitle("Weather conditions vs alert frequency", fontsize=13)
    fig.tight_layout()
    _save(fig, "weather_correlation.png", output_dir)


def plot_autocorrelation(panel: pd.DataFrame, oblast: str, output_dir: Path):
    sub = panel[panel["oblast"] == oblast].sort_values("date")
    series = sub["alert_count"].values
    lags = range(1, 61)
    acf_vals = [pd.Series(series).autocorr(lag=l) for l in lags]
    ci = 1.96 / np.sqrt(len(series))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(list(lags), acf_vals, color=P_MAIN, alpha=0.85, width=0.85)
    ax.axhline(ci, ls="--", color="gray", lw=0.8, label="95% CI")
    ax.axhline(-ci, ls="--", color="gray", lw=0.8)
    ax.axhline(0, color="black", lw=0.5)
    for wk in [7,14,21,28]:
        ax.axvline(wk, color=P_WARM, alpha=0.25, lw=1.5)
    ax.set_xlabel("Lag (days)"); ax.set_ylabel("Autocorrelation")
    ax.set_title(f"Autocorrelation — {oblast}"); ax.legend()
    fig.tight_layout()
    _save(fig, f"autocorrelation_{oblast.split()[0]}.png", output_dir)


def plot_cooccurrence(panel: pd.DataFrame, output_dir: Path):
    binary = (
        panel.groupby(["date","oblast"])["alert_count"]
        .max().unstack(fill_value=0).clip(upper=1)
    )
    corr = binary.corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, mask=mask, cmap="coolwarm", center=0,
                vmin=-0.3, vmax=1, linewidths=0.3,
                cbar_kws={"label":"Pearson r","shrink":0.6})
    ax.set_title("Oblast co-occurrence — same-day targeting correlation", fontsize=12)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    _save(fig, "region_cooccurrence.png", output_dir)


def plot_rolling_trend(panel: pd.DataFrame, oblast: str, output_dir: Path):
    sub = panel[panel["oblast"]==oblast].sort_values("date").copy()
    sub["date_dt"] = pd.to_datetime(sub["date"])

    fig, axes = plt.subplots(3,1, figsize=(13,9), sharex=True)
    axes[0].plot(sub["date_dt"], sub["alert_count"], alpha=0.4, lw=0.8, color=P_MAIN, label="Daily")
    if "roll_7d_mean" in sub.columns:
        axes[0].plot(sub["date_dt"], sub["roll_7d_mean"], color=P_WARM, lw=1.5, label="7d mean")
    axes[0].set_ylabel("Alert count"); axes[0].set_title(f"{oblast} – time series"); axes[0].legend()

    if "roll_30d_mean" in sub.columns:
        axes[1].plot(sub["date_dt"], sub["roll_30d_mean"], color="#0F6E56", lw=1.5)
    axes[1].set_ylabel("30d rolling mean"); axes[1].set_title("Long-term trend")

    if "roll_7d_mean" in sub.columns:
        axes[2].plot(sub["date_dt"], sub["alert_count"]-sub["roll_7d_mean"].fillna(0),
                     alpha=0.55, color=P_MAIN, lw=0.7)
        axes[2].axhline(0, color="black", lw=0.5)
    axes[2].set_ylabel("Residual"); axes[2].set_title("De-trended residual (7d)")
    fig.tight_layout()
    _save(fig, f"rolling_trend_{oblast.split()[0]}.png", output_dir)


def plot_news_effect(panel: pd.DataFrame, output_dir: Path):
    if "news_severity" not in panel.columns or panel["news_severity"].nunique() <= 1:
        print("  No usable news features – skipping news effect plot")
        return
    agg = (
        panel.groupby("date").agg(
            national_alerts=("alert_count","sum"),
            news_severity=("news_severity","first"),
            nato_visit=("nato_visit_bin","first") if "nato_visit_bin" in panel.columns
                       else ("news_severity", lambda x: 0),
        ).reset_index().sort_values("date")
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    x = agg["news_severity"].fillna(0)
    y_next = agg["national_alerts"].shift(-1).fillna(0)
    ax.scatter(x, y_next, alpha=0.25, color=P_MAIN, s=18)
    m, b, r, p, _ = stats.linregress(x, y_next)
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, m*xs+b, color=P_WARM, lw=2, label=f"r={r:.2f}, p={p:.3f}")
    ax.set_xlabel("News severity (day T)"); ax.set_ylabel("National alerts (day T+1)")
    ax.set_title("News severity → next-day alerts"); ax.legend()

    ax = axes[1]
    if "nato_visit" in agg.columns:
        grps = [agg[agg["nato_visit"]==v]["national_alerts"].dropna() for v in [0,1]]
        ax.boxplot(grps, tick_labels=["No visit","NATO/EU visit"], patch_artist=True,
                   boxprops=dict(facecolor=P_LIGHT, color=P_MAIN),
                   medianprops=dict(color=P_WARM, lw=2))
        ax.set_ylabel("National alert count")
        ax.set_title("Alerts on days with vs without NATO/EU visits")
    fig.tight_layout()
    _save(fig, "news_effect.png", output_dir)


def plot_level_analysis(df_raw: pd.DataFrame, output_dir: Path):
    """
    Unique to this dataset: analyse what alert level (raion/hromada/oblast)
    is most common and whether it correlates with severity.
    """
    if "level" not in df_raw.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cnt = df_raw["level"].value_counts()
    axes[0].bar(cnt.index, cnt.values, color=P_MAIN, alpha=0.85)
    axes[0].set_title("Alert records by geographic level")
    axes[0].set_ylabel("Count")

    if "duration_min" in df_raw.columns:
        level_dur = df_raw.groupby("level")["duration_min"].median().sort_values()
        axes[1].barh(level_dur.index, level_dur.values, color=P_WARM, alpha=0.85)
        axes[1].set_xlabel("Median duration (minutes)")
        axes[1].set_title("Median alert duration by level")
    fig.tight_layout()
    _save(fig, "level_analysis.png", output_dir)


def run_all(df_raw_path, panel_path, output_dir: Path = Path("reports/figures")):
    df_raw = pd.read_csv(df_raw_path)
    df_raw.columns = df_raw.columns.str.lower().str.strip()
    df_raw["started_at"] = pd.to_datetime(df_raw["started_at"], utc=True, errors="coerce")
    if "duration_min" not in df_raw.columns and "finished_at" in df_raw.columns:
        df_raw["finished_at"] = pd.to_datetime(df_raw["finished_at"], utc=True, errors="coerce")
        df_raw["duration_min"] = (df_raw["finished_at"] - df_raw["started_at"]).dt.total_seconds().div(60).clip(0)

    panel = pd.read_parquet(panel_path)
    if "year" not in panel.columns:
        panel["year"]  = pd.to_datetime(panel["date"].astype(str)).dt.year
    if "month" not in panel.columns:
        panel["month"] = pd.to_datetime(panel["date"].astype(str)).dt.month

    print("  Hour distribution..."); plot_hour_distribution(df_raw, output_dir)
    print("  Day of week...");       plot_dow_pattern(df_raw, output_dir)
    print("  Seasonality...");       plot_monthly_seasonality(panel, output_dir)
    print("  Weather correlation..."); plot_weather_correlation(panel, output_dir)
    print("  Autocorrelation...");   plot_autocorrelation(panel, panel["oblast"].value_counts().idxmax(), output_dir)
    print("  Co-occurrence...");     plot_cooccurrence(panel, output_dir)
    print("  Rolling trend...");     plot_rolling_trend(panel, panel["oblast"].value_counts().idxmax(), output_dir)
    print("  News effect...");       plot_news_effect(panel, output_dir)
    print("  Level analysis...");    plot_level_analysis(df_raw, output_dir)
    print(f"  All pattern plots saved → {output_dir}")
