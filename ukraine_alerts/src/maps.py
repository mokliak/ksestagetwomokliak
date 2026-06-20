"""
src/maps.py
===========
Module 2: Live alert map + historical risk heatmap.
Uses Folium (interactive HTML) and matplotlib (static PNG).
"""

from __future__ import annotations
import os, requests, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    import folium
    from folium.plugins import HeatMap
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

OBLAST_COORDS = {
    "Mykolaivska oblast":        (46.97, 31.99),
    "Dnipropetrovska oblast":    (48.46, 35.04),
    "Kharkivska oblast":         (49.99, 36.23),
    "Kyivska oblast":            (50.45, 30.52),
    "Odeska oblast":             (46.47, 30.73),
    "Zaporizka oblast":          (47.84, 35.14),
    "Lvivska oblast":            (49.84, 24.03),
    "Donetska oblast":           (48.00, 37.80),
    "Khersonska oblast":         (46.64, 32.62),
    "Poltavska oblast":          (49.59, 34.55),
    "Sumska oblast":             (50.91, 34.80),
    "Chernihivska oblast":       (51.49, 31.29),
    "Zhytomyrska oblast":        (50.25, 28.66),
    "Vinnytska oblast":          (49.23, 28.47),
    "Cherkaska oblast":          (49.44, 32.06),
    "Khmelnytska oblast":        (49.42, 26.99),
    "Rivnenska oblast":          (50.62, 26.25),
    "Volynska oblast":           (50.74, 25.32),
    "Ivano-Frankivska oblast":   (48.92, 24.71),
    "Ternopilska oblast":        (49.55, 25.60),
    "Chernivetska oblast":       (48.30, 25.94),
    "Zakarpatska oblast":        (48.62, 22.29),
    "Kirovohradska oblast":      (48.51, 32.27),
    "Luhanska oblast":           (48.57, 39.31),
    "Avtonomna Respublika Krym": (44.95, 34.12),
    "Kyiv":                      (50.45, 30.52),
}

# ukrainealarm.com → normalised name map
UA_TO_OBLAST = {v: v for v in OBLAST_COORDS}
UA_TO_OBLAST.update({
    "Миколаївська": "Mykolaivska oblast",
    "Дніпропетровська": "Dnipropetrovska oblast",
    "Харківська": "Kharkivska oblast",
    "Київська": "Kyivska oblast",
    "Одеська": "Odeska oblast",
    "Запорізька": "Zaporizka oblast",
    "Львівська": "Lvivska oblast",
    "Донецька": "Donetska oblast",
    "Херсонська": "Khersonska oblast",
    "Полтавська": "Poltavska oblast",
    "Сумська": "Sumska oblast",
    "Чернігівська": "Chernihivska oblast",
    "Житомирська": "Zhytomyrska oblast",
    "Вінницька": "Vinnytska oblast",
    "Черкаська": "Cherkaska oblast",
    "Хмельницька": "Khmelnytska oblast",
    "Рівненська": "Rivnenska oblast",
    "Волинська": "Volynska oblast",
    "Івано-Франківська": "Ivano-Frankivska oblast",
    "Тернопільська": "Ternopilska oblast",
    "Чернівецька": "Chernivetska oblast",
    "Закарпатська": "Zakarpatska oblast",
    "Кіровоградська": "Kirovohradska oblast",
    "Луганська": "Luhanska oblast",
    "Крим": "Avtonomna Respublika Krym",
})

ALARM_API = "https://api.ukrainealarm.com/api/v3/alerts"


def fetch_live_alerts(api_key: str = "") -> dict[str, bool]:
    """Returns {oblast_name: is_active} dict."""
    headers = {"Authorization": api_key} if api_key else {}
    try:
        r = requests.get(ALARM_API, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Live alert API unavailable ({e}); using empty state")
        data = []

    active = {}
    for entry in data:
        ua_name = entry.get("regionName","")
        en_name = UA_TO_OBLAST.get(ua_name, ua_name)
        has_air = any(
            a.get("type") in ("AIR","MISSILE","DRONE")
            for a in entry.get("activeAlerts",[])
        )
        active[en_name] = has_air
    return active


def live_alert_map(
    api_key: str = "",
    output_path: Path = Path("reports/live_alerts.html"),
):
    assert HAS_FOLIUM, "pip install folium"
    active = fetch_live_alerts(api_key)
    m = folium.Map(location=[49.0,31.5], zoom_start=6, tiles="CartoDB positron")

    title = (
        f'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
        f'z-index:1000;background:white;padding:8px 18px;border-radius:8px;'
        f'box-shadow:0 2px 8px rgba(0,0,0,.2);font-family:sans-serif;font-size:14px;">'
        f'<b>Ukraine Air Raid Alert Status</b> &nbsp;|&nbsp;'
        f'<span style="color:#e74c3c">● Active</span> &nbsp;'
        f'<span style="color:#27ae60">● Clear</span> &nbsp;|&nbsp;'
        f'Updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</div>'
    )
    m.get_root().html.add_child(folium.Element(title))

    for oblast, (lat, lon) in OBLAST_COORDS.items():
        is_active = active.get(oblast, False)
        color = "#e74c3c" if is_active else "#27ae60"
        folium.CircleMarker(
            location=[lat, lon], radius=14 if is_active else 8,
            color=color, fill=True, fill_color=color,
            fill_opacity=0.75 if is_active else 0.35,
            popup=folium.Popup(f"<b>{oblast}</b><br>{'ACTIVE' if is_active else 'Clear'}",
                               max_width=180),
            tooltip=f"{oblast}: {'ACTIVE ⚠' if is_active else 'Clear'}",
        ).add_to(m)
        if is_active:
            folium.CircleMarker(
                location=[lat, lon], radius=24, color="#e74c3c",
                fill=False, weight=2, opacity=0.3,
            ).add_to(m)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"  Live map → {output_path}")
    return m


def risk_heatmap_folium(
    panel: pd.DataFrame,
    metric: str = "alert_rate",
    output_path: Path = Path("reports/risk_heatmap.html"),
):
    assert HAS_FOLIUM, "pip install folium"
    stats = (
        panel.groupby("oblast").agg(
            alert_rate         = ("had_alert", "mean"),
            avg_duration       = ("avg_dur_min", "mean"),
            alert_count_total  = ("alert_count", "sum"),
            unique_raions_mean = ("unique_raions_hit", "mean"),
        ).reset_index()
    )
    max_val = stats[metric].max() or 1
    cmap = plt.cm.YlOrRd

    m = folium.Map(location=[49.0,31.5], zoom_start=6, tiles="CartoDB positron")
    for _, row in stats.iterrows():
        coords = OBLAST_COORDS.get(row["oblast"])
        if not coords:
            continue
        val = row[metric]
        hex_c = mcolors.to_hex(cmap(val / max_val))
        folium.CircleMarker(
            location=list(coords),
            radius=max(6, (val / max_val) * 30),
            color=hex_c, fill=True, fill_color=hex_c, fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>{row['oblast']}</b><br>"
                f"Alert rate: {row['alert_rate']:.1%}<br>"
                f"Total alerts: {int(row['alert_count_total'])}<br>"
                f"Avg duration: {row['avg_duration']:.0f} min",
                max_width=220,
            ),
            tooltip=f"{row['oblast']}: {val:.2f}",
        ).add_to(m)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"  Risk heatmap → {output_path}")
    return m


def risk_heatmap_static(
    panel: pd.DataFrame,
    output_path: Path = Path("reports/figures/risk_heatmap_static.png"),
):
    stats = (
        panel.groupby("oblast")
        .agg(alert_rate=("had_alert","mean"))
        .reset_index()
        .sort_values("alert_rate", ascending=True)
    )
    norm = plt.Normalize(0, stats["alert_rate"].max())
    colors = plt.cm.YlOrRd(norm(stats["alert_rate"]))

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(stats["oblast"], stats["alert_rate"], color=colors, edgecolor="white", lw=0.4)
    ax.set_xlabel("Fraction of days with ≥1 alert")
    ax.set_title("Ukraine – regional air raid risk", fontsize=13)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.4)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Static heatmap → {output_path}")
