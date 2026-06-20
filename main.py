#!/usr/bin/env python3
"""
setup_project.py
================
Run this ONCE to scaffold the entire Ukraine Air Raid Alert analysis project.

Usage:
    python setup_project.py                        # scaffold into ./ukraine_alerts/
    python setup_project.py --dir my_folder        # custom target directory
    python setup_project.py --dir . --no-venv      # scaffold in-place, skip venv

What it does:
  1. Creates the full directory tree
  2. Writes every source file (features, models, maps, patterns, dashboard, main)
  3. Writes README, requirements.txt, .env.example, .gitignore
  4. Optionally creates and activates a virtual environment
  5. Prints the exact commands to run next
"""

import argparse
import os
import sys
import subprocess
import textwrap
from pathlib import Path

# ─── colour helpers ────────────────────────────────────────────────────────────
def g(s): return f"\033[92m{s}\033[0m"   # green
def b(s): return f"\033[94m{s}\033[0m"   # blue
def y(s): return f"\033[93m{s}\033[0m"   # yellow
def r(s): return f"\033[91m{s}\033[0m"   # red
def h(s): return f"\033[1m{s}\033[0m"    # bold


def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"))
    print(f"  {g('✓')} {path.relative_to(path.parents[len(path.parts)-2])}")


# ══════════════════════════════════════════════════════════════════════════════
# FILE CONTENTS
# ══════════════════════════════════════════════════════════════════════════════

def write_requirements(root: Path):
    write(root / "requirements.txt", """
        # Core data
        pandas>=2.1
        numpy>=1.26
        pyarrow>=14.0
        scikit-learn>=1.4
        scipy>=1.11

        # Visualisation
        matplotlib>=3.8
        seaborn>=0.13
        plotly>=5.18
        folium>=0.15
        geopandas>=0.14

        # Dashboard
        streamlit>=1.32
        streamlit-folium>=0.18

        # HTTP / APIs
        requests>=2.31
        python-dotenv>=1.0
        newsapi-python>=0.2.7

        # Optional speed-up
        fastparquet>=2023.0

        # Notebooks
        jupyter>=1.0
        ipykernel>=6.28
    """)


def write_env_example(root: Path):
    write(root / ".env.example", """
        # Copy to .env and fill in your values

        # Anthropic – used for news severity scoring (Haiku model, very cheap)
        ANTHROPIC_API_KEY=sk-ant-...

        # NewsAPI – free tier: 100 requests/day
        # Register: https://newsapi.org/register
        NEWSAPI_KEY=your_newsapi_key_here

        # Ukraine Alarm API – free registration at https://ukrainealarm.com
        # Live map falls back to mock data if not set
        UKRAINE_ALARM_API_KEY=your_ukraine_alarm_key_here

        # Open-Meteo – no key needed (free, rate-limited)
    """)


def write_gitignore(root: Path):
    write(root / ".gitignore", """
        # Environment
        .env
        *.env

        # Python
        __pycache__/
        *.py[cod]
        *.so
        .Python
        build/
        dist/
        *.egg-info/
        .eggs/

        # Jupyter
        .ipynb_checkpoints/
        *.ipynb

        # Virtual envs
        venv/
        .venv/
        env/

        # Data – keep schema, not raw data
        data/alerts.csv
        data/*.parquet
        data/*.pkl

        # Generated outputs
        reports/figures/*.png
        reports/*.html
        data/models.pkl

        # OS / IDE
        .DS_Store
        Thumbs.db
        .vscode/
        .idea/
        *.swp
    """)


def write_readme(root: Path):
    write(root / "README.md", r"""
        # 🇺🇦 Ukraine Air Raid Alert — Time Series Analysis

        > KSE Stage 2 | Forecasting · Mapping · Pattern Mining · Interactive Dashboard

        ---

        ## Quickstart

        ```bash
        # 1. Install dependencies
        pip install -r requirements.txt

        # 2. Configure API keys (optional — degrades gracefully without them)
        cp .env.example .env
        # Edit .env with your keys

        # 3. Drop your CSV at data/alerts.csv, then run the full pipeline:
        python main.py data/alerts.csv

        # 4. Launch dashboard
        streamlit run dashboard/app.py
        ```

        ---

        ## CSV Format Expected

        | Column | Description |
        |--------|-------------|
        | `oblast` | Oblast name (e.g. `Mykolaivska oblast`) |
        | `raion` | Raion name |
        | `hromada` | Hromada name (may be NaN) |
        | `level` | `oblast` / `raion` / `hromada` |
        | `started_at` | Alert start (ISO 8601, UTC offset) |
        | `finished_at` | Alert end (ISO 8601, UTC offset) |
        | `source` | `official` or other |

        ---

        ## Project Structure

        ```
        .
        ├── data/
        │   ├── alerts.csv              ← your raw data
        │   ├── features.parquet        ← auto-generated
        │   ├── weather_cache.parquet   ← auto-cached
        │   └── news_cache.parquet      ← auto-cached
        ├── src/
        │   ├── features.py             ← feature engineering
        │   ├── models.py               ← Logistic + Random Forest
        │   ├── maps.py                 ← Folium live + heat map
        │   └── patterns.py             ← all pattern charts
        ├── dashboard/
        │   └── app.py                  ← 4-page Streamlit dashboard
        ├── reports/
        │   ├── figures/                ← PNG charts
        │   ├── live_alerts.html
        │   └── risk_heatmap.html
        ├── notebooks/                  ← EDA playground
        ├── main.py                     ← end-to-end runner
        ├── requirements.txt
        └── .env.example
        ```

        ---

        ## Features

        ### Temporal
        - Hour (cyclical sin/cos), day-of-week, month, is_weekend, is_night
        - Duration in minutes per alert

        ### Lag / rolling (per oblast)
        - Lag counts: 1d, 2d, 3d, 7d, 14d
        - Rolling mean: 3d, 7d, 14d, 30d
        - Rolling std: 7d, 14d

        ### Geographic hierarchy
        - Oblast-level, raion-level, and combined daily counts
        - Granularity ratio: raion alerts / oblast alerts (fragmentation index)

        ### Weather (Open-Meteo, free)
        - Max wind speed, precipitation, cloud cover
        - `high_wind` binary (≥40 km/h — drone suppressor hypothesis)

        ### News (NewsAPI + Claude Haiku)
        - `news_severity` [0–1]: strike severity from headline classification
        - `nato_visit_flag` [0/1]: NATO/EU official visit detected

        ### Geopolitical
        - `days_since_massive_strike`: buildup pressure proxy
        - `level_raion_pct`: fraction of alerts at raion vs oblast level
    """)


def write_features(root: Path):
    write(root / "src" / "features.py", '''
        """
        src/features.py
        ===============
        Feature engineering pipeline for Ukraine air raid alerts.

        Input CSV columns (from real data):
            oblast, raion, hromada, level, started_at, finished_at, source

        Outputs:
            data/features.parquet  – daily panel: one row per (date × oblast)
        """

        from __future__ import annotations

        import os
        import time
        import json
        import warnings
        import requests
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from dotenv import load_dotenv

        load_dotenv()
        warnings.filterwarnings("ignore")

        # ── Oblast centroids (lat, lon) ─────────────────────────────────────────────
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

        WIND_THRESHOLD_KMH = 40
        OPEN_METEO_URL     = "https://archive-api.open-meteo.com/v1/archive"
        NEWSAPI_KEY        = os.getenv("NEWSAPI_KEY", "")
        ANTHROPIC_KEY      = os.getenv("ANTHROPIC_API_KEY", "")


        # ════════════════════════════════════════════════════════════════════════════
        # 1. LOAD & CLEAN RAW CSV
        # ════════════════════════════════════════════════════════════════════════════

        def load_raw(path: str | Path) -> pd.DataFrame:
            """
            Load the real alert CSV.
            Expected columns: oblast, raion, hromada, level, started_at, finished_at, source
            """
            df = pd.read_csv(path, index_col=0)  # first col is unnamed index

            # Normalise column names
            df.columns = df.columns.str.strip().str.lower()

            # Parse datetimes (already have tz offset)
            df["started_at"]  = pd.to_datetime(df["started_at"],  utc=True, errors="coerce")
            df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True, errors="coerce")

            # Drop rows without a valid start
            df = df.dropna(subset=["started_at", "oblast"])
            df = df.sort_values("started_at").reset_index(drop=True)

            # Clean oblast names (strip whitespace)
            df["oblast"] = df["oblast"].str.strip()
            df["raion"]  = df["raion"].str.strip()
            df["level"]  = df["level"].str.strip().str.lower()

            # Duration in minutes
            df["duration_min"] = (
                (df["finished_at"] - df["started_at"])
                .dt.total_seconds()
                .div(60)
                .clip(lower=0)
            )

            print(f"  Loaded {len(df):,} alert records across {df['oblast'].nunique()} oblasts")
            print(f"  Date range: {df['started_at'].min().date()} → {df['started_at'].max().date()}")
            print(f"  Levels: {df['level'].value_counts().to_dict()}")
            return df


        # ════════════════════════════════════════════════════════════════════════════
        # 2. TIME FEATURES (on alert-level df)
        # ════════════════════════════════════════════════════════════════════════════

        def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            ts = df["started_at"].dt.tz_convert("Europe/Kyiv")

            df["date"]        = ts.dt.date
            df["hour"]        = ts.dt.hour
            df["day_of_week"] = ts.dt.dayofweek       # 0=Mon
            df["month"]       = ts.dt.month
            df["year"]        = ts.dt.year
            df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
            df["is_night"]    = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)

            # Cyclical
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
            return df


        # ════════════════════════════════════════════════════════════════════════════
        # 3. DAILY PANEL — one row per (date × oblast)
        # ════════════════════════════════════════════════════════════════════════════

        def build_daily_panel(df: pd.DataFrame) -> pd.DataFrame:
            """
            Aggregate to daily (date, oblast) level.
            Also captures raion-level fragmentation and alert level mix.
            """
            # ── overall counts per (date, oblast) ──
            base = (
                df.groupby(["date", "oblast"])
                .agg(
                    alert_count   = ("started_at", "count"),
                    total_min     = ("duration_min", "sum"),
                    avg_dur_min   = ("duration_min", "mean"),
                    first_hour    = ("hour", "min"),
                    night_alerts  = ("is_night", "sum"),
                )
                .reset_index()
            )

            # ── level breakdown (raion / hromada / oblast alerts) ──
            level_counts = (
                df.groupby(["date", "oblast", "level"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )
            for lvl in ("raion", "hromada", "oblast"):
                if lvl not in level_counts.columns:
                    level_counts[lvl] = 0
            level_counts = level_counts.rename(columns={
                "raion":   "raion_alerts",
                "hromada": "hromada_alerts",
                "oblast":  "oblast_alerts",
            })
            level_cols = [c for c in ["raion_alerts","hromada_alerts","oblast_alerts"]
                          if c in level_counts.columns]
            base = base.merge(level_counts[["date","oblast"] + level_cols],
                              on=["date","oblast"], how="left")

            # ── unique raions hit per day ──
            raion_count = (
                df.groupby(["date","oblast"])["raion"]
                .nunique()
                .reset_index()
                .rename(columns={"raion": "unique_raions_hit"})
            )
            base = base.merge(raion_count, on=["date","oblast"], how="left")

            # ── raion fragmentation index ──
            base["raion_frag"] = (
                base.get("raion_alerts", 0) /
                base["alert_count"].replace(0, np.nan)
            ).fillna(0)

            # ── Complete the grid: every (date × oblast) pair ──
            all_dates   = pd.date_range(
                str(base["date"].min()), str(base["date"].max()), freq="D"
            ).date
            all_oblasts = base["oblast"].unique()
            full_idx    = pd.MultiIndex.from_product(
                [all_dates, all_oblasts], names=["date", "oblast"]
            )
            base = (
                base.set_index(["date","oblast"])
                .reindex(full_idx, fill_value=0)
                .reset_index()
            )
            base["alert_count"]    = base["alert_count"].astype(int)
            base["had_alert"]      = (base["alert_count"] > 0).astype(int)
            return base


        # ════════════════════════════════════════════════════════════════════════════
        # 4. LAG & ROLLING FEATURES
        # ════════════════════════════════════════════════════════════════════════════

        def add_lag_features(
            panel: pd.DataFrame,
            lags: tuple = (1, 2, 3, 7, 14),
            windows: tuple = (3, 7, 14, 30),
        ) -> pd.DataFrame:
            panel = panel.copy().sort_values(["oblast", "date"])

            for lag in lags:
                panel[f"lag_{lag}d_count"] = (
                    panel.groupby("oblast")["alert_count"].shift(lag)
                )
                panel[f"lag_{lag}d_had"] = (
                    panel.groupby("oblast")["had_alert"].shift(lag)
                )

            for w in windows:
                panel[f"roll_{w}d_mean"] = (
                    panel.groupby("oblast")["alert_count"]
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                )
                panel[f"roll_{w}d_std"] = (
                    panel.groupby("oblast")["alert_count"]
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0))
                )
                panel[f"roll_{w}d_max"] = (
                    panel.groupby("oblast")["alert_count"]
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
                )

            # National daily total (all oblasts combined) – lagged
            national = (
                panel.groupby("date")["alert_count"]
                .sum()
                .reset_index()
                .rename(columns={"alert_count": "national_total"})
                .sort_values("date")
            )
            national["nat_lag1d"] = national["national_total"].shift(1)
            national["nat_roll7d"] = national["national_total"].shift(1).rolling(7, min_periods=1).mean()
            panel = panel.merge(national[["date","national_total","nat_lag1d","nat_roll7d"]],
                                on="date", how="left")
            return panel


        # ════════════════════════════════════════════════════════════════════════════
        # 5. WEATHER FEATURES (Open-Meteo, free)
        # ════════════════════════════════════════════════════════════════════════════

        def _fetch_weather(oblast: str, start: str, end: str) -> pd.DataFrame:
            coords = OBLAST_COORDS.get(oblast)
            if not coords:
                return pd.DataFrame()
            lat, lon = coords
            try:
                r = requests.get(OPEN_METEO_URL, params={
                    "latitude": lat, "longitude": lon,
                    "start_date": start, "end_date": end,
                    "daily": "wind_speed_10m_max,precipitation_sum,cloudcover_mean,temperature_2m_mean",
                    "timezone": "Europe/Kyiv",
                    "wind_speed_unit": "kmh",
                }, timeout=20)
                r.raise_for_status()
                d = r.json()["daily"]
                df = pd.DataFrame(d)
                df["date"]   = pd.to_datetime(df["time"]).dt.date
                df["oblast"] = oblast
                df = df.rename(columns={
                    "wind_speed_10m_max":    "wind_speed_max",
                    "precipitation_sum":     "precipitation",
                    "cloudcover_mean":       "cloud_cover",
                    "temperature_2m_mean":   "temperature",
                }).drop(columns=["time"])
                df["high_wind"] = (df["wind_speed_max"] >= WIND_THRESHOLD_KMH).astype(int)
                return df
            except Exception as e:
                print(f"    Weather fetch failed ({oblast}): {e}")
                return pd.DataFrame()


        def add_weather_features(
            panel: pd.DataFrame,
            cache_path: Path | None = None,
        ) -> pd.DataFrame:
            if cache_path and cache_path.exists():
                weather = pd.read_parquet(cache_path)
                print(f"  Weather loaded from cache ({len(weather):,} rows)")
            else:
                start = str(panel["date"].min())
                end   = str(panel["date"].max())
                frames = []
                oblasts = panel["oblast"].unique()
                for i, ob in enumerate(oblasts):
                    print(f"  Fetching weather {i+1}/{len(oblasts)}: {ob}", end="\\r")
                    wdf = _fetch_weather(ob, start, end)
                    if not wdf.empty:
                        frames.append(wdf)
                    time.sleep(0.35)
                print()
                if not frames:
                    return panel
                weather = pd.concat(frames, ignore_index=True)
                if cache_path:
                    weather.to_parquet(cache_path, index=False)
                    print(f"  Weather cached → {cache_path}")

            panel = panel.merge(
                weather[["date","oblast","wind_speed_max","precipitation",
                          "cloud_cover","temperature","high_wind"]],
                on=["date","oblast"], how="left",
            )
            return panel


        # ════════════════════════════════════════════════════════════════════════════
        # 6. NEWS FEATURES (NewsAPI + Claude Haiku scorer)
        # ════════════════════════════════════════════════════════════════════════════

        NEWS_QUERIES = {
            "strike_severity": "Ukraine missile drone attack airstrike explosion",
            "nato_visit":      "NATO EU official minister visiting Ukraine Kyiv delegation",
            "peace_talks":     "Ukraine ceasefire peace negotiations talks diplomat",
        }

        def _fetch_headlines(date_str: str, query: str, page_size: int = 10) -> list[str]:
            if not NEWSAPI_KEY:
                return []
            try:
                r = requests.get("https://newsapi.org/v2/everything", params={
                    "q": query,
                    "from": date_str,
                    "to":   date_str,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": page_size,
                    "apiKey": NEWSAPI_KEY,
                }, timeout=12)
                return [a["title"] for a in r.json().get("articles", []) if a.get("title")]
            except Exception as e:
                print(f"    NewsAPI error ({date_str}): {e}")
                return []


        def _score_with_claude(date_str: str, headlines: list[str], query_type: str) -> float:
            """
            Send headlines to Claude Haiku for 0-1 scoring.
            query_type controls the prompt framing.
            """
            if not ANTHROPIC_KEY or not headlines:
                return 0.0

            headline_block = " | ".join(headlines[:10])

            prompts = {
                "strike_severity": f"""Headlines for {date_str}: {headline_block}

Rate the severity of Russian air/missile/drone strikes on Ukraine this day.
Scale: 0.0 = no attack mentioned, 1.0 = massive nationwide strike, many regions, ballistic missiles.
Consider: number of regions, weapon type (ballistic > cruise > drone), casualties.
Reply ONLY with a single decimal number between 0.0 and 1.0.""",

                "nato_visit": f"""Headlines for {date_str}: {headline_block}

Did a senior NATO, EU, or G7 official (minister or above) visit Ukraine or Kyiv today?
0.0 = no visit, 1.0 = confirmed high-level visit.
Reply ONLY with 0.0 or 1.0.""",

                "peace_talks": f"""Headlines for {date_str}: {headline_block}

Is there significant peace negotiation or ceasefire activity involving Ukraine today?
0.0 = no, 1.0 = active high-level talks or ceasefire announced.
Reply ONLY with a decimal 0.0 to 1.0.""",
            }

            try:
                r = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-haiku-4-5-20251001",
                        "max_tokens": 10,
                        "messages": [{"role": "user", "content": prompts[query_type]}],
                    },
                    timeout=20,
                )
                text = r.json()["content"][0]["text"].strip().split()[0]
                return max(0.0, min(1.0, float(text)))
            except Exception as e:
                print(f"    Claude error ({date_str}/{query_type}): {e}")
                return 0.0


        def add_news_features(
            panel: pd.DataFrame,
            cache_path: Path | None = None,
        ) -> pd.DataFrame:
            if cache_path and cache_path.exists():
                news_df = pd.read_parquet(cache_path)
                print(f"  News loaded from cache ({len(news_df)} dates)")
            else:
                if not NEWSAPI_KEY:
                    print("  ⚠ NEWSAPI_KEY not set – skipping news features")
                    for col in ["news_severity","nato_visit_flag","peace_talks_flag"]:
                        panel[col] = 0.0
                    return panel

                unique_dates = sorted(panel["date"].unique())
                rows = []
                for i, d in enumerate(unique_dates):
                    print(f"  News scoring {i+1}/{len(unique_dates)}: {d}", end="\\r")
                    date_str = str(d)
                    row = {"date": d}

                    for qt, query in NEWS_QUERIES.items():
                        headlines = _fetch_headlines(date_str, query)
                        row[qt]   = _score_with_claude(date_str, headlines, qt)
                        time.sleep(1.2)   # avoid rate limits

                    rows.append(row)

                print()
                news_df = pd.DataFrame(rows)
                if cache_path:
                    news_df.to_parquet(cache_path, index=False)
                    print(f"  News cached → {cache_path}")

            # Rename to final column names
            news_df = news_df.rename(columns={
                "strike_severity": "news_severity",
                "nato_visit":      "nato_visit_flag",
                "peace_talks":     "peace_talks_flag",
            })

            panel = panel.merge(news_df, on="date", how="left")
            for col in ["news_severity","nato_visit_flag","peace_talks_flag"]:
                panel[col] = panel.get(col, pd.Series(0.0)).fillna(0.0)

            # Binary flags from continuous scores
            panel["nato_visit_bin"]    = (panel["nato_visit_flag"]  > 0.5).astype(int)
            panel["peace_talks_bin"]   = (panel["peace_talks_flag"] > 0.5).astype(int)
            return panel


        # ════════════════════════════════════════════════════════════════════════════
        # 7. GEOPOLITICAL / DERIVED FEATURES
        # ════════════════════════════════════════════════════════════════════════════

        def add_geopolitical_features(
            panel: pd.DataFrame,
            massive_threshold: int = 5,  # oblasts active on same day = massive
        ) -> pd.DataFrame:
            panel = panel.copy().sort_values("date")

            # National daily total already computed; find "massive" days
            nat = (
                panel.groupby("date")["had_alert"]
                .sum()
                .reset_index()
                .rename(columns={"had_alert": "oblasts_active"})
            )
            nat["is_massive"] = (nat["oblasts_active"] >= massive_threshold).astype(int)
            massive_dates = nat[nat["is_massive"] == 1]["date"].tolist()

            def days_since(d):
                past = [m for m in massive_dates if m < d]
                return (d - max(past)).days if past else np.nan

            nat["days_since_massive"] = nat["date"].apply(days_since)
            nat["oblasts_active_lag1"] = nat["oblasts_active"].shift(1)

            panel = panel.merge(
                nat[["date","oblasts_active","is_massive","days_since_massive","oblasts_active_lag1"]],
                on="date", how="left",
            )
            return panel


        # ════════════════════════════════════════════════════════════════════════════
        # 8. DUMMY / CATEGORICAL ENCODING
        # ════════════════════════════════════════════════════════════════════════════

        def add_dummy_variables(panel: pd.DataFrame) -> pd.DataFrame:
            """
            Create statistically-motivated dummy variables:
            - Time-of-day buckets (from hour-of-day analysis)
            - Season (winter = higher missile activity historically)
            - East/West geographic risk zone
            - Source reliability (if source column present)
            """
            panel = panel.copy()

            # ── Temporal dummies ──────────────────────────────────────────────
            # Based on empirical patterns: early-morning window most dangerous
            panel["time_bucket"] = pd.cut(
                panel.get("first_hour", pd.Series(12, index=panel.index)),
                bins=[-1, 5, 11, 17, 23],
                labels=["night_0_5","morning_6_11","afternoon_12_17","evening_18_23"],
            )

            # Season dummy (December–February = winter)
            month = pd.to_datetime(panel["date"].astype(str)).dt.month
            panel["season"] = pd.cut(
                month,
                bins=[0, 2, 5, 8, 11, 12],
                labels=["winter","spring","summer","autumn","winter2"],
            ).astype(str).str.replace("winter2","winter")

            # One-hot encode season (drop first to avoid multicollinearity)
            season_dummies = pd.get_dummies(panel["season"], prefix="season", drop_first=True)
            panel = pd.concat([panel, season_dummies], axis=1)

            # ── Geographic zone dummies ───────────────────────────────────────
            # Front-line / Eastern oblasts face highest persistent risk
            EAST_OBLASTS = {
                "Donetska oblast","Luhanska oblast","Zaporizka oblast",
                "Kharkivska oblast","Dnipropetrovska oblast","Khersonska oblast",
            }
            CENTRAL_OBLASTS = {
                "Mykolaivska oblast","Poltavska oblast","Cherkaska oblast",
                "Kirovohradska oblast","Vinnytska oblast","Zhytomyrska oblast",
                "Kyivska oblast","Chernihivska oblast","Sumska oblast",
            }
            panel["zone_east"]    = panel["oblast"].isin(EAST_OBLASTS).astype(int)
            panel["zone_central"] = panel["oblast"].isin(CENTRAL_OBLASTS).astype(int)
            panel["zone_west"]    = (~panel["oblast"].isin(EAST_OBLASTS | CENTRAL_OBLASTS)).astype(int)

            # ── Alert level mix dummies ───────────────────────────────────────
            # Was yesterday dominated by raion-level (localised) or oblast-wide alerts?
            if "raion_frag" in panel.columns:
                panel["mostly_raion"]  = (panel["raion_frag"]  > 0.7).astype(int)
                panel["mostly_oblast"] = (panel["raion_frag"]  < 0.3).astype(int)

            # ── High-wind suppressor ──────────────────────────────────────────
            if "high_wind" not in panel.columns:
                panel["high_wind"] = 0

            return panel


        # ════════════════════════════════════════════════════════════════════════════
        # 9. MASTER PIPELINE
        # ════════════════════════════════════════════════════════════════════════════

        def build_feature_set(
            raw_csv: str | Path,
            data_dir: Path = Path("data"),
        ) -> pd.DataFrame:
            data_dir = Path(data_dir)
            data_dir.mkdir(exist_ok=True)

            print("\\n[1/7] Loading raw CSV...")
            df = load_raw(raw_csv)
            df = add_time_features(df)

            print("[2/7] Building daily panel...")
            panel = build_daily_panel(df)

            print("[3/7] Adding lag / rolling features...")
            panel = add_lag_features(panel)

            print("[4/7] Fetching weather features (Open-Meteo)...")
            panel = add_weather_features(panel, cache_path=data_dir / "weather_cache.parquet")

            print("[5/7] Scoring news features (NewsAPI + Claude)...")
            panel = add_news_features(panel, cache_path=data_dir / "news_cache.parquet")

            print("[6/7] Computing geopolitical features...")
            panel = add_geopolitical_features(panel)

            print("[7/7] Encoding dummy variables...")
            panel = add_dummy_variables(panel)

            out = data_dir / "features.parquet"
            panel.to_parquet(out, index=False)
            print(f"\\n✓ Feature set saved → {out}")
            print(f"  Shape: {panel.shape}  ({panel.columns.tolist()[:8]}... +{len(panel.columns)-8} more)")
            return panel


        if __name__ == "__main__":
            import sys
            csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/alerts.csv"
            build_feature_set(csv_path)
    ''')


def write_models(root: Path):
    write(root / "src" / "models.py", '''
        """
        src/models.py
        =============
        Two models × two horizons + monthly regressor.
        All validated with walk-forward TimeSeriesSplit.
        """

        from __future__ import annotations

        import pickle, warnings
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        from pathlib import Path
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score, mean_absolute_error

        warnings.filterwarnings("ignore")

        # Feature columns (all optional — missing ones are dropped automatically)
        FEATURE_COLS = [
            # Lag counts
            "lag_1d_count","lag_2d_count","lag_3d_count","lag_7d_count","lag_14d_count",
            "lag_1d_had","lag_7d_had",
            # Rolling stats
            "roll_3d_mean","roll_7d_mean","roll_14d_mean","roll_30d_mean",
            "roll_7d_std","roll_14d_std","roll_7d_max",
            # National context
            "nat_lag1d","nat_roll7d","oblasts_active_lag1","days_since_massive",
            # Time cyclical
            "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
            "is_weekend",
            # Weather
            "wind_speed_max","precipitation","cloud_cover","temperature","high_wind",
            # News
            "news_severity","nato_visit_bin","peace_talks_bin",
            # Geography
            "zone_east","zone_central","zone_west",
            # Alert structure
            "raion_frag","unique_raions_hit",
            # Season dummies (auto-generated, check existence)
            "season_spring","season_summer","season_autumn",
        ]


        def _available(df: pd.DataFrame) -> list[str]:
            return [c for c in FEATURE_COLS if c in df.columns]


        def _make_binary_target(panel: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
            """1 if any alert occurs in this oblast within the next horizon_days days."""
            panel = panel.copy().sort_values(["oblast","date"])
            panel[f"y_{horizon_days}d"] = (
                panel.groupby("oblast")["alert_count"]
                .transform(lambda x: x.shift(-horizon_days).rolling(horizon_days, min_periods=1).max())
                .clip(upper=1).astype(int)
            )
            return panel


        def train_classifier(
            panel: pd.DataFrame,
            horizon_days: int = 1,
            n_splits: int = 5,
            output_dir: Path = Path("reports"),
        ) -> dict:
            output_dir = Path(output_dir)
            (output_dir / "figures").mkdir(parents=True, exist_ok=True)

            panel = _make_binary_target(panel, horizon_days)
            target = f"y_{horizon_days}d"
            features = _available(panel)
            sub = panel.dropna(subset=features + [target]).copy()
            X, y = sub[features].values, sub[target].values

            tscv = TimeSeriesSplit(n_splits=n_splits)

            model_defs = {
                "logistic": Pipeline([
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(
                        C=0.1, max_iter=500, class_weight="balanced", solver="lbfgs"
                    )),
                ]),
                "random_forest": Pipeline([
                    ("sc", StandardScaler()),
                    ("clf", RandomForestClassifier(
                        n_estimators=300, max_depth=8, min_samples_leaf=10,
                        class_weight="balanced", random_state=42, n_jobs=-1,
                    )),
                ]),
            }

            results = {}
            for name, pipe in model_defs.items():
                fold_aucs = []
                for tr, va in tscv.split(X):
                    pipe.fit(X[tr], y[tr])
                    fold_aucs.append(roc_auc_score(y[va], pipe.predict_proba(X[va])[:, 1]))
                pipe.fit(X, y)

                fi = (
                    pipe.named_steps["clf"].feature_importances_
                    if name == "random_forest"
                    else np.abs(pipe.named_steps["clf"].coef_[0])
                )
                fi_series = pd.Series(fi, index=features).sort_values(ascending=False)

                print(f"  [{name}] {horizon_days}d horizon  CV AUC={np.mean(fold_aucs):.3f}"
                      f"  (±{np.std(fold_aucs):.3f})")

                results[name] = {
                    "pipeline":   pipe,
                    "cv_auc":     np.mean(fold_aucs),
                    "fold_aucs":  fold_aucs,
                    "feature_importances": fi_series,
                    "features":   features,
                }

            _plot_auc(results, horizon_days, output_dir)
            _plot_fi(results["random_forest"]["feature_importances"], horizon_days, output_dir)
            return results


        def _plot_auc(results, horizon_days, output_dir):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for ax, (name, r) in zip(axes, results.items()):
                aucs = r["fold_aucs"]
                bars = ax.bar(range(1, len(aucs)+1), aucs, color="#534AB7", alpha=0.85)
                ax.axhline(np.mean(aucs), linestyle="--", color="#D85A30",
                           label=f"Mean AUC {np.mean(aucs):.3f}")
                ax.set_ylim(0.45, 1.0)
                ax.set_xlabel("Fold"); ax.set_ylabel("AUC-ROC")
                ax.set_title(f"{name.replace(\'_\',' ').title()} — {horizon_days}d horizon")
                ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(output_dir / f"figures/cv_auc_{horizon_days}d.png", dpi=120, bbox_inches="tight")
            plt.close(fig)


        def _plot_fi(fi: pd.Series, horizon_days, output_dir):
            top = fi.head(15)
            fig, ax = plt.subplots(figsize=(8, 5))
            top[::-1].plot.barh(ax=ax, color="#534AB7", alpha=0.85)
            ax.set_xlabel("Importance")
            ax.set_title(f"RF feature importance — {horizon_days}d horizon")
            fig.tight_layout()
            fig.savefig(output_dir / f"figures/feature_importance_{horizon_days}d.png",
                        dpi=120, bbox_inches="tight")
            plt.close(fig)


        def train_monthly_regressor(
            panel: pd.DataFrame,
            output_dir: Path = Path("reports"),
        ) -> dict:
            output_dir = Path(output_dir)
            (output_dir / "figures").mkdir(parents=True, exist_ok=True)

            panel = panel.copy().sort_values(["oblast","date"])
            panel["y_monthly"] = (
                panel.groupby("oblast")["alert_count"]
                .transform(lambda x: x.shift(-1).rolling(30, min_periods=5).sum())
            )
            features = _available(panel)
            sub = panel.dropna(subset=features + ["y_monthly"])
            X, y = sub[features].values, sub["y_monthly"].values

            tscv = TimeSeriesSplit(n_splits=5)
            models = {
                "ridge": Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
                "rf_regressor": Pipeline([
                    ("sc", StandardScaler()),
                    ("reg", RandomForestRegressor(
                        n_estimators=300, max_depth=8, min_samples_leaf=5,
                        random_state=42, n_jobs=-1,
                    )),
                ]),
            }
            results = {}
            for name, pipe in models.items():
                maes = []
                for tr, va in tscv.split(X):
                    pipe.fit(X[tr], y[tr])
                    maes.append(mean_absolute_error(y[va], pipe.predict(X[va])))
                pipe.fit(X, y)
                print(f"  [monthly/{name}] CV MAE = {np.mean(maes):.1f} alerts")
                results[name] = {"pipeline": pipe, "cv_mae": np.mean(maes), "features": features}

            # Actual vs predicted chart
            pred = results["rf_regressor"]["pipeline"].predict(X)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(sub["date"].values, y, label="Actual", alpha=0.7, lw=1.2)
            ax.plot(sub["date"].values, pred, label="RF Predicted", alpha=0.7, lw=1.2, ls="--",
                    color="#D85A30")
            ax.set_title("Monthly alert count — actual vs RF forecast"); ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "figures/monthly_forecast.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            return results


        def train_all(
            feature_parquet: str | Path = "data/features.parquet",
            output_dir: Path = Path("reports"),
        ) -> dict:
            panel = pd.read_parquet(feature_parquet)
            all_results = {}
            for label, h in [("1d", 1), ("7d", 7)]:
                print(f"\\n── Horizon: {label} ──")
                all_results[label] = train_classifier(panel, h, output_dir=output_dir)
            print("\\n── Monthly regressor ──")
            all_results["monthly"] = train_monthly_regressor(panel, output_dir=output_dir)
            return all_results


        if __name__ == "__main__":
            results = train_all()
            with open("data/models.pkl", "wb") as f:
                pickle.dump(results, f)
            print("Models saved → data/models.pkl")
    ''')


def write_maps(root: Path):
    write(root / "src" / "maps.py", '''
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
                f\'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);\'
                f\'z-index:1000;background:white;padding:8px 18px;border-radius:8px;\'
                f\'box-shadow:0 2px 8px rgba(0,0,0,.2);font-family:sans-serif;font-size:14px;">\'
                f\'<b>Ukraine Air Raid Alert Status</b> &nbsp;|&nbsp;\'
                f\'<span style="color:#e74c3c">● Active</span> &nbsp;\'
                f\'<span style="color:#27ae60">● Clear</span> &nbsp;|&nbsp;\'
                f\'Updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</div>\'
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
                        f"<b>{row[\'oblast\']}</b><br>"
                        f"Alert rate: {row[\'alert_rate\']:.1%}<br>"
                        f"Total alerts: {int(row[\'alert_count_total\'])}<br>"
                        f"Avg duration: {row[\'avg_duration\']:.0f} min",
                        max_width=220,
                    ),
                    tooltip=f"{row[\'oblast\']}: {val:.2f}",
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
    ''')


def write_patterns(root: Path):
    write(root / "src" / "patterns.py", '''
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
                ax.set_xticks([0,1,2,3]); ax.set_xticklabels(["Q1\\n(low)","Q2","Q3","Q4\\n(high)"])
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
            if "news_severity" not in panel.columns:
                print("  No news features – skipping news effect plot")
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
                ax.boxplot(grps, labels=["No visit","NATO/EU visit"], patch_artist=True,
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
            df_raw = pd.read_csv(df_raw_path, index_col=0)
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
    ''')


def write_dashboard(root: Path):
    write(root / "dashboard" / "app.py", '''
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
            st.title("Ukraine Air Raid\\nAnalysis")
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
    ''')


def write_main(root: Path):
    write(root / "main.py", '''
        #!/usr/bin/env python3
        """
        main.py – end-to-end pipeline runner.

        Usage:
            python main.py data/alerts.csv
            python main.py data/alerts.csv --skip-news --skip-maps
        """

        import argparse, pickle, sys
        from pathlib import Path

        ROOT = Path(__file__).resolve().parent
        sys.path.insert(0, str(ROOT / "src"))


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("csv", nargs="?", default="data/alerts.csv")
            parser.add_argument("--skip-news",    action="store_true")
            parser.add_argument("--skip-weather", action="store_true")
            parser.add_argument("--skip-train",   action="store_true")
            parser.add_argument("--skip-maps",    action="store_true")
            parser.add_argument("--skip-patterns",action="store_true")
            args = parser.parse_args()

            data_dir    = ROOT / "data"
            reports_dir = ROOT / "reports"
            feat_f      = data_dir / "features.parquet"
            models_f    = data_dir / "models.pkl"

            print("\\n" + "="*55)
            print("STEP 1 — Feature engineering")
            print("="*55)
            from features import build_feature_set
            build_feature_set(args.csv, data_dir=data_dir)

            if not args.skip_train:
                print("\\n" + "="*55)
                print("STEP 2 — Model training")
                print("="*55)
                from models import train_all
                results = train_all(feat_f, output_dir=reports_dir)
                with open(models_f, "wb") as f: pickle.dump(results, f)
                print(f"  Models saved → {models_f}")

            if not args.skip_patterns:
                print("\\n" + "="*55)
                print("STEP 3 — Pattern analysis")
                print("="*55)
                from patterns import run_all
                run_all(args.csv, feat_f, output_dir=reports_dir / "figures")

            if not args.skip_maps:
                print("\\n" + "="*55)
                print("STEP 4 — Maps")
                print("="*55)
                import os, pandas as pd
                from maps import risk_heatmap_folium, risk_heatmap_static, live_alert_map
                panel = pd.read_parquet(feat_f)
                live_alert_map(api_key=os.getenv("UKRAINE_ALARM_API_KEY",""),
                               output_path=reports_dir/"live_alerts.html")
                risk_heatmap_folium(panel, output_path=reports_dir/"risk_heatmap.html")
                risk_heatmap_static(panel, output_path=reports_dir/"figures/risk_heatmap_static.png")

            print("\\n" + "="*55)
            print("✓ Pipeline complete!")
            print(f"  Launch dashboard:  streamlit run dashboard/app.py")
            print("="*55)


        if __name__ == "__main__":
            main()
    ''')


# ══════════════════════════════════════════════════════════════════════════════
# SCAFFOLD RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def scaffold(root: Path, venv: bool = True):
    root.mkdir(parents=True, exist_ok=True)
    print(f"\n{h('Scaffolding project')} → {b(str(root))}\n")

    # Directory tree
    for d in ["data", "src", "dashboard/components", "reports/figures", "notebooks"]:
        (root / d).mkdir(parents=True, exist_ok=True)

    # Touch __init__ files
    for pkg in ["src", "dashboard"]:
        (root / pkg / "__init__.py").touch()

    print(g("Writing files:"))
    write_requirements(root)
    write_env_example(root)
    write_gitignore(root)
    write_readme(root)
    write_features(root)
    write_models(root)
    write_maps(root)
    write_patterns(root)
    write_dashboard(root)
    write_main(root)

    # Optional venv
    if venv:
        print(f"\n{h('Creating virtual environment...')}")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(root / "venv")], check=True)
            pip = str(root / "venv" / ("Scripts/pip" if sys.platform=="win32" else "bin/pip"))
            print(f"  Installing requirements (this may take a minute)...")
            subprocess.run([pip, "install", "-r", str(root / "requirements.txt"), "-q"], check=True)
            print(g("  ✓ Dependencies installed"))
        except Exception as e:
            print(y(f"  ⚠ Venv/install failed: {e}. Install manually with pip install -r requirements.txt"))

    # Final instructions
    activate = (
        f".\\\\venv\\\\Scripts\\\\activate" if sys.platform == "win32"
        else f"source {root}/venv/bin/activate"
    )
    print(f"""
{h('═' * 55)}
{g('✓ Project scaffolded successfully!')}
{h('═' * 55)}

{b('Next steps:')}

  1. Activate the environment:
     {y(activate)}

  2. Copy your CSV to:
     {y(str(root / 'data/alerts.csv'))}

  3. Set up API keys:
     {y(f'cp {root}/.env.example {root}/.env')}
     Then edit .env with your keys (all optional)

  4. Run the full pipeline:
     {y(f'python {root}/main.py data/alerts.csv')}

     (skip slow steps while developing):
     {y(f'python {root}/main.py data/alerts.csv --skip-news --skip-weather')}

  5. Launch the dashboard:
     {y(f'streamlit run {root}/dashboard/app.py')}

{h('API keys (all optional – pipeline degrades gracefully):')}
  NEWSAPI_KEY         → newsapi.org/register  (free, 100 req/day)
  ANTHROPIC_API_KEY   → console.anthropic.com (for news scoring)
  UKRAINE_ALARM_API_KEY → ukrainealarm.com   (for live map)
{h('═' * 55)}
""")


def main():
    parser = argparse.ArgumentParser(
        description="Scaffold the Ukraine air raid alert analysis project"
    )
    parser.add_argument(
        "--dir", default="ukraine_alerts",
        help="Target directory (default: ./ukraine_alerts)"
    )
    parser.add_argument(
        "--no-venv", action="store_true",
        help="Skip creating a virtual environment"
    )
    args = parser.parse_args()
    scaffold(Path(args.dir).resolve(), venv=not args.no_venv)


if __name__ == "__main__":
    main()