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
        GEMINI_KEY         = os.getenv("GEMINI_API_KEY", "")


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
                    print(f"  Fetching weather {i+1}/{len(oblasts)}: {ob}", end="\r")
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


        def _score_with_gemini(date_str: str, headlines: list[str], query_type: str) -> float:
            """
            Send headlines to Gemini Flash for 0-1 scoring (free tier, no credit card).
            query_type controls the prompt framing.
            """
            if not GEMINI_KEY or not headlines:
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
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_KEY}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompts[query_type]}]}],
                        "generationConfig": {"maxOutputTokens": 10, "temperature": 0.0},
                    },
                    timeout=20,
                )
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip().split()[0]
                return max(0.0, min(1.0, float(text)))
            except Exception as e:
                print(f"    Gemini error ({date_str}/{query_type}): {e}")
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
                    print(f"  News scoring {i+1}/{len(unique_dates)}: {d}", end="\r")
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

            print("\n[1/7] Loading raw CSV...")
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
            print(f"\n✓ Feature set saved → {out}")
            print(f"  Shape: {panel.shape}  ({panel.columns.tolist()[:8]}... +{len(panel.columns)-8} more)")
            return panel


        if __name__ == "__main__":
            import sys
            csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/alerts.csv"
            build_feature_set(csv_path)
