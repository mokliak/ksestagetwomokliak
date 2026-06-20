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
