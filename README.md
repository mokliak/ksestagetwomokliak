# Ukraine Air Raid Alert Analysis

Real alert data, real forecasting models, and a live dashboard — built to find out whether weather, news, and history can actually predict when and where the next air raid alert will hit.

---

## Watch the Demo

The video of the final result is in the [`media`](media/demo.mp4) folder of this repo.

GitHub doesn't preview files over 50MB inline, so here's a backup link to the same video, just in case:

**[Watch on Google Drive](https://drive.google.com/file/d/1VxFZQvOMyc1bAfkMUkk1hmnpj4LBdAdb/view?usp=sharing)**

---

## What It Does

- Turns raw alert records into a clean daily panel, one row per date x oblast
- Adds weather from Russian launch-site regions, news severity scoring, and geography
- Trains Random Forest and Logistic Regression models to forecast alerts 1 and 7 days ahead
- Tests whether weather and news actually improve predictions, or whether alert history alone does the job
- Wraps everything in an interactive dashboard — forecasts, live map, pattern analysis, summary findings

---

## Quick Start

```bash
git clone <your-repo-url>
cd ksestagetwomokliak
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # add your own API keys here
python3 main.py data/official_data_en.csv
python3 -m streamlit run dashboard/app.py
```

## Running Without API Keys

| Feature | Needs a key? | If skipped |
|---|---|---|
| Core pipeline & dashboard | No | Fully works |
| Weather | No — Open-Meteo is free | Works automatically |
| News scoring | Yes | Defaults to 0, rest still works |
| Live map | Yes | Shows no live data, heat map still works |

No news keys? Run:

```bash
python3 main.py data/official_data_en.csv --skip-news
```

---

## One Thing Worth Knowing

Alerts logged under "Kyiv City" and "Kyivska oblast" are merged into one entry — they were being recorded inconsistently for what's effectively the same place. Data starts from 2024 onward.

---

## Why I Built This

This project pushed me past a single notebook — cleaning messy real-world data, wiring up multiple live APIs, training and honestly comparing forecasting models, and packaging all of it into something anyone can open and actually explore. I'm genuinely excited about bringing that same curiosity and follow-through to the KSE x HEC Paris program, and I'd love the chance to keep building things that turn data into something people can act on.

— Andrii Mokliak
