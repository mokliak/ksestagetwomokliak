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
sys.path.insert(0, str(ROOT))


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

    print("\n" + "="*55)
    print("STEP 1 — Feature engineering")
    print("="*55)
    from features import build_feature_set
    build_feature_set(args.csv, data_dir=data_dir)

    if not args.skip_train:
        print("\n" + "="*55)
        print("STEP 2 — Model training")
        print("="*55)
        from models import train_all
        results = train_all(feat_f, output_dir=reports_dir)
        with open(models_f, "wb") as f: pickle.dump(results, f)
        print(f"  Models saved → {models_f}")

    if not args.skip_patterns:
        print("\n" + "="*55)
        print("STEP 3 — Pattern analysis")
        print("="*55)
        from patterns import run_all
        run_all(args.csv, feat_f, output_dir=reports_dir / "figures")

    if not args.skip_maps:
        print("\n" + "="*55)
        print("STEP 4 — Maps")
        print("="*55)
        import os, pandas as pd
        from maps import risk_heatmap_folium, risk_heatmap_static, live_alert_map
        panel = pd.read_parquet(feat_f)
        live_alert_map(api_key=os.getenv("UKRAINE_ALARM_API_KEY",""),
                       output_path=reports_dir/"live_alerts.html")
        risk_heatmap_folium(panel, output_path=reports_dir/"risk_heatmap.html")
        risk_heatmap_static(panel, output_path=reports_dir/"figures/risk_heatmap_static.png")

    print("\n" + "="*55)
    print("✓ Pipeline complete!")
    print(f"  Launch dashboard:  streamlit run dashboard/app.py")
    print("="*55)


if __name__ == "__main__":
    main()
