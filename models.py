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

WEATHER_FEATURES = ["wind_speed_max", "precipitation", "cloud_cover", "temperature", "high_wind"]
NEWS_FEATURES    = ["news_severity", "nato_visit_bin", "peace_talks_bin"]

def _select_top_features(panel: pd.DataFrame, target_col: str, features: list[str], top_n: int = 12) -> list[str]:
    """Train a quick RF on all available features, return the top_n most important."""
    sub = panel.dropna(subset=features + [target_col])
    X, y = sub[features].values, sub[target_col].values
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)
    fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    return fi.head(top_n).index.tolist()


def _make_binary_target(panel: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """1 if any alert occurs in this oblast within the next horizon_days days."""
    panel = panel.copy().sort_values(["oblast","date"])
    panel[f"y_{horizon_days}d"] = (
        panel.groupby("oblast")["alert_count"]
        .transform(lambda x: x.shift(-horizon_days).rolling(horizon_days, min_periods=1).max())
        .clip(upper=1)
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
    all_features = _available(panel)
    features = _select_top_features(panel, target, all_features, top_n=12)
    print(f"  Selected top {len(features)} features: {features}")
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

def train_comparison_models(
    panel: pd.DataFrame,
    base_features: list[str],
    horizon_days: int = 1,
    n_splits: int = 5,
    output_dir: Path = Path("reports"),
) -> dict:
    """
    Train 3 extra RF variants on top of the base 12 features:
      - base + weather
      - base + news
      - base + weather + news
    Compares CV AUC against the base model to see if each addition helps.
    """
    panel = _make_binary_target(panel, horizon_days)
    target = f"y_{horizon_days}d"

    variants = {
        "base_plus_weather": base_features + [f for f in WEATHER_FEATURES if f in panel.columns],
        "base_plus_news":    base_features + [f for f in NEWS_FEATURES if f in panel.columns],
        "base_plus_both":    base_features + [f for f in WEATHER_FEATURES + NEWS_FEATURES if f in panel.columns],
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {}

    for name, feats in variants.items():
        feats = list(dict.fromkeys(feats))  # dedupe, preserve order
        sub = panel.dropna(subset=feats + [target]).copy()
        X, y = sub[feats].values, sub[target].values

        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=10,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ])

        fold_aucs = []
        for tr, va in tscv.split(X):
            pipe.fit(X[tr], y[tr])
            fold_aucs.append(roc_auc_score(y[va], pipe.predict_proba(X[va])[:, 1]))
        pipe.fit(X, y)

        fi_series = pd.Series(
            pipe.named_steps["clf"].feature_importances_, index=feats
        ).sort_values(ascending=False)

        print(f"  [{name}] {horizon_days}d horizon  CV AUC={np.mean(fold_aucs):.3f} (±{np.std(fold_aucs):.3f})")

        results[name] = {
            "pipeline":  pipe,
            "cv_auc":    np.mean(fold_aucs),
            "fold_aucs": fold_aucs,
            "feature_importances": fi_series,
            "features":  feats,
        }

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
        ax.set_title(f"{name.replace('_',' ').title()} — {horizon_days}d horizon")
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
        print(f"\n── Horizon: {label} ──")
        all_results[label] = train_classifier(panel, h, output_dir=output_dir)

    print("\n── Comparison models: weather / news / both (1d horizon) ──")
    base_12 = all_results["1d"]["random_forest"]["features"]
    all_results["comparison"] = train_comparison_models(
        panel, base_features=base_12, horizon_days=1, output_dir=output_dir
    )

    print("\n── Monthly regressor ──")
    all_results["monthly"] = train_monthly_regressor(panel, output_dir=output_dir)
    return all_results


if __name__ == "__main__":
    results = train_all()
    with open("data/models.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Models saved → data/models.pkl")
