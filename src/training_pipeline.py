"""
Training Pipeline - Pearls AQI Predictor (Karachi)
====================================================
- Pulls features from MongoDB
- Chronological 80/20 train/test split (no shuffling - this is time series)
- Trains Ridge Regression, Random Forest, XGBoost
- Evaluates with RMSE, MAE, R2
- Saves the best model locally (models/best_model.joblib) and its
  metadata to MongoDB's `models` collection (model registry)

Usage:
    python -m src.training_pipeline
"""
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from . import config
from .database import Database

MODELS_DIR = "models"

# Columns that must NOT be used as model inputs (identifiers / leakage / target)
DROP_COLS = ["datetime", "aqi", "target_aqi_72h"]


def load_training_data(db: Database) -> pd.DataFrame:
    df = db.load_all_features()
    df = df.sort_values("datetime").reset_index(drop=True)

    # --- Data quality fix ---
    # OpenWeather occasionally returns a glitched pollutant reading for a
    # single hour (e.g. stored aqi=500 while pm2_5=12, which is physically
    # inconsistent since pm2_5=12 should map to an AQI of ~50). Because
    # `target_aqi_72h` was pre-computed as a shift of `aqi` at ingestion
    # time, a single bad row doesn't just corrupt itself — it becomes the
    # *target* for a different row 72h earlier too. So we recompute `aqi`
    # fresh from `pm2_5` (a deterministic, always-consistent function) and
    # rebuild `target_aqi_72h` locally from that clean series, rather than
    # trusting the values already stored in Mongo.
    from .aqi_utils import pm25_to_aqi
    df["aqi"] = df["pm2_5"].apply(pm25_to_aqi)
    df["target_aqi_72h"] = df["aqi"].shift(-72)

    # Drop rows without a usable target (the most recent 72h)
    df = df.dropna(subset=["target_aqi_72h"]).reset_index(drop=True)
    # Drop rows still missing lag features (the earliest 72h)
    lag_cols = [c for c in df.columns if "lag_72h" in c]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    split_idx = int(len(df) * (1 - test_frac))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in DROP_COLS]


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def get_model_instances():
    """Fresh, unfitted model instances (used per-fold in CV and for the final fit).

    Ridge is wrapped with StandardScaler: our lag/rolling features are highly
    correlated with raw pm2_5 (multicollinearity), and without scaling, Ridge's
    coefficients can blow up on ill-conditioned folds (seen as RMSE=51 on one
    CV fold before this fix). A stronger alpha also adds more regularization
    headroom given we only have ~900 rows and 32 features.
    """
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
        "xgboost": xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42, n_jobs=-1,
        ),
    }


def time_series_cv_evaluate(df, feature_cols, n_splits=5):
    """Rolling-origin time series CV: trains on an expanding window, tests on
    the next chunk, repeated n_splits times. Averaging across folds gives a
    far more reliable estimate than one fixed chronological split, which is
    highly sensitive to whether the single test window happens to look like
    the training period or not (exactly what we saw with a single split)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X, y = df[feature_cols], df["target_aqi_72h"]

    fold_metrics = {name: [] for name in get_model_instances()}

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, model in get_model_instances().items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            m = evaluate(y_test, preds)
            fold_metrics[name].append(m)
            print(f"  fold {fold_i} [{name:>13}] RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} R2={m['r2']:.3f} "
                  f"(train={len(train_idx)}, test={len(test_idx)})")

    # Average metrics across folds
    avg_metrics = {}
    for name, folds in fold_metrics.items():
        avg_metrics[name] = {
            "rmse": float(np.mean([f["rmse"] for f in folds])),
            "mae": float(np.mean([f["mae"] for f in folds])),
            "r2": float(np.mean([f["r2"] for f in folds])),
        }
    return avg_metrics


def train_models(train_df, test_df, feature_cols):
    X_train, y_train = train_df[feature_cols], train_df["target_aqi_72h"]
    X_test, y_test = test_df[feature_cols], test_df["target_aqi_72h"]

    results = {}
    for name, model in get_model_instances().items():
        model.fit(X_train, y_train)
        results[name] = {
            "model": model,
            "metrics": evaluate(y_test, model.predict(X_test)),
        }
    return results


def main():
    config.require_env()
    db = Database()

    df = load_training_data(db)
    print(f"Training rows available: {len(df)}")
    if len(df) < 50:
        print("WARNING: very few rows — results will be unstable. Consider backfilling more days.")

    feature_cols = get_feature_columns(df)
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    train_df, test_df = chronological_split(df, test_frac=0.2)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Train range: {train_df['datetime'].min()} -> {train_df['datetime'].max()}")
    print(f"Test range:  {test_df['datetime'].min()} -> {test_df['datetime'].max()}")

    # --- Naive baselines (sanity check: models should beat these) ---
    naive_pred = train_df["target_aqi_72h"].mean()
    naive_rmse = np.sqrt(mean_squared_error(test_df["target_aqi_72h"], [naive_pred] * len(test_df)))
    persistence_rmse = np.sqrt(mean_squared_error(test_df["target_aqi_72h"], test_df["aqi"]))
    print(f"\nBaseline RMSE (predict train mean): {naive_rmse:.2f}")
    print(f"Baseline RMSE (persistence, aqi unchanged in 72h): {persistence_rmse:.2f}")

    # --- Time-series cross-validation (the real, reportable evaluation) ---
    print("\n--- Time-series cross-validation (5 rolling folds) ---")
    cv_metrics = time_series_cv_evaluate(df, feature_cols, n_splits=5)
    print("\n--- CV average across folds ---")
    for name, m in cv_metrics.items():
        beats_baseline = "✓ beats baseline" if m["rmse"] < naive_rmse else "✗ worse than baseline"
        print(f"{name:>15}  RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}  ({beats_baseline})")

    # --- Final fit on the 80/20 split (for the deployed model + dashboard) ---
    print("\n--- Final single-split fit (used for deployed model) ---")
    results = train_models(train_df, test_df, feature_cols)

    print("\n--- Model comparison ---")
    for name, r in results.items():
        m = r["metrics"]
        print(f"{name:>15}  RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")

    # Pick best by CV RMSE (more reliable than a single split's RMSE)
    best_name = min(cv_metrics, key=lambda n: cv_metrics[n]["rmse"])
    best_model = results[best_name]["model"]
    best_metrics = results[best_name]["metrics"]
    print(f"\nBest model (by CV): {best_name} (CV RMSE={cv_metrics[best_name]['rmse']:.2f})")

    # Save locally
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.joblib"))
    print(f"Saved best model -> {model_path}")

    # Also save each model individually (useful for report/comparison + SHAP later)
    for name, r in results.items():
        joblib.dump(r["model"], os.path.join(MODELS_DIR, f"{name}_model.joblib"))

    # Save metadata / registry entry to MongoDB
    metadata = {
        "trained_at": datetime.now(tz=timezone.utc),
        "best_model_name": best_name,
        "feature_columns": feature_cols,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "cv_metrics": cv_metrics,
        "single_split_metrics": {name: r["metrics"] for name, r in results.items()},
        "best_metrics": best_metrics,
    }
    db.save_model_metadata(metadata)
    print("Saved model metadata to MongoDB `models` collection.")


if __name__ == "__main__":
    main()
