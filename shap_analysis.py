"""
SHAP Feature Importance - Pearls AQI Predictor
================================================
Explains which features drive the trained model's 72h-ahead AQI predictions.
Generates a summary plot (saved as PNG) and prints ranked feature importance.

Usage:
    python shap_analysis.py
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
import shap

from src.database import Database
from src.training_pipeline import load_training_data, chronological_split

MODEL_PATH = "models/best_model.joblib"
FEATURE_COLUMNS_PATH = "models/feature_columns.joblib"
OUTPUT_DIR = "reports"


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLUMNS_PATH)

    db = Database()
    df = load_training_data(db)
    _, test_df = chronological_split(df, test_frac=0.2)
    X_test = test_df[feature_cols]

    print(f"Computing SHAP values for {len(X_test)} test rows...")

    # Use the model-agnostic Explainer for all model types. TreeExplainer is
    # faster for tree models, but it depends on parsing internal model
    # attributes that vary between XGBoost versions — this installation's
    # XGBoost stores base_score in a format TreeExplainer's parser rejects
    # (see: https://github.com/shap/shap/issues — a known version mismatch).
    # The generic Explainer works against any model's predict() function
    # regardless of internal format, so it avoids that entirely.
    print("Using model-agnostic Explainer (avoids XGBoost/SHAP version mismatch in TreeExplainer)...")
    background = X_test.sample(min(50, len(X_test)), random_state=42)
    explainer = shap.Explainer(model.predict, background)
    shap_values = explainer(X_test)

    # --- Summary bar plot: mean |SHAP value| per feature ---
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Feature Importance — 72h AQI Forecast")
    plt.tight_layout()
    bar_path = f"{OUTPUT_DIR}/shap_importance_bar.png"
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Saved bar plot -> {bar_path}")

    # --- Beeswarm summary plot: shows direction of effect, not just magnitude ---
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary — 72h AQI Forecast")
    plt.tight_layout()
    beeswarm_path = f"{OUTPUT_DIR}/shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    print(f"Saved beeswarm plot -> {beeswarm_path}")

    # --- Print ranked importance as text (easy to paste into report) ---
    vals = np.abs(shap_values.values).mean(axis=0)
    importance = pd.Series(vals, index=feature_cols).sort_values(ascending=False)

    print("\n--- Top 15 features by mean |SHAP value| ---")
    print(importance.head(15).to_string())

    importance.to_csv(f"{OUTPUT_DIR}/shap_importance.csv", header=["mean_abs_shap"])
    print(f"\nSaved full ranking -> {OUTPUT_DIR}/shap_importance.csv")


if __name__ == "__main__":
    main()
