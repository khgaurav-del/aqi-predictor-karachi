"""
SHAP feature importance for Pearls AQI Predictor.

Explains which features matter most for the trained model's 72-hour AQI
predictions, saves summary plots, and prints a ranked importance list.

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

    # Use the model-agnostic Explainer so we do not depend on SHAP parsing
    # XGBoost internals that can change across versions.
    print("Using the model-agnostic Explainer to avoid SHAP/XGBoost version mismatch issues...")
    background = X_test.sample(min(50, len(X_test)), random_state=42)
    explainer = shap.Explainer(model.predict, background)
    shap_values = explainer(X_test)

    # Summary bar plot: average absolute SHAP value per feature.
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Feature Importance — 72h AQI Forecast")
    plt.tight_layout()
    bar_path = f"{OUTPUT_DIR}/shap_importance_bar.png"
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Saved bar plot -> {bar_path}")

    # Beeswarm plot: shows both direction and size of each effect.
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary — 72h AQI Forecast")
    plt.tight_layout()
    beeswarm_path = f"{OUTPUT_DIR}/shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    print(f"Saved beeswarm plot -> {beeswarm_path}")

    # Print the ranking as text so it is easy to reuse in a report.
    vals = np.abs(shap_values.values).mean(axis=0)
    importance = pd.Series(vals, index=feature_cols).sort_values(ascending=False)

    print("\n--- Top 15 features by mean |SHAP value| ---")
    print(importance.head(15).to_string())

    importance.to_csv(f"{OUTPUT_DIR}/shap_importance.csv", header=["mean_abs_shap"])
    print(f"\nSaved full ranking -> {OUTPUT_DIR}/shap_importance.csv")


if __name__ == "__main__":
    main()
