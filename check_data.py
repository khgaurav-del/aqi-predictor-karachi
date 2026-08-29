"""Quick feature-store check for after a backfill or pipeline run."""
import pandas as pd
from src.database import Database

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

db = Database()
df = db.load_all_features()

print(f"Total rows: {len(df)}")
print(f"Date range: {df['datetime'].min()} -> {df['datetime'].max()}")
print(f"\nColumns ({len(df.columns)}):\n{list(df.columns)}")

print("\n--- Null counts per column ---")
print(df.isnull().sum().sort_values(ascending=False))

print("\n--- First 3 rows ---")
print(df.head(3))

print("\n--- Last 3 rows ---")
print(df.tail(3))

print("\n--- AQI target stats ---")
print(df["aqi"].describe())
print("\n--- target_aqi_72h (non-null rows usable for training) ---")
print(f"Non-null target_aqi_72h count: {df['target_aqi_72h'].notna().sum()} / {len(df)}")
