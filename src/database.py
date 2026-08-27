"""MongoDB Atlas handler — acts as our 'feature store'."""
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
import pandas as pd

from . import config


class Database:
    def __init__(self):
        config.require_env()
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.features = self.db[config.FEATURES_COLLECTION]
        self.models = self.db[config.MODELS_COLLECTION]
        self.predictions = self.db[config.PREDICTIONS_COLLECTION]
        # Unique index on datetime prevents duplicate hourly rows on re-runs
        self.features.create_index([("datetime", ASCENDING)], unique=True)

    # ---------------- Features ----------------
    def upsert_features(self, records: list[dict]) -> int:
        """Insert feature rows, skipping ones that already exist (by datetime)."""
        inserted = 0
        for rec in records:
            try:
                self.features.insert_one(rec)
                inserted += 1
            except DuplicateKeyError:
                continue
        return inserted

    def load_all_features(self) -> pd.DataFrame:
        cursor = self.features.find({}, {"_id": 0}).sort("datetime", ASCENDING)
        df = pd.DataFrame(list(cursor))
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def count_features(self) -> int:
        return self.features.count_documents({})

    # ---------------- Models ----------------
    def save_model_metadata(self, metadata: dict):
        self.models.insert_one(metadata)

    def get_latest_model_metadata(self, model_name: str = None):
        query = {"model_name": model_name} if model_name else {}
        return self.models.find_one(query, sort=[("trained_at", -1)])

    def list_models(self):
        return list(self.models.find({}, {"_id": 0}).sort("trained_at", -1))

    # ---------------- Predictions ----------------
    def save_predictions(self, records: list[dict]):
        if records:
            self.predictions.insert_many(records)

    def get_latest_predictions(self):
        return list(self.predictions.find({}, {"_id": 0}).sort("predicted_at", -1).limit(1))
