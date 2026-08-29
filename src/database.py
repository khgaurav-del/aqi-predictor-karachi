"""MongoDB Atlas handler used as the project's feature store."""
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
        # Keep datetime unique so reruns do not create duplicate hourly rows.
        self.features.create_index([("datetime", ASCENDING)], unique=True)

    # Features
    def upsert_features(self, records: list[dict]) -> int:
        """Insert or update feature rows by datetime.

        Using replace_one(upsert=True) means rerunning the pipeline for an
        existing hour overwrites the old row instead of skipping it.
        """
        upserted = 0
        for rec in records:
            result = self.features.replace_one({"datetime": rec["datetime"]}, rec, upsert=True)
            if result.upserted_id is not None or result.modified_count > 0:
                upserted += 1
        return upserted

    def load_all_features(self) -> pd.DataFrame:
        cursor = self.features.find({}, {"_id": 0}).sort("datetime", ASCENDING)
        df = pd.DataFrame(list(cursor))
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def load_recent_features(self, hours: int = 80) -> pd.DataFrame:
        """Load only the most recent N hours of rows.

        The hourly pipeline uses this to rebuild lag and rolling features
        without fetching the full feature store every time.
        """
        from datetime import datetime, timedelta, timezone
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        cursor = self.features.find({"datetime": {"$gte": cutoff}}, {"_id": 0}).sort("datetime", ASCENDING)
        df = pd.DataFrame(list(cursor))
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def count_features(self) -> int:
        return self.features.count_documents({})

    # Models
    def save_model_metadata(self, metadata: dict):
        self.models.insert_one(metadata)

    def get_latest_model_metadata(self, model_name: str = None):
        query = {"model_name": model_name} if model_name else {}
        return self.models.find_one(query, sort=[("trained_at", -1)])

    def list_models(self):
        return list(self.models.find({}, {"_id": 0}).sort("trained_at", -1))

    # Predictions
    def save_predictions(self, records: list[dict]):
        if records:
            self.predictions.insert_many(records)

    def get_latest_predictions(self):
        return list(self.predictions.find({}, {"_id": 0}).sort("predicted_at", -1).limit(1))
