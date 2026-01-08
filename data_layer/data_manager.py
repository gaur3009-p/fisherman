import pandas as pd
import os
from datetime import datetime
from config.settings import Settings

class DatasetManager:
    def __init__(self):
        os.makedirs(os.path.dirname(Settings.DATASET_FILE), exist_ok=True)
        if not os.path.exists(Settings.DATASET_FILE):
            self._initialize_dataset()

    def _initialize_dataset(self):
        columns = [
            "record_id",
            "audio_path",
            "native_text",
            "english_text",
            "issue_title",
            "clean_summary",
            "sentiment",
            "urgency",
            "ngo_action",
            "rag_context",
            "timestamp"
        ]
        pd.DataFrame(columns=columns).to_csv(Settings.DATASET_FILE, index=False)

    def save_record(self, record: dict):
        df = pd.DataFrame([record])
        df.to_csv(Settings.DATASET_FILE, mode="a", header=False, index=False)
