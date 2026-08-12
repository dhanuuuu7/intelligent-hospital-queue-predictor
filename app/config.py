"""
Step 9.2: Application Environment Configuration
Loads settings from environment variables or .env file cleanly.
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Settings:
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH", 
        os.path.join(os.path.dirname(__file__), "..", "models", "artifacts", "final_waiting_time_model.joblib")
    )
    INTERVAL_METADATA_PATH: str = os.getenv(
        "INTERVAL_METADATA_PATH", 
        os.path.join(os.path.dirname(__file__), "..", "models", "artifacts", "prediction_interval_metadata.json")
    )
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0.0")
    FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")


settings = Settings()
