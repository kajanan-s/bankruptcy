from pydantic_settings import BaseSettings
from pathlib import Path
import logging

class Settings(BaseSettings):
    ENV: str = "production"
    MODEL_PATH: str = str(Path("model.pkl"))
    FEATURE_NAMES_PATH: str = str(Path("feature_names.pkl"))
    SECRET_KEY: str = "your-secret-key-here"
    GCP_PROJECT_ID: str = "your-gcp-project-id"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

    def validate_paths(self):
        for path in [self.MODEL_PATH, self.FEATURE_NAMES_PATH]:
            if not Path(path).parent.exists():
                logging.error(f"Parent directory does not exist: {path}")
                raise FileNotFoundError(f"Ensure parent directory exists for {path}")
            if not Path(path).exists():
                logging.warning(f"File not found: {path}")

settings = Settings()
settings.validate_paths()