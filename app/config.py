import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    env: str = "production"
    
    class Config:
        env_file = ".env"

settings = Settings()