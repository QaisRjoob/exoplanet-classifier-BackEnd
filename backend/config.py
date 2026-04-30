"""
Configuration settings for the FastAPI application
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_TITLE: str = "KOI Classification API"
    API_DESCRIPTION: str = "API for classifying Kepler Objects of Interest (KOI) into CONFIRMED, CANDIDATE, or FALSE POSITIVE using GPU-accelerated ensemble ML"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Model Settings
    MODEL_PATH: str = "saved_models/stacking_model.pkl"
    METADATA_PATH: str = "saved_models/model_metadata.pkl"
    MODEL_METADATA_PATH: str = "saved_models/model_metadata.pkl"  # Alias for compatibility
    
    # CORS Settings — allow all localhost/127.0.0.1 ports for local dev
    CORS_ORIGINS: List[str] = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Rate Limiting (optional)
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_PER_MINUTE: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()
