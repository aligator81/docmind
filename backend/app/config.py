from pydantic_settings import BaseSettings
import os
import json
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables from .env file in the project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

class Settings(BaseSettings):
    # Core settings
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Database
    database_url: str = os.getenv("NEON_CONNECTION_STRING", "")

    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")

    # Chat Model Configuration
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    mistral_chat_model: str = os.getenv("MISTRAL_CHAT_MODEL", "mistral-large-latest")
    
    # Embedding Model Configuration
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    mistral_embedding_model: str = os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
    
    # Generation Parameters
    openai_max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
    mistral_max_tokens: int = int(os.getenv("MISTRAL_MAX_TOKENS", "1000"))
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    mistral_temperature: float = float(os.getenv("MISTRAL_TEMPERATURE", "0.7"))

    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_reload: bool = os.getenv("API_RELOAD", "true").lower() == "true"

    # File upload settings
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))
    # Comprehensive list of file formats supported by Docling document processor
    allowed_extensions: List[str] = json.loads(os.getenv("ALLOWED_EXTENSIONS", '["pdf","docx","doc","xlsx","xls","pptx","ppt","odt","ods","odp","rtf","md","html","htm","txt","png","jpg","jpeg","tiff","bmp","gif","svg","csv","tsv","xml","json","epub"]'))

    # Security Configuration
    file_validation_enabled: bool = os.getenv("FILE_VALIDATION_ENABLED", "false").lower() == "true"
    rate_limiting_enabled: bool = os.getenv("RATE_LIMITING_ENABLED", "false").lower() == "true"
    compression_bomb_detection: bool = os.getenv("COMPRESSION_BOMB_DETECTION", "false").lower() == "true"

    # Logging Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")
    enable_structured_logging: bool = os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true"

    # Monitoring Configuration
    enable_prometheus_metrics: bool = os.getenv("ENABLE_PROMETHEUS_METRICS", "false").lower() == "true"
    metrics_port: int = int(os.getenv("METRICS_PORT", "9090"))

    # Development Configuration
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"
    testing: bool = os.getenv("TESTING", "false").lower() == "true"


    # Email Configuration
    smtp_server: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "your-email@gmail.com")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "your-app-password")

    # Admin Configuration
    admin_email: str = os.getenv("ADMIN_EMAIL", "admin@example.com")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "secure-admin-password")

    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields to prevent validation errors

    @property
    def max_tokens(self) -> int:
        """Get max tokens based on current provider preference"""
        # Default to OpenAI settings for now
        return self.openai_max_tokens

    @property
    def temperature(self) -> float:
        """Get temperature based on current provider preference"""
        # Default to OpenAI settings for now
        return self.openai_temperature

settings = Settings()