from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment aligned with pharmacio-backend OCR / storage settings where applicable."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Inbound dispatch (must match backend AI_ENGINE_API_KEY — compared as full Authorization value)
    ai_engine_api_key: str = Field(default="", alias="AI_ENGINE_API_KEY")

    # Outbound callback (must match backend INTERNAL_SERVICE_TOKEN)
    internal_service_token: str = Field(default="", alias="INTERNAL_SERVICE_TOKEN")
    backend_base_url: str = Field(default="http://localhost:8000", alias="BACKEND_BASE_URL")

    # Same semantics as backend FILE_STORAGE_BACKEND / USE_S3
    storage_backend: str = Field(default="local", alias="STORAGE_BACKEND")

    # Local filesystem storage: base directory containing uploaded keys (Django MEDIA_ROOT)
    local_media_root: str = Field(default="media", alias="LOCAL_MEDIA_ROOT")

    # S3-compatible (MinIO): same vars as Django storages
    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_s3_endpoint_url: str | None = Field(default=None, alias="AWS_S3_ENDPOINT_URL")
    aws_storage_bucket_name: str | None = Field(default=None, alias="AWS_STORAGE_BUCKET_NAME")
    aws_s3_region_name: str = Field(default="us-east-1", alias="AWS_S3_REGION_NAME")

    callback_timeout_seconds: int = Field(default=120, alias="OCR_CALLBACK_TIMEOUT_SECONDS")


def get_settings() -> Settings:
    return Settings()
