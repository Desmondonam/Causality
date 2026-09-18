"""Backend settings, loaded from environment variables (see .env.example)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CAUSALITY_")

    # Artifacts produced by `python -m ml.pipeline` at the repo root.
    models_dir: Path = Path(__file__).resolve().parent.parent.parent / "models"
    reports_dir: Path = Path(__file__).resolve().parent.parent.parent / "outputs" / "reports"

    # Comma-separated list of allowed frontend origins (Vercel URL(s) + local dev).
    cors_origins: str = "http://localhost:3000"

    api_title: str = "Causality API"
    api_version: str = "1.0.0"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
