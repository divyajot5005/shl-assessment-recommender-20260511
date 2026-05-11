from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "shl-assessment-recommender"
    catalog_url: str = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
    sample_traces_url: str = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/sample_conversations.zip"
    request_timeout_seconds: int = 20
    max_recommendations: int = 10
    max_turns: int = 8
    llm_base_url: str | None = "https://api.groq.com/openai/v1"
    llm_api_key: str | None = None
    llm_model: str | None = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_state_extraction_enabled: bool = False
    llm_reply_rewrite_enabled: bool = True
    data_dir: Path = Path("data")

    @property
    def raw_catalog_path(self) -> Path:
        return self.data_dir / "raw" / "shl_product_catalog.json"

    @property
    def raw_traces_zip_path(self) -> Path:
        return self.data_dir / "raw" / "sample_conversations.zip"

    @property
    def public_traces_dir(self) -> Path:
        return self.data_dir / "traces" / "GenAI_SampleConversations"

    @property
    def processed_catalog_path(self) -> Path:
        return self.data_dir / "processed" / "catalog_records.json"

    @property
    def artifacts_dir(self) -> Path:
        return Path("artifacts")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "data"))

    def parse_bool_env(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    return Settings(
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        llm_state_extraction_enabled=parse_bool_env("LLM_STATE_EXTRACTION_ENABLED", False),
        llm_reply_rewrite_enabled=parse_bool_env("LLM_REPLY_REWRITE_ENABLED", True),
        data_dir=data_dir,
    )
