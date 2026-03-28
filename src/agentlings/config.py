from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    agent_api_key: str = ""
    agent_model: str = "claude-sonnet-4-6"
    agent_max_tokens: int = 4096
    agent_host: str = "0.0.0.0"
    agent_port: int = 8420
    agent_data_dir: Path = Path("./data")
    agent_name: str = "agentling"
    agent_description: str = "A lightweight AI agent"
    agent_system_prompt_file: str | None = None
    agent_log_level: str = "INFO"
    agent_llm_backend: Literal["anthropic", "mock"] = "anthropic"

    @model_validator(mode="after")
    def _ensure_data_dir(self) -> AgentConfig:
        self.agent_data_dir.mkdir(parents=True, exist_ok=True)
        return self
