"""Configuration for MCP eval."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Configuration class for MCP eval."""

    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "3000"))

    # LLM configuration
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")

    # MCP Server configuration
    MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://localhost:1984")

    # Timeout configuration
    DEFAULT_TIMEOUT: float = float(os.getenv("DEFAULT_TIMEOUT", "300.0"))
    TOOL_CALL_TIMEOUT: float = float(os.getenv("TOOL_CALL_TIMEOUT", "60.0"))
    # Transient tool errors get more than one chance, matching upstream
    # MCP-Atlas's April 2026 change. The timeout escalates per attempt
    # (60s, then 120s, then 180s at the default), because a tool that is
    # merely slow needs more time rather than another equally short window.
    # Set to 1 to disable retries and restore the pre-2026 behaviour.
    TOOL_CALL_ATTEMPTS: int = int(os.getenv("TOOL_CALL_ATTEMPTS", "3"))
    LIST_TOOLS_TIMEOUT: float = float(os.getenv("LIST_TOOLS_TIMEOUT", "180.0"))

    # Logging configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def validate_required_config(self) -> None:
        """Validate that all required configuration values are set."""
        required_configs = [("LLM_API_KEY", self.LLM_API_KEY)]

        missing_configs = []
        for name, value in required_configs:
            if not value or not value.strip():
                missing_configs.append(name)

        if missing_configs:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing_configs)}. "
                f"Please set these environment variables or add them to your .env file."
            )


config = Config()
