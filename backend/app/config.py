"""Application configuration from environment variables."""

import os
from dotenv import load_dotenv

# In production (Railway), env vars are set in the dashboard — no .env file needed.
# Locally, load from .env two levels up.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


class Settings:
    # Neo4j
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")

    # Gemini (placeholder LLM)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # LLM Server (future: fine-tuned model via vLLM)
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "not-needed")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "")

    # CORS — configurable via env var (comma-separated)
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if o.strip()
    ]


settings = Settings()
