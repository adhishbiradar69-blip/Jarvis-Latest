"""
config.py
=========
Single source of truth for every configurable value in Jarvis.

Design decisions:
- All settings live here so changing a value (e.g. the model name,
  memory thresholds) never requires hunting through multiple files.
- Uses dataclasses so settings are typed and IDE-friendly.
- Loaded once at import time; modules import the singleton instances
  (GROQ_CFG, MEMORY_CFG, etc.) rather than the classes themselves.
- Secrets come from .env via python-dotenv — never hardcoded.

Future expansion:
- Add OllamaConfig, RAGConfig, VisionConfig etc. here when those
  features are implemented.  Nothing else needs to change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (the directory that contains this file)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
MEMORY_DIR = ROOT_DIR / "Memory"
ASSETS_DIR = ROOT_DIR / "Assets"


# ---------------------------------------------------------------------------
# Groq / LLM
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroqConfig:
    """Configuration for the Groq-hosted LLM."""

    api_key: str = field(default_factory=lambda: os.environ["GROQ_API_KEY"])
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.7
    max_tokens: int = 1024

    # Tool-call settings
    max_tool_calls_per_request: int = 3


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryConfig:
    """Thresholds and file paths for the three-layer memory system."""

    # File paths (relative to MEMORY_DIR)
    recent_path: Path = field(default_factory=lambda: MEMORY_DIR / "recent.json")
    summaries_path: Path = field(default_factory=lambda: MEMORY_DIR / "summaries.json")
    profile_path: Path = field(default_factory=lambda: MEMORY_DIR / "profile.json")

    # How many messages to keep in the rolling recent window
    recent_max_messages: int = 10

    # How many user exchanges before we summarise + clear recent
    summary_trigger_exchanges: int = 20

    # How many past summaries to inject into context
    context_summary_count: int = 3


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LogConfig:
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: Path = field(default_factory=lambda: ROOT_DIR / "jarvis.log")


# ---------------------------------------------------------------------------
# Singleton instances imported by other modules
# ---------------------------------------------------------------------------

GROQ_CFG = GroqConfig()
MEMORY_CFG = MemoryConfig()
LOG_CFG = LogConfig()
