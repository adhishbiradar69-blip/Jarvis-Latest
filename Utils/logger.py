"""
Utils/logger.py
===============
Centralised logging setup for Jarvis.

Design decisions:
- A single `get_logger(name)` factory ensures consistent formatting
  across all modules without each file fiddling with handler setup.
- The root "jarvis" logger is configured once; child loggers
  (e.g. "jarvis.memory", "jarvis.ai") inherit its handlers.
- Secrets must never be logged — callers are responsible for not
  passing sensitive values, but we add a reminder in the docstring.
- Structured enough for production (timestamp, level, module name)
  while remaining readable in a terminal during development.

Future expansion:
- Replace StreamHandler with a rotating FileHandler or ship logs to
  an observability platform (Datadog, Sentry) by swapping the handler
  here without touching any other file.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache

from config import LOG_CFG


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_root_logger() -> None:
    """Set up the root 'jarvis' logger exactly once."""
    root = logging.getLogger("jarvis")
    if root.handlers:
        # Already configured (e.g. during tests that re-import)
        return

    root.setLevel(LOG_CFG.log_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Always log to stderr so stdout stays clean for actual output
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if LOG_CFG.log_to_file:
        file_handler = logging.FileHandler(LOG_CFG.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


_configure_root_logger()


@lru_cache(maxsize=None)
def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'jarvis' namespace.

    Usage:
        from Utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Memory saved.")

    IMPORTANT: Never log secrets, API keys, or raw user credentials.
    """
    return logging.getLogger(f"jarvis.{name}")
