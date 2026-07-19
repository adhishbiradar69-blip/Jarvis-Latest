"""
Utils/helpers.py
================
Small, pure utility functions shared across the codebase.

Design decisions:
- No imports from other Jarvis modules — this layer sits at the
  bottom of the dependency graph to keep the import tree acyclic.
- Functions are intentionally tiny and stateless so they can be
  tested in isolation without any mocking.
- `safe_json_*` wrappers centralise the pattern of reading/writing
  JSON files with sensible defaults, rather than scattering
  try/except blocks across Memory files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def safe_read_json(path: Path, default: Any = None) -> Any:
    """
    Read a JSON file and return its contents.

    Returns `default` if the file does not exist or is malformed,
    rather than raising — callers should handle missing data gracefully.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def safe_write_json(path: Path, data: Any, *, indent: int = 2) -> bool:
    """
    Write `data` to a JSON file atomically-ish (write then rename).

    Returns True on success, False on failure.
    Uses a temp file so a crash mid-write never corrupts the real file.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def truncate(text: str, max_chars: int = 200, suffix: str = "…") -> str:
    """Shorten `text` to `max_chars` characters for display / logging."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def count_words(text: str) -> int:
    """Return the number of whitespace-separated words in `text`."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Dict helpers
# ---------------------------------------------------------------------------

def tool_result(*, success: bool, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Standardised return structure for all tool functions.

    Every tool returns this shape so the AI layer can parse results
    uniformly without special-casing individual tools.
    """
    return {
        "success": success,
        "message": message,
        "data": data or {},
    }
