"""
Memory/memory.py
================
Three-layer persistent memory system for Jarvis.

Layers
------
1. Profile   — Long-term, user-specific facts that rarely change.
               (name, projects, preferences, goals)
               Never cluttered with conversation.

2. Recent    — Rolling window of the last N raw messages.
               Written after *every* exchange so no data is lost
               if the process crashes.

3. Summaries — Compressed records of older conversations.
               Created when `recent` crosses a threshold; after
               summarising, recent is cleared to stay cheap.

Design decisions:
- All three layers use plain JSON files on disk.  No database
  dependency at this stage — easy to swap for SQLite or a vector
  store later by replacing `_load` / `_save` calls.
- `MemoryManager` is a single object; main.py creates one instance
  and passes it down.  No global state.
- `build_context()` returns the assembled string that the context
  builder injects into every LLM call.  Keeping it here (rather than
  in AI/) means memory and context stay co-located.
- Profile updates are intentionally gated behind an explicit call
  (`update_profile`) rather than happening automatically, to avoid
  noisy writes every turn.

Future expansion:
- `build_context()` will grow a `rag_chunks` parameter once RAG is
  implemented — the rest of the pipeline passes straight through.
- Summaries could be embedded and stored in a vector DB without
  touching the public interface.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import MEMORY_CFG
from Utils.exceptions import MemoryError as JarvisMemoryError
from Utils.helpers import safe_read_json, safe_write_json, truncate
from Utils.logger import get_logger

logger = get_logger("memory")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single conversation turn stored in recent memory."""
    role: str          # "user" | "assistant"
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "Message":
        return cls(role=d["role"], content=d["content"], timestamp=d.get("timestamp", ""))


@dataclass
class Summary:
    """A compressed record of an older conversation window."""
    content: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, str]:
        return {"content": self.content, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "Summary":
        return cls(content=d["content"], created_at=d.get("created_at", ""))


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Manages all three memory layers for a single Jarvis session.

    Thread safety: a `threading.Lock` guards all writes because the
    summarisation step runs in a background thread.
    """

    def __init__(self, cfg: type = MEMORY_CFG) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()

        # In-memory caches — loaded from disk on first access
        self._recent: list[Message] = []
        self._summaries: list[Summary] = []
        self._profile: dict[str, Any] = {}

        self._load_all()
        logger.info("MemoryManager initialised. Recent=%d, Summaries=%d",
                    len(self._recent), len(self._summaries))

    # ------------------------------------------------------------------
    # Public read interface
    # ------------------------------------------------------------------

    @property
    def profile(self) -> dict[str, Any]:
        return dict(self._profile)

    @property
    def recent_messages(self) -> list[Message]:
        return list(self._recent)

    @property
    def latest_summaries(self) -> list[Summary]:
        """Return the N most recent summaries for context injection."""
        n = self._cfg.context_summary_count
        return self._summaries[-n:]

    def build_context(self) -> str:
        """
        Assemble the full context string passed to the LLM on every call.

        Structure:
            [PROFILE]
            ...

            [SUMMARIES]
            ...

            [RECENT CONVERSATION]
            ...

        The caller (ContextBuilder in AI/) prepends the system prompt
        and appends the live user message.
        """
        parts: list[str] = []

        # 1. Profile
        if self._profile:
            profile_lines = "\n".join(f"- {k}: {v}" for k, v in self._profile.items())
            parts.append(f"[USER PROFILE]\n{profile_lines}")

        # 2. Summaries (most recent N)
        recent_summaries = self.latest_summaries
        if recent_summaries:
            summary_text = "\n\n".join(s.content for s in recent_summaries)
            parts.append(f"[CONVERSATION SUMMARIES]\n{summary_text}")

        # 3. Recent messages
        if self._recent:
            lines = []
            for msg in self._recent:
                prefix = "User" if msg.role == "user" else "Jarvis"
                lines.append(f"{prefix}: {msg.content}")
            parts.append(f"[RECENT CONVERSATION]\n" + "\n".join(lines))

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Public write interface
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """
        Append a message to recent memory and immediately persist it.

        Enforces the rolling window: if we exceed `recent_max_messages`,
        the oldest messages are dropped.
        """
        with self._lock:
            msg = Message(role=role, content=content)
            self._recent.append(msg)

            # Trim to window size
            if len(self._recent) > self._cfg.recent_max_messages:
                self._recent = self._recent[-self._cfg.recent_max_messages :]

            self._save_recent()
            logger.debug("Message added [%s]: %s", role, truncate(content, 80))

    def should_summarise(self) -> bool:
        """
        Return True when recent memory has accumulated enough user
        exchanges to warrant summarisation.
        """
        user_exchanges = sum(1 for m in self._recent if m.role == "user")
        return user_exchanges >= self._cfg.summary_trigger_exchanges

    def store_summary(self, summary_text: str) -> None:
        """
        Persist a new summary and clear recent memory.
        Called by AI/ after it has asked the LLM to summarise.
        """
        with self._lock:
            summary = Summary(content=summary_text)
            self._summaries.append(summary)
            self._recent = []

            self._save_summaries()
            self._save_recent()
            logger.info("Summary stored. Total summaries: %d", len(self._summaries))

    def update_profile(self, updates: dict[str, Any]) -> None:
        """
        Merge `updates` into the profile and persist.

        Designed to be called sparingly — only when long-term facts
        are detected (e.g. user mentions their preferred language).
        """
        with self._lock:
            self._profile.update(updates)
            self._save_profile()
            logger.info("Profile updated: %s", list(updates.keys()))

    # ------------------------------------------------------------------
    # Persistence (private)
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all three layers from disk at startup."""
        self._recent = self._load_recent()
        self._summaries = self._load_summaries()
        self._profile = self._load_profile()

    def _load_recent(self) -> list[Message]:
        raw = safe_read_json(self._cfg.recent_path, default=[])
        return [Message.from_dict(d) for d in raw if isinstance(d, dict)]

    def _load_summaries(self) -> list[Summary]:
        raw = safe_read_json(self._cfg.summaries_path, default=[])
        return [Summary.from_dict(d) for d in raw if isinstance(d, dict)]

    def _load_profile(self) -> dict[str, Any]:
        return safe_read_json(self._cfg.profile_path, default={})

    def _save_recent(self) -> None:
        ok = safe_write_json(self._cfg.recent_path, [m.to_dict() for m in self._recent])
        if not ok:
            raise JarvisMemoryError("Failed to persist recent memory.")

    def _save_summaries(self) -> None:
        ok = safe_write_json(self._cfg.summaries_path, [s.to_dict() for s in self._summaries])
        if not ok:
            raise JarvisMemoryError("Failed to persist summaries.")

    def _save_profile(self) -> None:
        ok = safe_write_json(self._cfg.profile_path, self._profile)
        if not ok:
            raise JarvisMemoryError("Failed to persist profile.")
