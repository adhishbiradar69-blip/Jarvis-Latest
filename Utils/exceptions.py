"""
Utils/exceptions.py
===================
Custom exception hierarchy for Jarvis.

Design decisions:
- A base JarvisError makes it easy to catch *any* Jarvis-specific
  error at the top level (main.py) without accidentally swallowing
  unrelated exceptions.
- Specific subclasses let callers handle different failure modes
  independently (e.g. show a different message for RateLimitError
  vs ToolExecutionError).
- All exceptions carry a human-readable `message` that can be
  surfaced directly to the user, keeping error handling in business
  logic simple.
"""

from __future__ import annotations


class JarvisError(Exception):
    """Base class for all Jarvis application errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ConfigurationError(JarvisError):
    """Raised when required config or environment variables are missing."""


class LLMError(JarvisError):
    """Raised when the LLM call itself fails (network, auth, etc.)."""


class RateLimitError(LLMError):
    """Raised when the API returns HTTP 429 Too Many Requests."""


class ToolExecutionError(JarvisError):
    """Raised when a tool fails during execution."""

    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {reason}")


class MemoryError(JarvisError):
    """Raised when reading or writing memory files fails."""


class ContextBuildError(JarvisError):
    """Raised when the context builder cannot assemble a valid prompt."""
