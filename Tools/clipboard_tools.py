"""
Tools/clipboard_tools.py
========================
Clipboard read/write tools using `pyperclip`.

Design decisions:
- pyperclip is optional — if absent or the platform has no clipboard
  (e.g. a headless CI server), tools return a graceful error instead
  of crashing at import time.
- Content is capped at 4 KB when reading to avoid dumping a massive
  clipboard into the LLM context.
"""

from __future__ import annotations

from langchain_core.tools import tool

from Utils.helpers import tool_result
from Utils.logger import get_logger

logger = get_logger("tools.clipboard")

_READ_CAP = 4_096  # characters

try:
    import pyperclip
    _PYPERCLIP_AVAILABLE = True
except ImportError:
    _PYPERCLIP_AVAILABLE = False


def _require_pyperclip() -> dict | None:
    if not _PYPERCLIP_AVAILABLE:
        return tool_result(
            success=False,
            message="pyperclip is not installed. Run: pip install pyperclip"
        )
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def read_clipboard() -> dict:
    """
    Read the current contents of the system clipboard.

    Use this when the user asks what is on their clipboard, or wants to
    process text they have copied.

    Returns:
        Standardised result dict with data["content"].
    """
    if err := _require_pyperclip():
        return err
    try:
        content = pyperclip.paste()
        if not content:
            return tool_result(success=True, message="Clipboard is empty.", data={"content": ""})

        truncated = content[:_READ_CAP]
        was_truncated = len(content) > _READ_CAP
        msg = f"Read {len(content)} chars from clipboard."
        if was_truncated:
            msg += f" (truncated to {_READ_CAP} chars)"

        logger.info("read_clipboard: %d chars", len(content))
        return tool_result(success=True, message=msg, data={"content": truncated})

    except Exception as exc:
        logger.error("read_clipboard error: %s", exc)
        return tool_result(success=False, message=str(exc))


@tool
def write_clipboard(content: str) -> dict:
    """
    Write text to the system clipboard.

    Use this when the user asks to copy something to their clipboard.

    Args:
        content: The text to place on the clipboard.

    Returns:
        Standardised result dict.
    """
    if err := _require_pyperclip():
        return err
    try:
        pyperclip.copy(content)
        logger.info("write_clipboard: %d chars written", len(content))
        return tool_result(
            success=True,
            message=f"Copied {len(content)} characters to clipboard.",
        )
    except Exception as exc:
        logger.error("write_clipboard error: %s", exc)
        return tool_result(success=False, message=str(exc))


# Exported list consumed by registry.py
CLIPBOARD_TOOLS = [read_clipboard, write_clipboard]
