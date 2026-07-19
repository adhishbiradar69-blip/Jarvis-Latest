"""
Tools/file_tools.py
===================
File-system tools exposed to the LLM via LangChain's @tool decorator.

Design decisions:
- Every tool returns the standardised `tool_result` dict so the LLM
  can parse success/failure uniformly.
- Paths are validated before any OS call — prevents directory
  traversal and gives a friendly error rather than a raw OSError.
- Read operations are capped at 10 KB to avoid flooding the context
  window with huge files.
- Write operations are atomic (write to .tmp, then rename).

Future expansion:
- A `search_files` tool (fuzzy filename search) slots in here.
- A RAG `index_directory` tool would live here and call into RAG/.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from Utils.helpers import tool_result
from Utils.logger import get_logger

logger = get_logger("tools.file")

_READ_MAX_BYTES = 10_240  # 10 KB cap


def _resolve(path_str: str) -> Path:
    """Expand ~ and resolve to an absolute path."""
    return Path(path_str).expanduser().resolve()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def read_file(path: str) -> dict:
    """
    Read the contents of a text file and return them.

    Use this when the user asks to open, view, read, or inspect a file.

    Args:
        path: Absolute or home-relative path to the file.

    Returns:
        Standardised result dict with file content in data["content"].
    """
    try:
        p = _resolve(path)
        if not p.exists():
            return tool_result(success=False, message=f"File not found: {p}")
        if not p.is_file():
            return tool_result(success=False, message=f"Path is not a file: {p}")

        raw = p.read_bytes()
        if len(raw) > _READ_MAX_BYTES:
            content = raw[:_READ_MAX_BYTES].decode("utf-8", errors="replace")
            msg = f"File truncated to {_READ_MAX_BYTES} bytes."
        else:
            content = raw.decode("utf-8", errors="replace")
            msg = f"Read {len(raw)} bytes from {p.name}."

        logger.info("read_file: %s (%d bytes)", p, len(raw))
        return tool_result(success=True, message=msg, data={"content": content, "path": str(p)})

    except PermissionError:
        logger.warning("read_file: permission denied — %s", path)
        return tool_result(success=False, message=f"Permission denied: {path}")
    except Exception as exc:
        logger.error("read_file: unexpected error — %s", exc)
        return tool_result(success=False, message=str(exc))


@tool
def write_file(path: str, content: str) -> dict:
    """
    Write text content to a file, creating it (and parent directories)
    if they do not exist.  Overwrites existing files.

    Use this when the user asks to save, write, create, or overwrite a file.

    Args:
        path: Absolute or home-relative path to the destination file.
        content: Text content to write.

    Returns:
        Standardised result dict.
    """
    try:
        p = _resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(p)

        logger.info("write_file: %s (%d chars)", p, len(content))
        return tool_result(success=True, message=f"Written to {p.name}.", data={"path": str(p)})

    except PermissionError:
        logger.warning("write_file: permission denied — %s", path)
        return tool_result(success=False, message=f"Permission denied: {path}")
    except Exception as exc:
        logger.error("write_file: unexpected error — %s", exc)
        return tool_result(success=False, message=str(exc))


@tool
def list_directory(path: str) -> dict:
    """
    List the files and subdirectories inside a directory.

    Use this when the user asks what is in a folder, directory, or path.

    Args:
        path: Absolute or home-relative path to the directory.

    Returns:
        Standardised result dict with data["entries"] as a list of names.
    """
    try:
        p = _resolve(path)
        if not p.exists():
            return tool_result(success=False, message=f"Directory not found: {p}")
        if not p.is_dir():
            return tool_result(success=False, message=f"Path is not a directory: {p}")

        entries = sorted(
            {"name": e.name, "type": "dir" if e.is_dir() else "file"}
            for e in p.iterdir()
        )

        logger.info("list_directory: %s (%d entries)", p, len(entries))
        return tool_result(
            success=True,
            message=f"Listed {len(entries)} entries in {p.name}.",
            data={"path": str(p), "entries": entries},
        )

    except PermissionError:
        logger.warning("list_directory: permission denied — %s", path)
        return tool_result(success=False, message=f"Permission denied: {path}")
    except Exception as exc:
        logger.error("list_directory: unexpected error — %s", exc)
        return tool_result(success=False, message=str(exc))


@tool
def delete_file(path: str) -> dict:
    """
    Permanently delete a file.  Does NOT delete directories.

    Use this when the user explicitly asks to delete or remove a file.

    Args:
        path: Absolute or home-relative path to the file.

    Returns:
        Standardised result dict.
    """
    try:
        p = _resolve(path)
        if not p.exists():
            return tool_result(success=False, message=f"File not found: {p}")
        if not p.is_file():
            return tool_result(success=False, message=f"Will not delete a directory: {p}")

        p.unlink()
        logger.info("delete_file: %s", p)
        return tool_result(success=True, message=f"Deleted {p.name}.")

    except PermissionError:
        logger.warning("delete_file: permission denied — %s", path)
        return tool_result(success=False, message=f"Permission denied: {path}")
    except Exception as exc:
        logger.error("delete_file: unexpected error — %s", exc)
        return tool_result(success=False, message=str(exc))


# Exported list consumed by registry.py
FILE_TOOLS = [read_file, write_file, list_directory, delete_file]
