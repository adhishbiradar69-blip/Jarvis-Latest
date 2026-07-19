"""
Tools/app_tools.py
==================
Application-level tools: opening URLs, launching apps, getting
the current time/date.

Design decisions:
- `open_url` delegates to the OS default browser via `webbrowser`
  (stdlib) — no external dependency required.
- `launch_app` uses `subprocess.Popen` with a controlled argument
  list (no shell=True) to prevent command injection.
- `get_current_time` is a cheap, no-network call that prevents the
  LLM from hallucinating the current date/time.

Future expansion:
- Spotify / media control tools go here (or in their own module).
- A `send_notification` tool fits naturally in this file.
"""

from __future__ import annotations

import subprocess
import webbrowser
from datetime import datetime, timezone

from langchain_core.tools import tool

from Utils.helpers import tool_result
from Utils.logger import get_logger

logger = get_logger("tools.app")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def get_current_time() -> dict:
    """
    Return the current local date and time.

    Use this whenever the user asks what time or date it is, or when
    you need the current timestamp to reason about schedules.

    Returns:
        Standardised result dict with data["datetime_local"] and
        data["datetime_utc"].
    """
    now_local = datetime.now()
    now_utc = datetime.now(timezone.utc)
    data = {
        "datetime_local": now_local.strftime("%Y-%m-%d %H:%M:%S"),
        "datetime_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "weekday": now_local.strftime("%A"),
    }
    logger.info("get_current_time: %s", data["datetime_local"])
    return tool_result(
        success=True,
        message=f"Current time: {data['datetime_local']} ({data['weekday']}).",
        data=data,
    )


@tool
def open_url(url: str) -> dict:
    """
    Open a URL in the system's default web browser.

    Use this when the user asks to open a website, URL, or link.

    Args:
        url: A fully-qualified URL (must start with http:// or https://).

    Returns:
        Standardised result dict.
    """
    if not url.startswith(("http://", "https://")):
        return tool_result(success=False, message=f"Invalid URL (must start with http/https): {url}")
    try:
        webbrowser.open(url)
        logger.info("open_url: %s", url)
        return tool_result(success=True, message=f"Opened {url} in the default browser.")
    except Exception as exc:
        logger.error("open_url error: %s", exc)
        return tool_result(success=False, message=str(exc))


@tool
def launch_app(app_name: str) -> dict:
    """
    Launch an application by its executable name.

    Use this when the user asks to open or start an application.
    The app_name must be the name of an executable available on the
    system PATH (e.g. "code", "firefox", "notepad").

    Args:
        app_name: Executable name (no shell metacharacters).

    Returns:
        Standardised result dict.
    """
    # Basic sanity check — no shell chars allowed
    forbidden = set(";|&><`$(){}[]\\\"'")
    if any(ch in forbidden for ch in app_name):
        return tool_result(
            success=False,
            message=f"Invalid app name '{app_name}': shell metacharacters are not allowed.",
        )
    try:
        subprocess.Popen([app_name])  # Non-blocking; does not capture stdout
        logger.info("launch_app: %s", app_name)
        return tool_result(success=True, message=f"Launched '{app_name}'.")
    except FileNotFoundError:
        logger.warning("launch_app: not found — %s", app_name)
        return tool_result(success=False, message=f"Application '{app_name}' not found on PATH.")
    except Exception as exc:
        logger.error("launch_app error: %s", exc)
        return tool_result(success=False, message=str(exc))


# Exported list consumed by registry.py
APP_TOOLS = [get_current_time, open_url, launch_app]
