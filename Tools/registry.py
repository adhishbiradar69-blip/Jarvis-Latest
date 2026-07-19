"""
Tools/registry.py
=================
Central registry that collects every LangChain tool and exposes them
to the AI layer as a single list.

Design decisions:
- The registry is the ONLY place that knows which tools exist.
  AI/ imports `get_all_tools()` and nothing else — decoupled.
- Adding a new tool module (e.g. web_search_tools.py) requires only
  two lines here: import the list and extend `_ALL_TOOLS`.
- Each tool module is responsible for its own error handling; the
  registry does not wrap them.
- Deduplication happens in AI/ (the executor), not here.

Future expansion:
- Conditional registration (e.g. only register Spotify tools if the
  env var `SPOTIFY_CLIENT_ID` is set) can be added with a simple
  `if` guard per module.
- Agent-specific tool subsets can be returned by adding a
  `get_tools_for_agent(agent_name)` function.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from Tools.file_tools import FILE_TOOLS
from Tools.system_tools import SYSTEM_TOOLS
from Tools.clipboard_tools import CLIPBOARD_TOOLS
from Tools.app_tools import APP_TOOLS

# Master list — order does not matter for the LLM
_ALL_TOOLS: list[BaseTool] = [
    *FILE_TOOLS,
    *SYSTEM_TOOLS,
    *CLIPBOARD_TOOLS,
    *APP_TOOLS,
]


def get_all_tools() -> list[BaseTool]:
    """Return every registered tool.  Called once by the AI layer."""
    return list(_ALL_TOOLS)


def get_tool_names() -> list[str]:
    """Return tool names for logging and deduplication."""
    return [t.name for t in _ALL_TOOLS]
