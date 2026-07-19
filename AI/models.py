"""
AI/models.py
============
Constructs and exposes LangChain-compatible LLM instances.

Design decisions:
- The factory function `build_llm()` is the single place where the
  ChatGroq object is created.  `ai.py` calls it; nothing else does.
- `build_llm_with_tools(tools)` binds the tool list to the LLM using
  LangChain's native `.bind_tools()` — the LLM itself decides when
  and how to call them.
- Both functions are cached with `@lru_cache` so the expensive
  LangChain object is only constructed once per process.
- The model name and parameters come from `config.py`; swapping to
  a different Groq model (or later to Ollama) requires only a
  config change.

Future expansion:
- Add `build_ollama_llm()` here when Ollama integration is added.
  `ai.py` can select which factory to call based on a config flag.
- Vision models (e.g. llava) would be returned from a separate
  `build_vision_llm()` function.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq

from config import GROQ_CFG
from Utils.logger import get_logger

logger = get_logger("models")


@lru_cache(maxsize=1)
def build_llm() -> ChatGroq:
    """
    Build and return a plain (no-tools) ChatGroq LLM instance.

    Used for utility calls that do not need tool access — specifically
    the summarisation and profile-extraction calls.
    """
    logger.info("Building LLM: model=%s temp=%.2f", GROQ_CFG.model, GROQ_CFG.temperature)
    return ChatGroq(
        api_key=GROQ_CFG.api_key,
        model=GROQ_CFG.model,
        temperature=GROQ_CFG.temperature,
        max_tokens=GROQ_CFG.max_tokens,
    )


def build_llm_with_tools(tools: Sequence[BaseTool]) -> ChatGroq:
    """
    Return a ChatGroq instance with the given tools bound to it.

    LangChain's `.bind_tools()` serialises the tool schemas and
    appends them to the system message so the LLM can request calls.

    Not cached because the tool list could theoretically vary between
    calls (e.g. per-agent subsets in future).
    """
    llm = build_llm()
    return llm.bind_tools(tools)  # type: ignore[return-value]
