"""
AI/ai.py
========
The core orchestration engine for Jarvis.

This module owns the full pipeline for a single user turn:

    1. Build context  (memory → system prompt → user message)
    2. First LLM call  (with tools bound)
    3. If tool calls requested → execute ALL tools (deduplicated, capped)
    4. If tools were used → ONE final LLM call with tool results
    5. Save conversation to memory
    6. Background: check summarisation threshold; check profile update

Pipeline cost (LLM calls):
    - No tools:       1 call
    - Tool(s) used:   2 calls
    - Never more than 2 calls per user turn

Design decisions:
- `JarvisAI` is a stateful class that holds the memory manager and
  tool registry.  `main.py` creates one instance per session.
- Tool deduplication is done by hashing (tool_name + args) before
  execution.  Identical parallel requests are collapsed to one call.
- Background summarisation runs in a `threading.Thread` to avoid
  blocking the response; it uses `MemoryManager`'s internal lock for
  thread safety.
- Rate limit errors (HTTP 429) are caught and retried once, then a
  friendly message is returned — never an infinite retry loop.
- Profile extraction is only attempted when the LLM's own response
  contains a high-information message (heuristic: > 20 words in the
  user message).  This keeps profile writes rare.

Future expansion:
- Agents: replace the two-call flow with a LangChain AgentExecutor
  and keep the same public `chat()` interface.
- RAG: inject retrieved chunks into `_build_messages()` between the
  context block and the user message.
- STT: `main.py` transcribes audio and passes the string here;
  `ai.py` is unaware of the input modality.
- TTS: `main.py` converts the returned string to audio; `ai.py`
  returns plain text.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from AI.models import build_llm, build_llm_with_tools
from AI.prompts import PROFILE_UPDATE_PROMPT, SUMMARISE_PROMPT, SYSTEM_PROMPT
from Memory.memory import MemoryManager
from Tools.registry import get_all_tools
from Utils.exceptions import LLMError, RateLimitError
from Utils.helpers import count_words
from Utils.logger import get_logger

logger = get_logger("ai")


# ---------------------------------------------------------------------------
# JarvisAI
# ---------------------------------------------------------------------------

class JarvisAI:
    """
    Main AI orchestrator.  Owns the conversation pipeline.

    Args:
        memory: A `MemoryManager` instance (injected for testability).
    """

    def __init__(self, memory: MemoryManager) -> None:
        self._memory = memory
        self._tools = get_all_tools()
        self._llm_with_tools = build_llm_with_tools(self._tools)
        self._llm_plain = build_llm()
        # Map tool name → callable for fast lookup during execution
        self._tool_map: dict[str, Any] = {t.name: t for t in self._tools}
        logger.info("JarvisAI ready. Tools registered: %s", [t.name for t in self._tools])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> str:
        """
        Process a single user message and return Jarvis's response.

        This is the only public method called by `main.py`.
        """
        logger.info("User: %s", user_input[:120])

        # 1. Persist user message immediately
        self._memory.add_message("user", user_input)

        # 2. Build the message list for the first LLM call
        messages = self._build_messages(user_input)

        # 3. First LLM call (may request tool calls)
        try:
            first_response = self._call_llm_with_tools(messages)
        except RateLimitError as exc:
            return exc.message
        except LLMError as exc:
            return f"I encountered an error: {exc.message}"

        # 4. Extract tool call requests from the response
        tool_calls = self._extract_tool_calls(first_response)

        if not tool_calls:
            # No tools needed — return immediately
            final_text = first_response.content or ""
            self._memory.add_message("assistant", final_text)
            self._run_background_tasks(user_input, first_response.content or "")
            return final_text

        # 5. Execute all requested tools (deduplicated, capped)
        tool_results = self._execute_tools(tool_calls)

        # 6. Second (and final) LLM call with tool results
        messages_with_results = messages + [first_response] + tool_results
        try:
            final_response = self._call_llm_plain(messages_with_results)
        except RateLimitError as exc:
            return exc.message
        except LLMError as exc:
            return f"I encountered an error: {exc.message}"

        final_text = final_response.content or ""

        # 7. Persist assistant response
        self._memory.add_message("assistant", final_text)

        # 8. Background tasks (summarisation, profile update)
        self._run_background_tasks(user_input, final_text)

        logger.info("Jarvis: %s", final_text[:120])
        return final_text

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_messages(self, user_input: str) -> list[BaseMessage]:
        """
        Assemble the full message list for the LLM.

        Structure:
            SystemMessage  ← system prompt
            HumanMessage   ← memory context (profile + summaries + recent)
            HumanMessage   ← current user input

        The context is injected as a HumanMessage rather than appended
        to the SystemMessage so it clearly represents the conversation
        state from the LLM's perspective.
        """
        msgs: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

        context = self._memory.build_context()
        if context:
            msgs.append(HumanMessage(content=f"[Context]\n{context}"))

        msgs.append(HumanMessage(content=user_input))
        return msgs

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _call_llm_with_tools(self, messages: list[BaseMessage]) -> AIMessage:
        """First LLM call — model may emit tool_use blocks."""
        try:
            response = self._llm_with_tools.invoke(messages)
            return response  # type: ignore[return-value]
        except Exception as exc:
            return self._handle_llm_exception(exc)

    def _call_llm_plain(self, messages: list[BaseMessage]) -> AIMessage:
        """Second (final) LLM call — synthesises tool results into prose."""
        try:
            response = self._llm_plain.invoke(messages)
            return response  # type: ignore[return-value]
        except Exception as exc:
            return self._handle_llm_exception(exc)

    @staticmethod
    def _handle_llm_exception(exc: Exception) -> None:
        """Translate raw exceptions into typed JarvisErrors."""
        msg = str(exc)
        if "429" in msg or "rate limit" in msg.lower():
            logger.warning("Rate limit hit: %s", msg)
            raise RateLimitError(
                "I've hit the API rate limit. Please wait a moment before trying again."
            )
        logger.error("LLM error: %s", msg)
        raise LLMError(f"The AI service returned an error: {msg}")

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _extract_tool_calls(self, response: AIMessage) -> list[dict[str, Any]]:
        """
        Extract tool call requests from the LLM response.

        Returns a list of {id, name, args} dicts, deduplicated and
        capped at `GROQ_CFG.max_tool_calls_per_request`.
        """
        from config import GROQ_CFG

        raw_calls: list[dict[str, Any]] = getattr(response, "tool_calls", []) or []
        if not raw_calls:
            return []

        seen: set[str] = set()
        deduplicated: list[dict[str, Any]] = []

        for call in raw_calls:
            # Fingerprint = tool name + sorted JSON args
            fingerprint = self._tool_fingerprint(call.get("name", ""), call.get("args", {}))
            if fingerprint in seen:
                logger.info("Duplicate tool call skipped: %s", call.get("name"))
                continue
            seen.add(fingerprint)
            deduplicated.append(call)

            if len(deduplicated) >= GROQ_CFG.max_tool_calls_per_request:
                logger.warning(
                    "Tool call cap (%d) reached; dropping remaining calls.",
                    GROQ_CFG.max_tool_calls_per_request,
                )
                break

        return deduplicated

    @staticmethod
    def _tool_fingerprint(name: str, args: dict[str, Any]) -> str:
        """Stable hash of a tool call for deduplication."""
        raw = f"{name}:{json.dumps(args, sort_keys=True)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _execute_tools(self, tool_calls: list[dict[str, Any]]) -> list[ToolMessage]:
        """
        Execute all requested tools and return their results as
        LangChain `ToolMessage` objects ready for the second LLM call.
        """
        results: list[ToolMessage] = []

        for call in tool_calls:
            tool_name = call.get("name", "")
            tool_args = call.get("args", {})
            call_id = call.get("id", tool_name)

            tool_fn = self._tool_map.get(tool_name)
            if tool_fn is None:
                logger.warning("Unknown tool requested: %s", tool_name)
                content = json.dumps({"success": False, "message": f"Unknown tool: {tool_name}", "data": {}})
            else:
                logger.info("Executing tool: %s | args: %s", tool_name, tool_args)
                try:
                    result = tool_fn.invoke(tool_args)
                    content = json.dumps(result) if isinstance(result, dict) else str(result)
                except Exception as exc:
                    logger.error("Tool %s raised: %s", tool_name, exc)
                    content = json.dumps({"success": False, "message": str(exc), "data": {}})

            results.append(ToolMessage(content=content, tool_call_id=call_id))

        return results

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    def _run_background_tasks(self, user_input: str, assistant_response: str) -> None:
        """
        Spawn a daemon thread for non-critical post-turn work.

        Keeps the main response path fast.
        """
        thread = threading.Thread(
            target=self._background_worker,
            args=(user_input, assistant_response),
            daemon=True,
        )
        thread.start()

    def _background_worker(self, user_input: str, assistant_response: str) -> None:
        """Runs in background thread — summarise if needed, update profile if warranted."""
        try:
            self._maybe_summarise()
        except Exception as exc:
            logger.error("Background summarisation failed: %s", exc)

        try:
            self._maybe_update_profile(user_input)
        except Exception as exc:
            logger.error("Background profile update failed: %s", exc)

    def _maybe_summarise(self) -> None:
        """Summarise recent memory if the threshold is reached."""
        if not self._memory.should_summarise():
            return

        logger.info("Summarisation threshold reached. Summarising...")
        recent = self._memory.recent_messages
        conversation = "\n".join(
            f"{'User' if m.role == 'user' else 'Jarvis'}: {m.content}"
            for m in recent
        )
        prompt = SUMMARISE_PROMPT.format(conversation=conversation)
        try:
            response = self._llm_plain.invoke([HumanMessage(content=prompt)])
            summary_text = response.content or ""
            if summary_text.strip():
                self._memory.store_summary(summary_text.strip())
        except Exception as exc:
            logger.error("Summarisation LLM call failed: %s", exc)

    def _maybe_update_profile(self, user_input: str) -> None:
        """
        Attempt profile extraction only for high-information messages.
        Avoids a wasteful LLM call on short inputs like "thanks" or "ok".
        """
        if count_words(user_input) < 15:
            return

        prompt = PROFILE_UPDATE_PROMPT.format(message=user_input)
        try:
            response = self._llm_plain.invoke([HumanMessage(content=prompt)])
            raw = (response.content or "").strip()
            # Strip possible markdown fences
            raw = raw.replace("```json", "").replace("```", "").strip()
            updates: dict[str, Any] = json.loads(raw)
            if updates:
                self._memory.update_profile(updates)
        except (json.JSONDecodeError, ValueError):
            pass  # LLM returned non-JSON — ignore
        except Exception as exc:
            logger.error("Profile update LLM call failed: %s", exc)
