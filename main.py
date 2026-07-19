"""
main.py
=======
Entry point for Jarvis.

Responsibilities:
- Bootstrap: validate environment, initialise all subsystems.
- Run the REPL (Read-Eval-Print Loop) for interactive use.
- Provide clean shutdown on Ctrl-C.

Design decisions:
- `main.py` is intentionally thin.  It wires together the subsystems
  (MemoryManager → JarvisAI) but contains no business logic itself.
- Error handling at this level is coarse: catch-all prints a friendly
  message rather than a stack trace, keeping the UX clean.
- The REPL is synchronous and single-threaded; background tasks inside
  `JarvisAI` use daemon threads so they don't block the loop.

Future expansion:
- Replace the REPL with a GUI, a FastAPI HTTP server, or a WebSocket
  server by swapping out the `_run_repl()` function.  `JarvisAI.chat()`
  remains the same regardless of the input channel.
- STT: transcribe audio to a string and pass it to `ai.chat()`.
- TTS: pass the returned string to a TTS engine before printing it.
"""

from __future__ import annotations

import sys

from AI.ai import JarvisAI
from Memory.memory import MemoryManager
from Utils.exceptions import ConfigurationError
from Utils.logger import get_logger

logger = get_logger("main")


# ---------------------------------------------------------------------------
# Boot sequence
# ---------------------------------------------------------------------------

def _bootstrap() -> JarvisAI:
    """Initialise and wire together all subsystems."""
    try:
        from config import GROQ_CFG  # Triggers .env load and validation
        _ = GROQ_CFG.api_key        # Will raise KeyError if missing
    except KeyError:
        raise ConfigurationError(
            "GROQ_API_KEY is not set.  Add it to your .env file."
        )

    memory = MemoryManager()
    ai = JarvisAI(memory=memory)
    return ai


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def _run_repl(ai: JarvisAI) -> None:
    """Interactive command-line loop."""
    print("\n╔══════════════════════════════╗")
    print("║        Jarvis  v0.1.0        ║")
    print("║  Type 'exit' or Ctrl-C quit  ║")
    print("╚══════════════════════════════╝\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nJarvis: Goodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Jarvis: Goodbye.")
            break

        response = ai.chat(user_input)
        print(f"\nJarvis: {response}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        ai = _bootstrap()
        logger.info("Jarvis started.")
        _run_repl(ai)
    except ConfigurationError as exc:
        print(f"[Configuration Error] {exc.message}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected startup error: %s", exc)
        print(f"[Fatal Error] {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        logger.info("Jarvis stopped.")


if __name__ == "__main__":
    main()
