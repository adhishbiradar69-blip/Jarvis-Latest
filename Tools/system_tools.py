"""
Tools/system_tools.py
=====================
System information tools: CPU, memory, disk, and running processes.

Design decisions:
- Uses `psutil` for cross-platform compatibility.
- Values are rounded to two decimal places for readability in LLM output.
- No shell command execution — prevents arbitrary command injection.

Future expansion:
- A `run_shell_command` tool could be added with an explicit allowlist
  of safe commands and user confirmation prompt.
"""

from __future__ import annotations

from langchain_core.tools import tool

from Utils.helpers import tool_result
from Utils.logger import get_logger

logger = get_logger("tools.system")

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


def _require_psutil() -> dict | None:
    if not _PSUTIL_AVAILABLE:
        return tool_result(
            success=False,
            message="psutil is not installed. Run: pip install psutil"
        )
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def get_cpu_usage() -> dict:
    """
    Return the current CPU usage percentage across all cores.

    Use this when the user asks about CPU load, processor usage, or system
    performance.

    Returns:
        Standardised result dict with data["cpu_percent"].
    """
    if err := _require_psutil():
        return err
    try:
        cpu = psutil.cpu_percent(interval=0.5)
        logger.info("get_cpu_usage: %.1f%%", cpu)
        return tool_result(
            success=True,
            message=f"CPU usage is {cpu:.1f}%.",
            data={"cpu_percent": cpu, "core_count": psutil.cpu_count()},
        )
    except Exception as exc:
        logger.error("get_cpu_usage error: %s", exc)
        return tool_result(success=False, message=str(exc))


@tool
def get_memory_usage() -> dict:
    """
    Return RAM usage statistics.

    Use this when the user asks about memory, RAM, or how much memory is free.

    Returns:
        Standardised result dict with total, used, free (all in MB), and
        percent used.
    """
    if err := _require_psutil():
        return err
    try:
        vm = psutil.virtual_memory()
        data = {
            "total_mb": round(vm.total / 1_048_576, 2),
            "used_mb": round(vm.used / 1_048_576, 2),
            "available_mb": round(vm.available / 1_048_576, 2),
            "percent": vm.percent,
        }
        logger.info("get_memory_usage: %.1f%% used", vm.percent)
        return tool_result(
            success=True,
            message=f"RAM: {data['used_mb']} MB used / {data['total_mb']} MB total ({vm.percent}% full).",
            data=data,
        )
    except Exception as exc:
        logger.error("get_memory_usage error: %s", exc)
        return tool_result(success=False, message=str(exc))


@tool
def get_disk_usage(path: str = "/") -> dict:
    """
    Return disk usage statistics for the given mount point or path.

    Use this when the user asks about disk space, storage, or how full a
    drive is.

    Args:
        path: Mount point or directory to check (default: root "/").

    Returns:
        Standardised result dict with total, used, free (all in GB), and
        percent used.
    """
    if err := _require_psutil():
        return err
    try:
        du = psutil.disk_usage(path)
        data = {
            "path": path,
            "total_gb": round(du.total / 1_073_741_824, 2),
            "used_gb": round(du.used / 1_073_741_824, 2),
            "free_gb": round(du.free / 1_073_741_824, 2),
            "percent": du.percent,
        }
        logger.info("get_disk_usage[%s]: %.1f%% used", path, du.percent)
        return tool_result(
            success=True,
            message=f"Disk ({path}): {data['used_gb']} GB used / {data['total_gb']} GB ({du.percent}% full).",
            data=data,
        )
    except FileNotFoundError:
        return tool_result(success=False, message=f"Path not found: {path}")
    except Exception as exc:
        logger.error("get_disk_usage error: %s", exc)
        return tool_result(success=False, message=str(exc))


@tool
def list_processes(limit: int = 10) -> dict:
    """
    Return the top N processes sorted by CPU usage.

    Use this when the user asks what processes are running, which app
    is using the most CPU, or for a process list.

    Args:
        limit: Maximum number of processes to return (default: 10).

    Returns:
        Standardised result dict with data["processes"] list.
    """
    if err := _require_psutil():
        return err
    try:
        procs = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                procs.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        procs.sort(key=lambda p: p.get("cpu_percent", 0) or 0, reverse=True)
        top = procs[: max(1, limit)]

        logger.info("list_processes: returned %d processes", len(top))
        return tool_result(
            success=True,
            message=f"Top {len(top)} processes by CPU usage.",
            data={"processes": top},
        )
    except Exception as exc:
        logger.error("list_processes error: %s", exc)
        return tool_result(success=False, message=str(exc))


# Exported list consumed by registry.py
SYSTEM_TOOLS = [get_cpu_usage, get_memory_usage, get_disk_usage, list_processes]
