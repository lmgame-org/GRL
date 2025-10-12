from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import List, Tuple


# Workspace root resolution for GRL (supports per-thread override)
_TLS = threading.local()


def set_thread_workspace_root(path: str | Path) -> None:
  try:
    _TLS.workspace_root = Path(path).resolve()
  except Exception:
    _TLS.workspace_root = None


def clear_thread_workspace_root() -> None:
  if hasattr(_TLS, "workspace_root"):
    try:
      delattr(_TLS, "workspace_root")
    except Exception:
      pass


def get_workspace_root() -> Path:
  # 1) Thread-local override if set
  root = getattr(_TLS, "workspace_root", None)
  if isinstance(root, Path):
    return root
  # 2) Environment variable (can be set per agent, but shared across threads)
  env_val = os.environ.get("GRL_WORKSPACE_ROOT")
  if env_val:
    try:
      return Path(env_val).resolve()
    except Exception:
      pass
  # 3) Fallback to repository root (two levels up from tools/)
  return Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))).resolve()


def safe_run_shell(cmd: str, timeout: int = 120) -> Tuple[str, str]:
  if not isinstance(cmd, str) or not cmd.strip():
    return "Empty command.", "Error: Exit code 2"
  try:
    workspace_root = get_workspace_root()
    cmd = f'cd "{workspace_root}" && {cmd}'
    proc = subprocess.run(
        ["/bin/bash", "-lc", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
      return output, f"Error: Exit code {proc.returncode}"
    return output, str(proc.returncode)
  except subprocess.TimeoutExpired:
    return f"The command took too long to execute (>{timeout}s)", "-1"
  except Exception as e:
    return f"Error: {repr(e)}", "-1"


def list_non_hidden_files(
    directory: Path, max_depth: int = 2, python_only: bool = True
) -> List[Path]:
  results: List[Path] = []
  base_depth = len(directory.resolve().parts)
  for root, dirs, files in os.walk(directory):
    dirs[:] = [d for d in dirs if not d.startswith(".")]
    depth = len(Path(root).resolve().parts) - base_depth
    if depth > max_depth:
      dirs[:] = []
      continue
    for f in files:
      if f.startswith("."):
        continue
      p = Path(root) / f
      if not python_only or p.suffix == ".py":
        results.append(p)
  return results
