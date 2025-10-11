from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base_tool import ToolGroup, tool
from .tool_utils import list_non_hidden_files


class SearchTools(ToolGroup):

  def __init__(self):
    super().__init__(name="search")

  @tool(
      schema={
          "type": "object",
          "properties": {
              "search_term": {
                  "type": "string",
                  "description": "Term to search for.",
              },
              "path": {
                  "type": "string",
                  "description": "File or directory to search (default .).",
              },
              "python_only": {
                  "type": "boolean",
                  "description": "If directory, only .py files (default true).",
              },
              "max_files": {
                  "type": "integer",
                  "description": "Max files to report for directories (default 100).",
              },
          },
          "required": ["search_term"],
      },
      description="Search for a term in a file or directory. Directory mode summarizes matches per file; file mode shows line numbers.",
  )
  def search(self, args: Dict[str, Any]) -> Dict[str, Any]:
    term: str = args.get("search_term", "")
    path_str: str = args.get("path", ".")
    python_only: bool = bool(args.get("python_only", True))
    max_files: int = int(args.get("max_files", 100))

    target = Path(path_str).resolve()
    if target.is_file():
      try:
        matches: List[Tuple[int, str]] = []
        with open(target, "r", errors="ignore") as f:
          for idx, line in enumerate(f, 1):
            if term in line:
              matches.append((idx, line.rstrip("\n")))
        if not matches:
          return {
              "output": f'No matches found for "{term}" in {target}',
              "exit_code": "0",
          }
        lines = [f"{ln}:{txt}" for ln, txt in matches]
        header = f'Matches for "{term}" in {target}:'
        return {"output": header + "\n" + "\n".join(lines), "exit_code": "0"}
      except Exception as e:
        return {"output": f"Error reading file: {repr(e)}", "exit_code": "-1"}

    if not target.exists() or not target.is_dir():
      return {
          "output": f"Path not found or not a directory: {target}",
          "exit_code": "-1",
      }

    files = list_non_hidden_files(target, max_depth=2, python_only=python_only)
    results: List[Tuple[Path, int]] = []
    for f in files:
      try:
        count = 0
        with open(f, "r", errors="ignore") as fh:
          for line in fh:
            if term in line:
              count += 1
        if count > 0:
          results.append((f, count))
      except Exception:
        continue

    if not results:
      return {
          "output": f'No matches found for "{term}" in {target}',
          "exit_code": "0",
      }

    if len(results) > max_files:
      return {
          "output": f'More than {len(results)} files matched for "{term}" in {target}. Please narrow your search.',
          "exit_code": "0",
      }

    lines = [
        f"{os.path.relpath(p, start=os.getcwd())} ({cnt} matches)"
        for p, cnt in results
    ]
    summary = f'Found {sum(cnt for _, cnt in results)} matches for "{term}" in {target}:'
    return {"output": summary + "\n" + "\n".join(lines), "exit_code": "0"}


__all__ = ["SearchTools"]
