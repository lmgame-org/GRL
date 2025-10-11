from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_tool import ToolGroup, tool
from .tool_utils import get_workspace_root


class FileEditorTools(ToolGroup):

  def __init__(self):
    super().__init__(name="editor")
    self._state_path = Path("/tmp/sokoban_editor_state.json")

  def _load_history(self) -> Dict[str, List[str]]:
    if self._state_path.exists():
      try:
        return json.loads(self._state_path.read_text(encoding="utf-8"))
      except Exception:
        return {}
    return {}

  def _save_history(self, history: Dict[str, List[str]]) -> None:
    try:
      self._state_path.write_text(json.dumps(history), encoding="utf-8")
    except Exception:
      pass

  @staticmethod
  def _read_text(path: Path) -> str:
    try:
      return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
      return path.read_text(errors="ignore")

  @staticmethod
  def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")

  @tool(
      schema={
          "type": "object",
          "properties": {
              "command": {
                  "type": "string",
                  "enum": [
                      "view",
                      "create",
                      "str_replace",
                      "insert",
                      "undo_edit",
                  ],
              },
              "path": {
                  "type": "string",
                  "description": "Absolute or relative file/directory path.",
              },
              "file_text": {
                  "type": "string",
                  "description": "Required for create.",
              },
              "old_str": {
                  "type": "string",
                  "description": "Required for str_replace (must match uniquely).",
              },
              "new_str": {
                  "type": "string",
                  "description": "Replacement string for str_replace or inserted text for insert.",
              },
              "insert_line": {
                  "type": "integer",
                  "description": "For insert: new_str inserted AFTER this 0-based line index.",
              },
              "view_range": {
                  "type": "array",
                  "items": {"type": "integer"},
                  "description": "[start, end]; end=-1 for EOF; 1-based indices.",
              },
          },
          "required": ["command", "path"],
      },
      description="View or edit files: view/create/str_replace/insert/undo_edit with snippet outputs and simple validation.",
  )
  def file_editor(self, args: Dict[str, Any]) -> Dict[str, Any]:
    command: str = args.get("command", "")
    path_str: str = args.get("path", "")
    file_text: Optional[str] = args.get("file_text")
    old_str: Optional[str] = args.get("old_str")
    new_str: Optional[str] = args.get("new_str")
    insert_line_raw: Optional[Any] = args.get("insert_line")
    view_range: Optional[List[int]] = args.get("view_range")

    workspace_root = get_workspace_root()
    raw_p = Path(path_str)
    p = (raw_p if raw_p.is_absolute() else (workspace_root / raw_p)).resolve()

    history = self._load_history()

    if command == "view":
      if p.is_dir():
        files = []
        base_depth = len(p.resolve().parts)
        for root, dirs, fs in os.walk(p):
          dirs[:] = [d for d in dirs if not d.startswith(".")]
          depth = len(Path(root).resolve().parts) - base_depth
          if depth > 2:
            dirs[:] = []
            continue
          for f in fs:
            if f.startswith("."):
              continue
            fp = Path(root) / f
            if fp.suffix == ".py":
              files.append(str(fp))
        return {
            "output": "\n".join(files) if files else f"<empty dir> {p}",
            "exit_code": "0",
        }
      if not p.exists():
        return {"output": f"The path '{p}' does not exist.", "exit_code": "-1"}
      text = self._read_text(p).expandtabs()
      lines = text.splitlines()
      start, end = 1, len(lines)
      if view_range and len(view_range) == 2:
        start = max(1, int(view_range[0]))
        end = (
            len(lines)
            if int(view_range[1]) == -1
            else min(int(view_range[1]), len(lines))
        )
        if start > end:
          return {
              "output": f"Invalid view_range {view_range}",
              "exit_code": "-1",
          }
      numbered = "\n".join(
          f"{i:6d} {line}"
          for i, line in enumerate(lines[start - 1 : end], start)
      )
      return {
          "output": f"Here's the result of running `cat -n` on {p}:\n"
          + numbered,
          "exit_code": "0",
      }

    if command == "create":
      if p.exists():
        return {"output": f"File already exists: {p}", "exit_code": "-1"}
      if file_text is None:
        return {"output": "Missing 'file_text' for create.", "exit_code": "-1"}
      self._write_text(p, file_text)
      history.setdefault(str(p), []).append("")
      self._save_history(history)
      return {"output": f"File created at {p}.", "exit_code": "0"}

    if command == "str_replace":
      if not p.exists() or p.is_dir():
        return {"output": f"Invalid file: {p}", "exit_code": "-1"}
      if old_str is None:
        return {
            "output": "Missing 'old_str' for str_replace.",
            "exit_code": "-1",
        }
      text = self._read_text(p).expandtabs()
      old_str_exp = old_str.expandtabs()
      occurrences = text.count(old_str_exp)
      if occurrences == 0:
        return {
            "output": f"No occurrences of provided old_str found in {p}.",
            "exit_code": "-1",
        }
      if occurrences > 1:
        return {
            "output": f"Multiple occurrences found; provide more context to make it unique.",
            "exit_code": "-1",
        }
      new_content = text.replace(old_str_exp, (new_str or "").expandtabs())
      history.setdefault(str(p), []).append(text)
      self._write_text(p, new_content)
      self._save_history(history)
      return {"output": f"Edited {p} successfully.", "exit_code": "0"}

    if command == "insert":
      if not p.exists() or p.is_dir():
        return {"output": f"Invalid file: {p}", "exit_code": "-1"}
      if new_str is None:
        return {"output": "Missing 'new_str' for insert.", "exit_code": "-1"}
      try:
        insert_line = int(insert_line_raw)
      except Exception:
        return {
            "output": "Invalid or missing 'insert_line' for insert.",
            "exit_code": "-1",
        }
      text = self._read_text(p).expandtabs()
      lines = text.split("\n")
      if insert_line < 0 or insert_line > len(lines):
        return {
            "output": f"insert_line out of range [0, {len(lines)}]",
            "exit_code": "-1",
        }
      new_lines = (
          lines[:insert_line]
          + (new_str.expandtabs()).split("\n")
          + lines[insert_line:]
      )
      history.setdefault(str(p), []).append(text)
      self._write_text(p, "\n".join(new_lines))
      self._save_history(history)
      return {
          "output": f"Inserted into {p} at line {insert_line}.",
          "exit_code": "0",
      }

    if command == "undo_edit":
      key = str(p)
      stack = history.get(key, [])
      if not stack:
        return {
            "output": f"No previous edits found for {p}.",
            "exit_code": "-1",
        }
      prev = stack.pop()
      self._write_text(p, prev)
      history[key] = stack
      self._save_history(history)
      return {"output": f"Undo successful for {p}.", "exit_code": "0"}

    return {"output": f"Unrecognized command '{command}'.", "exit_code": "-1"}


__all__ = ["FileEditorTools"]
