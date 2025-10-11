from __future__ import annotations

from typing import Any, Dict

from .base_tool import ToolGroup, tool


class FinishTools(ToolGroup):

  def __init__(self):
    super().__init__(name="finish")

  @tool(
      schema={
          "type": "object",
          "properties": {
              "command": {"type": "string", "enum": ["submit"]},
              "result": {"type": "string"},
          },
          "required": ["command"],
      },
      description="Signal completion and optionally return a final result string.",
  )
  def finish(self, args: Dict[str, Any]) -> Dict[str, Any]:
    cmd = args.get("command", "submit")
    if cmd != "submit":
      return {
          "output": f"Invalid command for finish: {cmd}",
          "exit_code": "-1",
          "done": False,
      }
    result = args.get("result", "")
    return {
        "output": "Submitted.",
        "exit_code": "0",
        "done": True,
        "result": result,
    }


__all__ = ["FinishTools"]
