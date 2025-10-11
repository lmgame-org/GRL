from __future__ import annotations

from typing import Any, Dict

from .base_tool import ToolGroup, tool
from .tool_utils import safe_run_shell


class ExecuteBashTools(ToolGroup):
  def __init__(self):
    super().__init__(name="bash")

  @tool(
    schema={
      "type": "object",
      "properties": {
        "cmd": {"type": "string", "description": "The shell command to execute."},
        "timeout": {"type": "integer", "description": "Timeout seconds (default 120)."},
      },
      "required": ["cmd"],
    },
    description="Execute a bash command in the local shell with a timeout.",
  )
  def execute_bash(self, args: Dict[str, Any]) -> Dict[str, Any]:
    cmd: str = args.get("cmd", "")
    timeout: int = int(args.get("timeout", 120))
    output, exit_code = safe_run_shell(cmd, timeout=timeout)
    return {"output": output, "exit_code": exit_code}


__all__ = ["ExecuteBashTools"]


