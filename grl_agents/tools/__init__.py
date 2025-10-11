from .base_tool import tool, ToolGroup, ToolManager
from .execute_bash import ExecuteBashTools
from .search import SearchTools
from .file_editor import FileEditorTools
from .finish import FinishTools


def build_default_tool_manager() -> ToolManager:
  manager = ToolManager()
  manager.add_group(ExecuteBashTools())
  manager.add_group(SearchTools())
  manager.add_group(FileEditorTools())
  manager.add_group(FinishTools())
  return manager


__all__ = [
  "tool",
  "ToolGroup",
  "ToolManager",
  "build_default_tool_manager",
]


