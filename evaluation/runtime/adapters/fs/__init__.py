"""
fs：文件系统策略的统一入口。

目标：
- 对外提供少量、稳定的 FS 工具与 prompt 附加说明；
- 对内维护安全沙箱与 step6.5 所需的 fixture 构造能力；
- 保留旧接口兼容，避免调用侧感知目录调整。
"""

from .runtime import SafeFileSandbox, ensure_fs_sandbox, infer_file_metadata
from .tools import build_fs_operation_specs, detect_fs_tool_names
from .prompting import build_fs_prompt_addendum, build_fs_input_prompt_addendum
from .fixture_builder import build_fs_fixtures_for_task

__all__ = [
    "SafeFileSandbox",
    "ensure_fs_sandbox",
    "infer_file_metadata",
    "build_fs_operation_specs",
    "detect_fs_tool_names",
    "build_fs_prompt_addendum",
    "build_fs_input_prompt_addendum",
    "build_fs_fixtures_for_task",
]
