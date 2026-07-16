"""
fs_adapter：兼容旧导入路径的 filesystem shim。

说明：
- 真实实现已迁移到 `adapters/fs/`；
- 本文件仅做转发，避免旧调用侧失效。
"""

from __future__ import annotations

from adapters.fs import (
    SafeFileSandbox,
    build_fs_input_prompt_addendum,
    build_fs_operation_specs,
    build_fs_prompt_addendum,
    build_fs_fixtures_for_task,
    detect_fs_tool_names,
    ensure_fs_sandbox,
    infer_file_metadata,
)

FS_CAPABILITY_TAG = "filesystem"

__all__ = [
    "FS_CAPABILITY_TAG",
    "SafeFileSandbox",
    "ensure_fs_sandbox",
    "infer_file_metadata",
    "build_fs_prompt_addendum",
    "build_fs_input_prompt_addendum",
    "build_fs_operation_specs",
    "detect_fs_tool_names",
    "build_fs_fixtures_for_task",
]
