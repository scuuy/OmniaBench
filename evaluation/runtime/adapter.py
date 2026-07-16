"""
adapter：兼容旧导入路径的系统能力 shim。

说明：
- 旧代码生成 prompt 允许 `from adapter import ...`；
- 新的系统能力实现在 `adapters/` 下；
- 本文件仅做轻量转发，避免调用侧感知目录重构。
"""

from __future__ import annotations

from pathlib import Path

from adapters.code_adapter import run_python_executor
from adapters.fs_adapter import ensure_fs_sandbox
from adapters.strategy_config import DEFAULT_TMP_ROOT


def get_fs_sandbox(self_obj):
    """为环境实例懒加载文件沙箱。"""
    runtime = getattr(self_obj, "_adapter_runtime", None)
    tmp_root = ""
    max_file_size_bytes = 262144
    max_list_entries = 200
    if isinstance(runtime, dict):
        fs_cfg = runtime.get("fs", {}) if isinstance(runtime.get("fs", {}), dict) else {}
        tmp_root = str(fs_cfg.get("tmp_root", "")).strip()
        max_file_size_bytes = int(fs_cfg.get("max_file_size_bytes", max_file_size_bytes))
        max_list_entries = int(fs_cfg.get("max_list_entries", max_list_entries))
    if not tmp_root:
        env_id = str(getattr(self_obj, "env_id", "") or "").strip()
        tmp_root = str(Path(DEFAULT_TMP_ROOT) / (env_id or "default_env"))
    return ensure_fs_sandbox(
        self_obj,
        tmp_root=tmp_root,
        max_file_size_bytes=max_file_size_bytes,
        max_list_entries=max_list_entries,
    )


__all__ = ["get_fs_sandbox", "run_python_executor"]
