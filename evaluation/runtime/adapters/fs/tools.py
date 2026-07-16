"""
fs tools：注入到环境中的少量通用文件工具定义。
"""

from __future__ import annotations

from typing import Any, Dict, List


FS_CAPABILITY_TAG = "filesystem"


def _normalize_language_mode(language_mode: str | None) -> str:
    text = str(language_mode or "").strip().lower()
    if text.startswith("en"):
        return "en"
    return "zh"


def build_fs_operation_specs(language_mode: str = "zh") -> List[Dict[str, Any]]:
    english_mode = _normalize_language_mode(language_mode) == "en"
    list_doc = "List sandbox directory contents." if english_mode else "列出沙箱目录内容。"
    read_doc = "Read a file from the sandbox." if english_mode else "读取沙箱中的文件。"
    write_doc = "Write a file into the sandbox." if english_mode else "向沙箱写入文件。"
    move_doc = "Move a path inside the sandbox." if english_mode else "在沙箱中移动路径。"
    return [
        {
            "operation_name": "fs_list_dir",
            "operation_description": (
                "List files and subdirectories in the sandbox, optionally recursively; useful for discovering input files and organizing output folders."
                if english_mode
                else "列出沙箱目录中的文件与子目录，可选递归，适合发现输入文件与整理结果目录。"
            ),
            "operation_type": "query",
            "capability_tags": ["filesystem"],
            "adapter_backed": True,
            "system_tool_kind": "filesystem",
            "code": """def fs_list_dir(self, relative_path: str = "", recursive: bool = False) -> dict:
    \"\"\"__DOC__\"\"\"
    try:
        from adapter import get_fs_sandbox
        sandbox = get_fs_sandbox(self)
        return sandbox.list_dir(relative_path=relative_path, recursive=recursive)
    except Exception as e:
        return {"success": False, "error": str(e), "message": "fs_list_dir_failed", "data": {}}
""".replace("__DOC__", list_doc),
        },
        {
            "operation_name": "fs_read_file",
            "operation_description": (
                "Read any file in the sandbox with auto/text/base64 output; useful for configs, exports, logs, and documents."
                if english_mode
                else "读取沙箱中的任意文件，支持 auto/text/base64 输出，适合读取配置、表格导出、日志与文档内容。"
            ),
            "operation_type": "query",
            "capability_tags": ["filesystem"],
            "adapter_backed": True,
            "system_tool_kind": "filesystem",
            "code": """def fs_read_file(
    self,
    relative_path: str,
    output_format: str = "auto",
    encoding: str = "utf-8",
    max_bytes: int = 65536,
) -> dict:
    \"\"\"__DOC__\"\"\"
    try:
        from adapter import get_fs_sandbox
        sandbox = get_fs_sandbox(self)
        return sandbox.read_file(
            relative_path=relative_path,
            output_format=output_format,
            encoding=encoding,
            max_bytes=max_bytes,
        )
    except Exception as e:
        return {"success": False, "error": str(e), "message": "fs_read_file_failed", "data": {}}
""".replace("__DOC__", read_doc),
        },
        {
            "operation_name": "fs_write_file",
            "operation_description": (
                "Write any file into the sandbox with text/base64 content; useful for intermediate artifacts, exports, and cache files."
                if english_mode
                else "向沙箱写入任意文件，支持 text/base64，用于生成中间结果、导出结果与缓存文件。"
            ),
            "operation_type": "state_change",
            "capability_tags": ["filesystem"],
            "adapter_backed": True,
            "system_tool_kind": "filesystem",
            "code": """def fs_write_file(
    self,
    relative_path: str,
    content: str,
    content_format: str = "text",
    encoding: str = "utf-8",
    overwrite: bool = False,
) -> dict:
    \"\"\"__DOC__\"\"\"
    try:
        from adapter import get_fs_sandbox
        sandbox = get_fs_sandbox(self)
        return sandbox.write_file(
            relative_path=relative_path,
            content=content,
            content_format=content_format,
            encoding=encoding,
            overwrite=overwrite,
        )
    except Exception as e:
        return {"success": False, "error": str(e), "message": "fs_write_file_failed", "data": {}}
""".replace("__DOC__", write_doc),
        },
        {
            "operation_name": "fs_move_path",
            "operation_description": (
                "Move or rename files/directories inside the sandbox; useful for archiving, organizing outputs, and adjusting paths."
                if english_mode
                else "在沙箱中移动或重命名文件/目录，用于归档、整理输出结构与调整文件位置。"
            ),
            "operation_type": "state_change",
            "capability_tags": ["filesystem"],
            "adapter_backed": True,
            "system_tool_kind": "filesystem",
            "code": """def fs_move_path(self, src_path: str, dst_path: str, overwrite: bool = False) -> dict:
    \"\"\"__DOC__\"\"\"
    try:
        from adapter import get_fs_sandbox
        sandbox = get_fs_sandbox(self)
        return sandbox.move_path(src_path=src_path, dst_path=dst_path, overwrite=overwrite)
    except Exception as e:
        return {"success": False, "error": str(e), "message": "fs_move_path_failed", "data": {}}
""".replace("__DOC__", move_doc),
        },
    ]


def detect_fs_tool_names(env_item: Dict[str, Any]) -> List[str]:
    names = []
    for op in (env_item.get("operation_list", []) if isinstance(env_item, dict) else []):
        if not isinstance(op, dict):
            continue
        tags = {str(x).strip() for x in (op.get("capability_tags", []) or [])}
        if op.get("system_tool_kind") == "filesystem" or FS_CAPABILITY_TAG in tags:
            name = str(op.get("operation_name", "")).strip()
            if name:
                names.append(name)
    return names
