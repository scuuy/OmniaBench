"""
fs runtime：安全文件沙箱与通用文件元信息处理。
"""

from __future__ import annotations

import base64
import mimetypes
import shutil
from pathlib import Path
from typing import Any, Dict


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".log",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".ini",
    ".toml",
    ".sql",
    ".py",
    ".js",
    ".ts",
    ".sh",
    ".eml",
}


def _looks_like_text(data: bytes) -> bool:
    if not data:
        return True
    sample = data[:4096]
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def infer_file_metadata(path: str | Path, size: int | None = None, provided_mime_type: str = "") -> Dict[str, Any]:
    target = Path(path)
    mime_type = str(provided_mime_type or "").strip() or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    extension = target.suffix.lower()
    is_text = mime_type.startswith("text/") or extension in TEXT_EXTENSIONS
    return {
        "path": str(target),
        "file_name": target.name,
        "extension": extension,
        "mime_type": mime_type,
        "is_text": bool(is_text),
        "size": int(size or 0),
    }


class SafeFileSandbox:
    """把文件操作限制在指定根目录下。"""

    def __init__(self, root_dir: str, max_file_size_bytes: int = 262144, max_list_entries: int = 200):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_bytes = max(1024, int(max_file_size_bytes or 262144))
        self.max_list_entries = max(10, int(max_list_entries or 200))

    def resolve_path(self, relative_path: str) -> Path:
        raw = str(relative_path or "").strip().replace("\\", "/").lstrip("/")
        target = (self.root_dir / raw).resolve()
        if target != self.root_dir and self.root_dir not in target.parents:
            raise ValueError("path escapes sandbox root")
        return target

    def list_dir(self, relative_path: str = "", recursive: bool = False) -> Dict[str, Any]:
        target = self.resolve_path(relative_path)
        if not target.exists():
            return {"success": False, "error": "path_not_found", "message": "目标路径不存在", "data": {}}
        if not target.is_dir():
            return {"success": False, "error": "path_is_not_directory", "message": "目标路径不是目录", "data": {}}
        entries = []
        iterator = target.rglob("*") if recursive else target.iterdir()
        for entry in iterator:
            if len(entries) >= self.max_list_entries:
                break
            meta = infer_file_metadata(entry, size=(entry.stat().st_size if entry.is_file() else 0))
            entries.append(
                {
                    "path": str(entry.relative_to(self.root_dir)),
                    "name": entry.name,
                    "is_dir": entry.is_dir(),
                    "size": meta["size"],
                    "extension": meta["extension"],
                    "mime_type": meta["mime_type"],
                }
            )
        return {
            "success": True,
            "error": "",
            "message": "directory_listed",
            "data": {
                "path": str(target.relative_to(self.root_dir)) if target != self.root_dir else "",
                "entries": entries,
                "truncated": len(entries) >= self.max_list_entries,
            },
        }

    def read_file(
        self,
        relative_path: str,
        output_format: str = "auto",
        encoding: str = "utf-8",
        max_bytes: int = 65536,
    ) -> Dict[str, Any]:
        target = self.resolve_path(relative_path)
        if not target.exists():
            return {"success": False, "error": "path_not_found", "message": "目标文件不存在", "data": {}}
        if not target.is_file():
            return {"success": False, "error": "path_is_not_file", "message": "目标路径不是文件", "data": {}}
        limit = min(max(1, int(max_bytes or 65536)), self.max_file_size_bytes)
        raw = target.read_bytes()
        data = raw[:limit]
        meta = infer_file_metadata(target, size=len(raw))
        chosen = str(output_format or "auto").strip().lower()
        if chosen == "auto":
            chosen = "text" if meta["is_text"] and _looks_like_text(data) else "base64"
        if chosen == "base64":
            content = base64.b64encode(data).decode("ascii")
        else:
            content = data.decode(str(encoding or "utf-8"), errors="replace")
            chosen = "text"
        return {
            "success": True,
            "error": "",
            "message": "file_read",
            "data": {
                **meta,
                "path": str(target.relative_to(self.root_dir)),
                "content": content,
                "output_format": chosen,
                "truncated": len(raw) > len(data),
            },
        }

    def write_file(
        self,
        relative_path: str,
        content: str,
        content_format: str = "text",
        encoding: str = "utf-8",
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        target = self.resolve_path(relative_path)
        if target.exists() and not bool(overwrite):
            return {"success": False, "error": "path_already_exists", "message": "目标文件已存在", "data": {}}
        target.parent.mkdir(parents=True, exist_ok=True)
        fmt = str(content_format or "text").strip().lower()
        if fmt == "base64":
            data = base64.b64decode(str(content or "").encode("ascii"))
        else:
            data = str(content or "").encode(str(encoding or "utf-8"))
        if len(data) > self.max_file_size_bytes:
            return {"success": False, "error": "file_too_large", "message": "文件超过大小限制", "data": {}}
        target.write_bytes(data)
        meta = infer_file_metadata(target, size=len(data))
        return {
            "success": True,
            "error": "",
            "message": "file_written",
            "data": {**meta, "path": str(target.relative_to(self.root_dir))},
        }

    def move_path(self, src_path: str, dst_path: str, overwrite: bool = False) -> Dict[str, Any]:
        src = self.resolve_path(src_path)
        dst = self.resolve_path(dst_path)
        if not src.exists():
            return {"success": False, "error": "source_not_found", "message": "源路径不存在", "data": {}}
        if dst.exists() and not bool(overwrite):
            return {"success": False, "error": "destination_exists", "message": "目标路径已存在", "data": {}}
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        meta = infer_file_metadata(dst, size=(dst.stat().st_size if dst.is_file() else 0))
        return {
            "success": True,
            "error": "",
            "message": "path_moved",
            "data": {
                "src_path": str(src.relative_to(self.root_dir)),
                "dst_path": str(dst.relative_to(self.root_dir)),
                "is_dir": dst.is_dir(),
                "size": meta["size"],
                "mime_type": meta["mime_type"],
            },
        }


def _ensure_runtime_config(self_obj) -> Dict[str, Any]:
    runtime = getattr(self_obj, "_adapter_runtime", None)
    if isinstance(runtime, dict):
        return runtime
    runtime = {"fs": {"tmp_root": "", "max_file_size_bytes": 262144, "max_list_entries": 200}}
    setattr(self_obj, "_adapter_runtime", runtime)
    return runtime


def ensure_fs_sandbox(self_obj, tmp_root: str, max_file_size_bytes: int = 262144, max_list_entries: int = 200) -> SafeFileSandbox:
    runtime = _ensure_runtime_config(self_obj)
    fs_cfg = runtime.setdefault("fs", {})
    if not fs_cfg.get("tmp_root"):
        fs_cfg["tmp_root"] = str(tmp_root)
    fs_cfg.setdefault("max_file_size_bytes", int(max_file_size_bytes))
    fs_cfg.setdefault("max_list_entries", int(max_list_entries))
    sandbox = getattr(self_obj, "_fs_sandbox", None)
    if isinstance(sandbox, SafeFileSandbox):
        return sandbox
    sandbox = SafeFileSandbox(
        root_dir=str(fs_cfg["tmp_root"]),
        max_file_size_bytes=int(fs_cfg["max_file_size_bytes"]),
        max_list_entries=int(fs_cfg["max_list_entries"]),
    )
    setattr(self_obj, "_fs_sandbox", sandbox)
    return sandbox
