"""
兜底生成器：尽量保证任意 file_type 都可落盘。
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict

from .base import BaseFixtureGenerator, ensure_mapping, normalize_file_type


_KNOWN_EXPLICIT_TYPES = {
    "csv", "tsv", "json", "jsonl", "yaml", "yml", "xml", "ini", "toml",
    "html", "htm", "sql", "py", "js", "sh", "xlsx", "docx", "pdf", "eml",
    "zip", "txt", "md", "log",
}


class DefaultStructuredTextGenerator(BaseFixtureGenerator):
    def supports(self, fs_input: Dict[str, Any]) -> bool:
        file_type = normalize_file_type(fs_input)
        if not file_type:
            return True
        return file_type not in _KNOWN_EXPLICIT_TYPES

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        gold = ensure_mapping(fs_input.get("gold_read_result"))
        payload = {
            "purpose": fs_input.get("purpose", ""),
            "gold_read_result": gold,
            "content_spec": fs_input.get("content_spec", {}),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


class DefaultBinaryLikeGenerator(BaseFixtureGenerator):
    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        gold = json.dumps(ensure_mapping(fs_input.get("gold_read_result")), ensure_ascii=False).encode("utf-8")
        return base64.b16encode(gold)
