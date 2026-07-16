"""
压缩包类生成器。
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any, Dict

from .base import BaseFixtureGenerator, ensure_mapping


class ZipBundleGenerator(BaseFixtureGenerator):
    supported_types = ("zip",)
    supported_mime_prefixes = ("application/zip",)

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        gold = ensure_mapping(fs_input.get("gold_read_result"))
        member_name = "payload.json"
        read_expectation = fs_input.get("read_expectation", {}) if isinstance(fs_input.get("read_expectation", {}), dict) else {}
        locator = read_expectation.get("locator_hint", {}) if isinstance(read_expectation.get("locator_hint", {}), dict) else {}
        hinted = str(locator.get("member_name", "") or "").strip()
        if hinted:
            member_name = hinted
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(member_name, json.dumps(gold, ensure_ascii=False, indent=2))
            if add_noise:
                zf.writestr("README.txt", "generated zip fixture\n")
        return mem.getvalue()
