"""
文本与文档类生成器。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseFixtureGenerator, build_email_bytes, build_minimal_docx, build_minimal_pdf, ensure_mapping, noise_count


def _gold_lines(fs_input: Dict[str, Any]) -> List[str]:
    purpose = str(fs_input.get("purpose", "") or "").strip()
    gold = ensure_mapping(fs_input.get("gold_read_result"))
    lines = []
    if purpose:
        lines.append(f"用途: {purpose}")
    for key, value in gold.items():
        lines.append(f"{key}: {value}")
    return lines or ["fixture"]


def _coerce_text_lines(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    return []


def _extract_content_spec(fs_input: Dict[str, Any]) -> Dict[str, Any]:
    return fs_input.get("content_spec", {}) if isinstance(fs_input.get("content_spec", {}), dict) else {}


def _document_title(fs_input: Dict[str, Any], spec: Dict[str, Any]) -> str:
    for key in ("title", "doc_title", "headline"):
        value = str(spec.get(key, "") or "").strip()
        if value:
            return value
    path_hint = str(fs_input.get("path_hint", "") or "").strip()
    stem = Path(path_hint).stem if path_hint else ""
    return stem.replace("_", " ").strip() or "Fixture Document"


def _build_body_lines(fs_input: Dict[str, Any], spec: Dict[str, Any]) -> List[str]:
    lines = []
    for key in ("summary", "excerpt"):
        lines.extend(_coerce_text_lines(spec.get(key)))
    for key in ("body_lines", "sample_lines", "key_points", "keywords"):
        lines.extend(_coerce_text_lines(spec.get(key)))
    if lines:
        return lines
    return _gold_lines(fs_input)


def _build_sections(spec: Dict[str, Any]) -> List[Dict[str, List[str]]]:
    sections = []
    raw_sections = spec.get("sections", [])
    if not isinstance(raw_sections, list):
        return sections
    for item in raw_sections:
        if not isinstance(item, dict):
            continue
        heading = str(item.get("heading") or item.get("title") or item.get("name") or "").strip()
        lines = []
        for key in ("lines", "content", "bullets", "points"):
            lines.extend(_coerce_text_lines(item.get(key)))
        if heading or lines:
            sections.append({"heading": heading or "Section", "lines": lines})
    return sections


def _content_driven_markdown(fs_input: Dict[str, Any], add_noise: bool) -> str:
    spec = _extract_content_spec(fs_input)
    title = _document_title(fs_input, spec)
    body_lines = _build_body_lines(fs_input, spec)
    sections = _build_sections(spec)
    gold_lines = _gold_lines(fs_input)
    text = f"# {title}\n\n"
    if body_lines:
        text += "\n\n".join(body_lines) + "\n"
    for section in sections:
        heading = str(section.get("heading", "")).strip() or "Section"
        section_lines = section.get("lines", []) if isinstance(section.get("lines", []), list) else []
        text += f"\n\n## {heading}\n"
        if section_lines:
            text += "\n".join(f"- {line}" for line in section_lines) + "\n"
    text += "\n\n## Key Facts\n" + "\n".join(f"- {line}" for line in gold_lines)
    if add_noise:
        extra_notes = _coerce_text_lines(spec.get("notes")) or [f"supporting_note_{idx}" for idx in range(noise_count(fs_input, default=3))]
        if extra_notes:
            text += "\n\n## Notes\n" + "\n".join(f"- {line}" for line in extra_notes)
    return text + "\n"


def _content_driven_text(fs_input: Dict[str, Any], add_noise: bool) -> str:
    spec = _extract_content_spec(fs_input)
    title = _document_title(fs_input, spec)
    body_lines = _build_body_lines(fs_input, spec)
    gold_lines = _gold_lines(fs_input)
    lines = [title, "=" * len(title), ""]
    lines.extend(body_lines)
    if body_lines:
        lines.append("")
    lines.append("Key Facts:")
    lines.extend(f"- {line}" for line in gold_lines)
    if add_noise:
        extra_lines = _coerce_text_lines(spec.get("notes")) or [f"extra_line_{idx}" for idx in range(noise_count(fs_input, default=3))]
        if extra_lines:
            lines.extend(["", "Notes:"])
            lines.extend(f"- {line}" for line in extra_lines)
    return "\n".join(lines).rstrip() + "\n"


def _content_driven_log(fs_input: Dict[str, Any], add_noise: bool) -> str:
    spec = _extract_content_spec(fs_input)
    entries = _coerce_text_lines(spec.get("log_lines")) or _coerce_text_lines(spec.get("body_lines"))
    if not entries:
        entries = _gold_lines(fs_input)
    lines = [f"INFO {line}" for line in entries]
    if add_noise:
        lines.extend(f"INFO supporting_event_{idx}" for idx in range(noise_count(fs_input, default=4)))
    return "\n".join(lines).rstrip() + "\n"


class TextLikeGenerator(BaseFixtureGenerator):
    supported_types = ("txt", "md", "log")
    supported_mime_prefixes = ("text/plain", "text/markdown")

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        file_type = str(fs_input.get("file_type", "") or "").strip().lower()
        if file_type == "log":
            text = _content_driven_log(fs_input, add_noise=add_noise)
        elif file_type == "md":
            text = _content_driven_markdown(fs_input, add_noise=add_noise)
        else:
            text = _content_driven_text(fs_input, add_noise=add_noise)
        return text.encode("utf-8")


class HtmlGenerator(BaseFixtureGenerator):
    supported_types = ("html", "htm")
    supported_mime_prefixes = ("text/html",)

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        lines = _gold_lines(fs_input)
        extras = ""
        if add_noise:
            extras = "".join(f"<li>extra_note_{idx}</li>" for idx in range(noise_count(fs_input, default=4)))
        body = "".join(f"<p>{line}</p>" for line in lines)
        return f"<html><body>{body}<ul>{extras}</ul></body></html>\n".encode("utf-8")


class XmlGenerator(BaseFixtureGenerator):
    supported_types = ("xml",)
    supported_mime_prefixes = ("application/xml", "text/xml")

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        from .structured import YamlLikeGenerator

        return YamlLikeGenerator().build_bytes(fs_input, add_noise=add_noise)


class ConfigTextGenerator(BaseFixtureGenerator):
    supported_types = ("ini", "toml", "yaml", "yml")

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        from .structured import YamlLikeGenerator

        return YamlLikeGenerator().build_bytes(fs_input, add_noise=add_noise)


class CodeTextGenerator(BaseFixtureGenerator):
    supported_types = ("sql", "py", "js", "sh")
    supported_mime_prefixes = ("text/x-", "application/sql")

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        file_type = str(fs_input.get("file_type", "") or "").strip().lower()
        gold = ensure_mapping(fs_input.get("gold_read_result"))
        if file_type == "sql":
            lines = ["-- generated fixture"] + [f"-- {k}={v}" for k, v in gold.items()]
            lines.append("SELECT 1;")
        elif file_type == "sh":
            lines = ["#!/usr/bin/env bash", "# generated fixture"] + [f"# {k}={v}" for k, v in gold.items()]
            lines.append('echo "fixture"')
        elif file_type == "js":
            payload = json.dumps(gold, ensure_ascii=False, indent=2)
            lines = ["// generated fixture", f"const gold = {payload};", "console.log(gold);"]
        else:
            payload = json.dumps(gold, ensure_ascii=False, indent=2)
            lines = ["# generated fixture", f"gold = {payload}", "print(gold)"]
        if add_noise:
            lines.append(f"# noise_count={noise_count(fs_input, default=3)}")
        return ("\n".join(lines) + "\n").encode("utf-8")


class DocxGenerator(BaseFixtureGenerator):
    supported_types = ("docx",)

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        paragraphs = _gold_lines(fs_input)
        if add_noise:
            paragraphs.extend(f"补充段落 {idx}" for idx in range(noise_count(fs_input, default=4)))
        return build_minimal_docx(paragraphs)


class PdfGenerator(BaseFixtureGenerator):
    supported_types = ("pdf",)
    supported_mime_prefixes = ("application/pdf",)

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        lines = _gold_lines(fs_input)
        if add_noise:
            lines.extend(f"extra note {idx}" for idx in range(noise_count(fs_input, default=3)))
        return build_minimal_pdf(lines)


class EmailGenerator(BaseFixtureGenerator):
    supported_types = ("eml",)
    supported_mime_prefixes = ("message/",)

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        lines = _gold_lines(fs_input)
        body = "\n".join(lines)
        if add_noise:
            body += "\n\n--\nSystem Generated Mail"
        subject = str(fs_input.get("purpose", "") or "Fixture Email")
        return build_email_bytes(subject=subject, body=body)
