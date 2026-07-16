"""
生成器基础能力与共享工具函数。
"""

from __future__ import annotations

import io
import json
import zipfile
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, Iterable, List
from xml.sax.saxutils import escape


class BaseFixtureGenerator:
    supported_types: tuple[str, ...] = ()
    supported_mime_prefixes: tuple[str, ...] = ()

    def supports(self, fs_input: Dict[str, Any]) -> bool:
        file_type = normalize_file_type(fs_input.get("file_type", ""))
        mime_type = str(fs_input.get("mime_type", "") or "").strip().lower()
        supported_types = tuple(getattr(type(self), "supported_types", ()) or ())
        supported_mime_prefixes = tuple(getattr(type(self), "supported_mime_prefixes", ()) or ())
        if file_type and file_type in supported_types:
            return True
        return any(mime_type.startswith(prefix) for prefix in supported_mime_prefixes)

    def write(self, target_path: Path, fs_input: Dict[str, Any], add_noise: bool = True) -> None:
        data = self.build_bytes(fs_input, add_noise=add_noise)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(data)

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        raise NotImplementedError


def normalize_file_type(file_type: Any) -> str:
    text = str(file_type or "").strip().lower()
    if text.startswith("."):
        text = text[1:]
    return text


def ensure_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"items": value}
    if value is None:
        return {}
    return {"value": value}


def noise_count(fs_input: Dict[str, Any], default: int = 5) -> int:
    spec = fs_input.get("content_spec", {}) if isinstance(fs_input.get("content_spec", {}), dict) else {}
    value = spec.get("min_noise_items", spec.get("min_noise_rows", default))
    try:
        return max(0, int(value))
    except Exception:
        return default


def schema_hint(fs_input: Dict[str, Any], fallback_keys: Iterable[str]) -> List[str]:
    spec = fs_input.get("content_spec", {}) if isinstance(fs_input.get("content_spec", {}), dict) else {}
    hint = spec.get("schema_hint", [])
    if isinstance(hint, list):
        fields = [str(x).strip() for x in hint if str(x).strip()]
        if fields:
            return fields
    keys = [str(x).strip() for x in fallback_keys if str(x).strip()]
    return keys or ["key", "value"]


def yaml_dump_simple(value: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(yaml_dump_simple(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {scalar_to_text(item)}")
        return "\n".join(lines) if lines else f"{prefix}{{}}"
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(yaml_dump_simple(item, indent + 2))
            else:
                lines.append(f"{prefix}- {scalar_to_text(item)}")
        return "\n".join(lines) if lines else f"{prefix}[]"
    return f"{prefix}{scalar_to_text(value)}"


def scalar_to_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if any(ch in text for ch in [":", "#", "\n", "\""]):
        return json.dumps(text, ensure_ascii=False)
    return text


def toml_dump_simple(value: Dict[str, Any]) -> str:
    lines: List[str] = []
    scalar_items = {}
    nested_items = {}
    for key, item in value.items():
        if isinstance(item, dict):
            nested_items[key] = item
        else:
            scalar_items[key] = item
    for key, item in scalar_items.items():
        lines.append(f"{key} = {json.dumps(item, ensure_ascii=False)}")
    for key, item in nested_items.items():
        lines.append(f"\n[{key}]")
        for sub_key, sub_value in item.items():
            lines.append(f"{sub_key} = {json.dumps(sub_value, ensure_ascii=False)}")
    return "\n".join(lines).strip() + "\n"


def ini_dump_simple(value: Dict[str, Any]) -> str:
    sections = []
    flat = {k: v for k, v in value.items() if not isinstance(v, dict)}
    if flat:
        sections.append("[default]")
        for key, item in flat.items():
            sections.append(f"{key} = {item}")
    for key, item in value.items():
        if not isinstance(item, dict):
            continue
        sections.append(f"\n[{key}]")
        for sub_key, sub_value in item.items():
            sections.append(f"{sub_key} = {sub_value}")
    return "\n".join(sections).strip() + "\n"


def xml_from_mapping(root_name: str, mapping: Dict[str, Any]) -> bytes:
    lines = [f"<{root_name}>"]
    for key, value in ensure_mapping(mapping).items():
        lines.append(f"  <{key}>{escape(str(value))}</{key}>")
    lines.append(f"</{root_name}>")
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_minimal_pdf(text_lines: List[str]) -> bytes:
    text_content = " ".join(text_lines[:12]) or "fixture"
    text_content = text_content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({text_content}) Tj ET"
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        f"5 0 obj << /Length {len(stream.encode('utf-8'))} >> stream\n{stream}\nendstream endobj\n".encode("utf-8"),
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(out.tell())
        out.write(obj)
    xref_pos = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(f"{off:010d} 00000 n \n".encode("ascii"))
    out.write(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode("ascii"))
    return out.getvalue()


def build_minimal_docx(paragraphs: List[str]) -> bytes:
    body = "".join(
        f"<w:p><w:r><w:t>{escape(line)}</w:t></w:r></w:p>"
        for line in paragraphs
        if str(line).strip()
    ) or "<w:p><w:r><w:t>fixture</w:t></w:r></w:p>"
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    document = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>{body}<w:sectPr/></w:body>
</w:document>"""
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document)
    return mem.getvalue()


def build_minimal_xlsx(rows: List[List[Any]]) -> bytes:
    worksheet_rows = []
    for row_idx, row in enumerate(rows, 1):
        cells = []
        for col_idx, value in enumerate(row, 1):
            col = chr(ord("A") + col_idx - 1)
            text = escape(str(value))
            cells.append(f'<c r="{col}{row_idx}" t="inlineStr"><is><t>{text}</t></is></c>')
        worksheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    sheet_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>{"".join(worksheet_rows)}</sheetData>
</worksheet>"""
    workbook = """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
    rels_root = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"""
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels_root)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    return mem.getvalue()


def build_email_bytes(subject: str, body: str, to_addr: str = "ops@example.com") -> bytes:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "noreply@example.com"
    msg["To"] = to_addr
    msg.set_content(body)
    return msg.as_bytes()
