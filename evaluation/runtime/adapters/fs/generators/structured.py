"""
结构化文件生成器：csv/json/yaml/xlsx 等。
"""

from __future__ import annotations

import csv
import io
import json
import random
from typing import Any, Dict, List

from .base import BaseFixtureGenerator, build_minimal_xlsx, ensure_mapping, noise_count, schema_hint, xml_from_mapping, yaml_dump_simple


def _coerce_rows(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            rows.append(dict(item))
    return rows


def _spec_rows(fs_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    spec = fs_input.get("content_spec", {}) if isinstance(fs_input.get("content_spec", {}), dict) else {}
    return _coerce_rows(spec.get("sample_rows")) or _coerce_rows(spec.get("rows"))


def _document_meta(fs_input: Dict[str, Any]) -> Dict[str, Any]:
    spec = fs_input.get("content_spec", {}) if isinstance(fs_input.get("content_spec", {}), dict) else {}
    meta: Dict[str, Any] = {}
    for key in ("title", "summary"):
        value = spec.get(key)
        if isinstance(value, str) and value.strip():
            meta[key] = value.strip()
    sections = spec.get("sections")
    if isinstance(sections, list) and sections:
        meta["sections"] = sections
    notes = spec.get("notes")
    if isinstance(notes, list) and notes:
        meta["notes"] = notes
    return meta


class CsvGenerator(BaseFixtureGenerator):
    supported_types = ("csv",)

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        gold = ensure_mapping(fs_input.get("gold_read_result"))
        spec_rows = _spec_rows(fs_input)
        field_candidates = list(gold.keys())
        for row in spec_rows:
            field_candidates.extend(row.keys())
        fields = schema_hint(fs_input, field_candidates)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: gold.get(field, "") for field in fields})
        for row in spec_rows:
            writer.writerow({field: row.get(field, gold.get(field, "")) for field in fields})
        count = noise_count(fs_input, default=12) if add_noise else 0
        rng = random.Random(str(fs_input.get("input_id", "csv")))
        for idx in range(count):
            row = {}
            for field in fields:
                if field in gold and idx == 0:
                    row[field] = f"{gold[field]}_alt"
                else:
                    row[field] = f"{field}_{rng.randint(1000, 9999)}"
            writer.writerow(row)
        return buffer.getvalue().encode("utf-8")


class TsvGenerator(CsvGenerator):
    supported_types = ("tsv",)

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        gold = ensure_mapping(fs_input.get("gold_read_result"))
        fields = schema_hint(fs_input, gold.keys())
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow({field: gold.get(field, "") for field in fields})
        count = noise_count(fs_input, default=12) if add_noise else 0
        for idx in range(count):
            writer.writerow({field: f"{field}_{idx}" for field in fields})
        return buffer.getvalue().encode("utf-8")


class JsonGenerator(BaseFixtureGenerator):
    supported_types = ("json",)

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        gold = ensure_mapping(fs_input.get("gold_read_result"))
        spec = fs_input.get("content_spec", {}) if isinstance(fs_input.get("content_spec", {}), dict) else {}
        mode = str(spec.get("json_mode", "") or "").strip().lower()
        read_expectation = fs_input.get("read_expectation", {}) if isinstance(fs_input.get("read_expectation", {}), dict) else {}
        spec_rows = _spec_rows(fs_input)
        if mode == "list_of_objects" or read_expectation.get("access_mode") == "table_lookup":
            count = noise_count(fs_input, default=8) if add_noise else 0
            items = [gold]
            items.extend(spec_rows)
            for idx in range(count):
                items.append({key: f"{key}_{idx}" for key in gold.keys()})
            payload: Any = items
        else:
            payload = dict(gold)
            document_meta = _document_meta(fs_input)
            if document_meta:
                payload["_document"] = document_meta
            if add_noise:
                payload["_meta"] = {"source": "fixture_builder", "noise_items": noise_count(fs_input, default=5)}
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


class JsonlGenerator(BaseFixtureGenerator):
    supported_types = ("jsonl",)

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        gold = ensure_mapping(fs_input.get("gold_read_result"))
        rows = [json.dumps(gold, ensure_ascii=False)]
        for row in _spec_rows(fs_input):
            rows.append(json.dumps(row, ensure_ascii=False))
        count = noise_count(fs_input, default=8) if add_noise else 0
        for idx in range(count):
            rows.append(json.dumps({key: f"{key}_{idx}" for key in gold.keys()}, ensure_ascii=False))
        return ("\n".join(rows) + "\n").encode("utf-8")


class YamlLikeGenerator(BaseFixtureGenerator):
    supported_types = ("yaml", "yml", "xml", "ini", "toml")

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        file_type = str(fs_input.get("file_type", "") or "").strip().lower()
        gold = ensure_mapping(fs_input.get("gold_read_result"))
        document_meta = _document_meta(fs_input)
        if document_meta:
            gold = {"document": document_meta, "payload": gold}
        if add_noise:
            gold = dict(gold)
            gold.setdefault("_meta", {"noise_level": (fs_input.get("content_spec", {}) or {}).get("noise_level", "medium")})
        if file_type in {"yaml", "yml"}:
            return (yaml_dump_simple(gold) + "\n").encode("utf-8")
        if file_type == "xml":
            return xml_from_mapping("root", gold)
        if file_type == "ini":
            from .base import ini_dump_simple

            return ini_dump_simple(gold).encode("utf-8")
        if file_type == "toml":
            from .base import toml_dump_simple

            return toml_dump_simple(gold).encode("utf-8")
        return json.dumps(gold, ensure_ascii=False, indent=2).encode("utf-8")


class XlsxGenerator(BaseFixtureGenerator):
    supported_types = ("xlsx",)

    def build_bytes(self, fs_input: Dict[str, Any], add_noise: bool = True) -> bytes:
        gold = ensure_mapping(fs_input.get("gold_read_result"))
        spec_rows = _spec_rows(fs_input)
        field_candidates = list(gold.keys())
        for row in spec_rows:
            field_candidates.extend(row.keys())
        fields = schema_hint(fs_input, field_candidates)
        rows: List[List[Any]] = [fields, [gold.get(field, "") for field in fields]]
        for row in spec_rows:
            rows.append([row.get(field, gold.get(field, "")) for field in fields])
        count = noise_count(fs_input, default=10) if add_noise else 0
        for idx in range(count):
            rows.append([f"{field}_{idx}" for field in fields])
        return build_minimal_xlsx(rows)
