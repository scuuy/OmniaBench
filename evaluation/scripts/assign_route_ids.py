#!/usr/bin/env python3
"""Deterministically assign stable global_ids to a route's task_items.

Routes 2 and 3 arrive as standalone JSON files that may not carry global_id.
We assign ids deterministically so that the *same* item always receives the
*same* id across runs (stable for later analysis), regardless of the order the
items appear in the source file.

Strategy:
- route with id_passthrough=True (route 1): keep existing global_id untouched;
  only fill missing ones by deterministic sort offset from id_base.
- route with id_passthrough=False (routes 2/3): always (re)assign global_id by
  sorting on (env_id, task_id, lang) then enumerating from id_base.

The function is import-friendly so the orchestrator can call it directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _stable_sort_key(item: dict) -> tuple:
    env_id = str(item.get("env_id") or "")
    task_id = str(item.get("task_id") or "")
    lang = str(item.get("lang") or "")
    return (env_id, task_id, lang)


def assign_ids(
    task_items: list[dict],
    id_base: int,
    id_passthrough: bool,
) -> list[dict]:
    """Return a new list with global_id assigned deterministically."""
    items = [dict(it) for it in task_items if isinstance(it, dict)]
    ordered = sorted(enumerate(items), key=lambda pair: _stable_sort_key(pair[1]))

    if id_passthrough:
        next_offset = 0
        used = {
            int(it["global_id"])
            for it in items
            if isinstance(it.get("global_id"), int)
        }
        for _orig_idx, it in ordered:
            if isinstance(it.get("global_id"), int):
                continue
            candidate = int(id_base) + next_offset
            while candidate in used:
                next_offset += 1
                candidate = int(id_base) + next_offset
            it["global_id"] = candidate
            used.add(candidate)
            next_offset += 1
    else:
        for offset, (_orig_idx, it) in enumerate(ordered):
            it["global_id"] = int(id_base) + offset

    return items


def load_task_items(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        # tolerate accidental wrapped payloads
        data = data["samples"]
    if not isinstance(data, list):
        raise ValueError(f"task_items file must be a JSON list, got {type(data)}: {path}")
    return [it for it in data if isinstance(it, dict)]


def prepare_route_file(
    source_path: str,
    output_path: str,
    id_base: int,
    id_passthrough: bool,
) -> dict[str, Any]:
    items = load_task_items(source_path)
    assigned = assign_ids(items, id_base=id_base, id_passthrough=id_passthrough)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(assigned, file, ensure_ascii=False, indent=2)
    gids = [it.get("global_id") for it in assigned]
    return {
        "source_path": source_path,
        "output_path": output_path,
        "item_count": len(assigned),
        "id_base": int(id_base),
        "id_passthrough": bool(id_passthrough),
        "min_global_id": min(gids) if gids else None,
        "max_global_id": max(gids) if gids else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministically assign global_ids to a route file.")
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--id-base", type=int, required=True)
    parser.add_argument("--id-passthrough", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    info = prepare_route_file(
        source_path=args.source_path,
        output_path=args.output_path,
        id_base=args.id_base,
        id_passthrough=bool(args.id_passthrough),
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
