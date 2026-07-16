"""
fixture_builder：根据 task 中声明的 fs_inputs 构造真实文件。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
from typing import Any, Dict, List

from .generators import get_generator


def _safe_name(text: str) -> str:
    raw = str(text or "").strip().replace("\\", "/").strip("/")
    raw = raw.replace("/", "_").replace(" ", "_")
    return raw or "unknown"


def extract_fs_inputs(task_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(task_obj, dict):
        return []
    direct = task_obj.get("fs_inputs")
    if isinstance(direct, list):
        return [deepcopy(x) for x in direct if isinstance(x, dict)]
    initial_intent_data = task_obj.get("initial_intent_data", {}) if isinstance(task_obj.get("initial_intent_data", {}), dict) else {}
    nested = initial_intent_data.get("fs_inputs")
    if isinstance(nested, list):
        return [deepcopy(x) for x in nested if isinstance(x, dict)]
    return []


def _task_fixture_root(task_obj: Dict[str, Any], fixture_root_dir: str) -> Path:
    env_id = _safe_name(task_obj.get("env_id", "env_unknown"))
    task_id = _safe_name(task_obj.get("task_id", "task_unknown"))
    return Path(fixture_root_dir).expanduser().resolve() / env_id / task_id


def _default_rel_path(fs_input: Dict[str, Any], index: int) -> str:
    input_id = str(fs_input.get("input_id", "")).strip() or f"input_{index}"
    file_type = str(fs_input.get("file_type", "")).strip().lower().lstrip(".")
    name = _safe_name(Path(input_id).name)
    if file_type and Path(name).suffix.lower() != f".{file_type}":
        name = f"{name}.{file_type}"
    return name or f"input_{index}"


def _normalize_rel_path(fs_input: Dict[str, Any], index: int) -> str:
    rel_path = str(fs_input.get("path_hint", "")).strip().replace("\\", "/").lstrip("/")
    if not rel_path:
        rel_path = _default_rel_path(fs_input, index)
    return rel_path


def _clear_directory_contents(target_root: Path) -> int:
    if not target_root.exists():
        return 0
    removed_count = 0
    for child in list(target_root.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed_count += 1
    return removed_count


def materialize_fs_inputs(
    fs_inputs: List[Dict[str, Any]],
    target_root_dir: str,
    overwrite: bool = False,
    add_noise: bool = True,
    clear_root: bool = False,
) -> Dict[str, Any]:
    target_root = Path(target_root_dir).expanduser().resolve()
    cleared_entries = 0
    if clear_root:
        cleared_entries = _clear_directory_contents(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    if not fs_inputs:
        return {
            "fixture_root_dir": str(target_root),
            "build_success": True,
            "build_skipped": True,
            "generated_files": [],
            "build_logs": (["cleared_task_dir"] if clear_root else []) + ["no_fs_inputs"],
            "cleared_entry_count": cleared_entries,
        }

    generated_files = []
    build_logs = []
    overall_success = True
    for index, fs_input in enumerate(fs_inputs, 1):
        current = deepcopy(fs_input)
        input_id = str(current.get("input_id", "")).strip() or f"input_{index}"
        rel_path = _normalize_rel_path(current, index)
        target_path = (target_root / rel_path).resolve()
        if target_root not in target_path.parents and target_path != target_root:
            overall_success = False
            build_logs.append(f"{input_id}: invalid_path")
            generated_files.append(
                {
                    "input_id": input_id,
                    "path": str(target_path),
                    "file_type": str(current.get("file_type", "") or "").strip(),
                    "generator_name": "",
                    "gold_embedded": False,
                    "exists": False,
                    "error": "invalid_path",
                }
            )
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and not bool(overwrite):
            generated_files.append(
                {
                    "input_id": input_id,
                    "path": str(target_path),
                    "relative_path": str(target_path.relative_to(target_root)),
                    "file_type": str(current.get("file_type", "") or "").strip(),
                    "generator_name": "reuse_existing",
                    "gold_embedded": True,
                    "exists": True,
                }
            )
            continue
        try:
            generator = get_generator(current)
            generator.write(target_path, current, add_noise=bool(add_noise))
            generated_files.append(
                {
                    "input_id": input_id,
                    "path": str(target_path),
                    "relative_path": str(target_path.relative_to(target_root)),
                    "file_type": str(current.get("file_type", "") or "").strip(),
                    "generator_name": generator.__class__.__name__,
                    "gold_embedded": True,
                    "exists": target_path.exists(),
                }
            )
        except Exception as e:
            overall_success = False
            build_logs.append(f"{input_id}: {type(e).__name__}: {e}")
            generated_files.append(
                {
                    "input_id": input_id,
                    "path": str(target_path),
                    "file_type": str(current.get("file_type", "") or "").strip(),
                    "generator_name": "",
                    "gold_embedded": False,
                    "exists": target_path.exists(),
                    "error": f"{type(e).__name__}: {e}",
                }
            )
    return {
        "fixture_root_dir": str(target_root),
        "build_success": overall_success,
        "build_skipped": False,
        "generated_files": generated_files,
        "build_logs": build_logs,
        "cleared_entry_count": cleared_entries,
    }


def bootstrap_fs_inputs_for_task(
    task_obj: Dict[str, Any],
    sandbox_root_dir: str,
    overwrite: bool = True,
) -> Dict[str, Any]:
    fs_inputs = extract_fs_inputs(task_obj)
    return materialize_fs_inputs(
        fs_inputs=fs_inputs,
        target_root_dir=sandbox_root_dir,
        overwrite=overwrite,
        add_noise=False,
        clear_root=bool(overwrite),
    )


def build_fs_fixtures_for_task(
    task_obj: Dict[str, Any],
    fixture_root_dir: str,
    overwrite: bool = False,
    add_noise: bool = True,
) -> Dict[str, Any]:
    fs_inputs = extract_fs_inputs(task_obj)
    target_root = _task_fixture_root(task_obj, fixture_root_dir)
    return materialize_fs_inputs(
        fs_inputs=fs_inputs,
        target_root_dir=str(target_root),
        overwrite=overwrite,
        add_noise=add_noise,
        clear_root=bool(overwrite),
    )
