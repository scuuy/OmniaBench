"""
策略配置解析。

目标：
- 为 search / fs / code 三类策略提供统一开关与默认参数；
- 保持旧配置兼容：未配置时全部默认关闭；
- 允许旧入口直接传平铺字段，也允许传 `strategy_config`。
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


RUNTIME_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TMP_ROOT = str(RUNTIME_ROOT / "tmp")


DEFAULT_STRATEGY_CONFIG: Dict[str, Any] = {
    "search": {
        "enabled": False,
        "provider": "openai",
        "model": "",
        "llm_api_url": "",
        "api_key_env_var": "",
        "generator_model": "",
        "generator_llm_api_url": "",
        "generator_api_key_env_var": "",
        "temperature": 0.2,
        "max_workers": 8,
        "max_try": 2,
        "target_new_items": 60,
        "generation_batch_size": 20,
        "max_generation_rounds": 4,
        "example_per_container": 3,
    },
    "fs": {
        "enabled": False,
        "root_dir": str(Path(DEFAULT_TMP_ROOT) / "fs_fixtures"),
        "tmp_root": DEFAULT_TMP_ROOT,
        "fixture_root_dir": str(Path(DEFAULT_TMP_ROOT) / "fs_fixtures"),
        "max_file_size_bytes": 262144,
        "max_list_entries": 200,
        "max_python_visible_bytes": 65536,
        "overwrite_fixtures": False,
        "add_noise": True,
    },
    "code": {
        "enabled": False,
        "min_python_steps_per_dag": 1,
        "max_python_steps_per_dag": 2,
        "execution_timeout_seconds": 8,
        "max_stdout_chars": 12000,
        "max_stderr_chars": 12000,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def normalize_strategy_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    raw = config if isinstance(config, dict) else {}
    raw_strategy = raw.get("strategy_config", {}) if isinstance(raw.get("strategy_config", {}), dict) else {}
    raw_fs = raw_strategy.get("fs", {}) if isinstance(raw_strategy.get("fs", {}), dict) else {}
    merged = _deep_merge(DEFAULT_STRATEGY_CONFIG, raw_strategy)

    # 兼容旧式平铺配置。
    if "search_enhancement_enabled" in raw:
        merged["search"]["enabled"] = bool(raw.get("search_enhancement_enabled"))
    if raw.get("search_enhancement_model"):
        merged["search"]["model"] = str(raw.get("search_enhancement_model"))
    if raw.get("search_enhancement_api_url"):
        merged["search"]["llm_api_url"] = str(raw.get("search_enhancement_api_url"))
    if raw.get("search_enhancement_api_key_env_var"):
        merged["search"]["api_key_env_var"] = str(raw.get("search_enhancement_api_key_env_var"))
    if raw.get("search_enhancement_provider"):
        merged["search"]["provider"] = str(raw.get("search_enhancement_provider"))
    if raw.get("search_enhancement_max_workers") is not None:
        merged["search"]["max_workers"] = int(raw.get("search_enhancement_max_workers"))
    if raw.get("search_enhancement_generator_model"):
        merged["search"]["generator_model"] = str(raw.get("search_enhancement_generator_model"))
    if raw.get("search_enhancement_generator_api_url"):
        merged["search"]["generator_llm_api_url"] = str(raw.get("search_enhancement_generator_api_url"))
    if raw.get("search_enhancement_generator_api_key_env_var"):
        merged["search"]["generator_api_key_env_var"] = str(raw.get("search_enhancement_generator_api_key_env_var"))
    if raw.get("search_enhancement_target_new_items") is not None:
        merged["search"]["target_new_items"] = int(raw.get("search_enhancement_target_new_items"))
    if raw.get("search_enhancement_generation_batch_size") is not None:
        merged["search"]["generation_batch_size"] = int(raw.get("search_enhancement_generation_batch_size"))
    if raw.get("search_enhancement_max_generation_rounds") is not None:
        merged["search"]["max_generation_rounds"] = int(raw.get("search_enhancement_max_generation_rounds"))

    if "fs_strategy_enabled" in raw:
        merged["fs"]["enabled"] = bool(raw.get("fs_strategy_enabled"))
    if raw.get("fs_root_dir"):
        merged["fs"]["root_dir"] = str(raw.get("fs_root_dir"))
    if raw.get("fs_tmp_root"):
        merged["fs"]["tmp_root"] = str(raw.get("fs_tmp_root"))
    if raw.get("fs_fixture_root_dir"):
        merged["fs"]["fixture_root_dir"] = str(raw.get("fs_fixture_root_dir"))

    explicit_root_dir = bool(raw.get("fs_root_dir")) or bool(raw_fs.get("root_dir"))
    explicit_tmp_root = bool(raw.get("fs_tmp_root")) or bool(raw_fs.get("tmp_root"))
    explicit_fixture_root = bool(raw.get("fs_fixture_root_dir")) or bool(raw_fs.get("fixture_root_dir"))
    root_dir = str(merged["fs"].get("root_dir") or merged["fs"].get("fixture_root_dir") or Path(DEFAULT_TMP_ROOT) / "fs_fixtures")
    merged["fs"]["root_dir"] = root_dir
    if explicit_root_dir and not explicit_fixture_root:
        merged["fs"]["fixture_root_dir"] = root_dir
    if explicit_root_dir and not explicit_tmp_root:
        merged["fs"]["tmp_root"] = str(Path(root_dir) / "_runtime")

    if "code_strategy_enabled" in raw:
        merged["code"]["enabled"] = bool(raw.get("code_strategy_enabled"))
    if raw.get("code_strategy_min_python_steps_per_dag") is not None:
        merged["code"]["min_python_steps_per_dag"] = int(raw.get("code_strategy_min_python_steps_per_dag"))
    if raw.get("code_strategy_max_python_steps_per_dag") is not None:
        merged["code"]["max_python_steps_per_dag"] = int(raw.get("code_strategy_max_python_steps_per_dag"))

    merged["fs"]["root_dir"] = str(Path(merged["fs"]["root_dir"]).expanduser())
    merged["fs"]["tmp_root"] = str(Path(merged["fs"]["tmp_root"]).expanduser())
    merged["fs"]["fixture_root_dir"] = str(Path(merged["fs"]["fixture_root_dir"]).expanduser())
    merged["search"]["temperature"] = float(merged["search"].get("temperature", 0.2))
    merged["search"]["max_workers"] = max(1, int(merged["search"].get("max_workers", 8)))
    merged["search"]["max_try"] = max(1, int(merged["search"].get("max_try", 2)))
    merged["search"]["target_new_items"] = max(1, int(merged["search"].get("target_new_items", 60)))
    merged["search"]["generation_batch_size"] = max(1, int(merged["search"].get("generation_batch_size", 20)))
    merged["search"]["max_generation_rounds"] = max(1, int(merged["search"].get("max_generation_rounds", 4)))
    merged["search"]["example_per_container"] = max(1, int(merged["search"].get("example_per_container", 3)))
    merged["code"]["min_python_steps_per_dag"] = max(0, int(merged["code"].get("min_python_steps_per_dag", 1)))
    merged["code"]["max_python_steps_per_dag"] = max(
        merged["code"]["min_python_steps_per_dag"],
        int(merged["code"].get("max_python_steps_per_dag", 2)),
    )
    merged["code"]["execution_timeout_seconds"] = max(1, int(merged["code"].get("execution_timeout_seconds", 8)))
    return merged


def is_search_strategy_enabled(config: Dict[str, Any] | None) -> bool:
    return bool(normalize_strategy_config(config)["search"]["enabled"])


def is_fs_strategy_enabled(config: Dict[str, Any] | None) -> bool:
    return bool(normalize_strategy_config(config)["fs"]["enabled"])


def is_code_strategy_enabled(config: Dict[str, Any] | None) -> bool:
    return bool(normalize_strategy_config(config)["code"]["enabled"])


def resolve_search_model_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    raw = config if isinstance(config, dict) else {}
    strategy = normalize_strategy_config(raw)["search"]
    model = str(strategy.get("model") or raw.get("model") or "").strip()
    api_url = str(strategy.get("llm_api_url") or raw.get("llm_api_url") or "").strip()
    api_key_env_var = str(strategy.get("api_key_env_var") or raw.get("api_key_env_var") or "OPENAI_API_KEY").strip()
    api_key = os.getenv(api_key_env_var, "")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    generator_api_key_env_var = str(strategy.get("generator_api_key_env_var") or api_key_env_var).strip()
    generator_api_key = os.getenv(generator_api_key_env_var, "")
    if not generator_api_key:
        generator_api_key = api_key
    return {
        "provider": str(strategy.get("provider") or raw.get("provider") or "openai"),
        "model": model,
        "llm_api_url": api_url,
        "api_key_env_var": api_key_env_var,
        "api_key": api_key,
        "generator_model": str(strategy.get("generator_model") or model or "").strip(),
        "generator_llm_api_url": str(strategy.get("generator_llm_api_url") or api_url or "").strip(),
        "generator_api_key_env_var": generator_api_key_env_var,
        "generator_api_key": generator_api_key,
        "temperature": float(strategy.get("temperature", 0.2)),
        "max_workers": int(strategy.get("max_workers", 8)),
        "max_try": int(strategy.get("max_try", 2)),
        "target_new_items": int(strategy.get("target_new_items", 60)),
        "generation_batch_size": int(strategy.get("generation_batch_size", 20)),
        "max_generation_rounds": int(strategy.get("max_generation_rounds", 4)),
        "example_per_container": int(strategy.get("example_per_container", 3)),
    }
