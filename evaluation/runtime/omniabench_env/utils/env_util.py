"""
Utility functions for environment initialization and state management.
"""
import sys
import types
from copy import deepcopy
from pathlib import Path


def _safe_path_segment(text: str, fallback: str) -> str:
    raw = str(text or "").strip().replace("\\", "/").strip("/")
    raw = raw.replace("/", "_").replace(" ", "_")
    return raw or fallback


def _resolve_task_scoped_fs_root(base_dir: str, env_id: str, task_id: str) -> Path:
    base_path = Path(base_dir).expanduser().resolve()
    safe_env_id = _safe_path_segment(env_id, "env_unknown")
    safe_task_id = _safe_path_segment(task_id, "task_unknown")

    if base_path.name == safe_task_id and base_path.parent.name == safe_env_id:
        return base_path

    candidate_paths = []
    if base_path.name == "fs_fixtures":
        candidate_paths.append(base_path / safe_env_id / safe_task_id)
    if base_path.name == "tmp" or (base_path / "fs_fixtures").exists():
        candidate_paths.append(base_path / "fs_fixtures" / safe_env_id / safe_task_id)
    candidate_paths.append(base_path / safe_env_id / safe_task_id)

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return candidate_paths[0]


def _directory_has_entries(target_dir: str) -> bool:
    try:
        target_path = Path(target_dir).expanduser().resolve()
        return target_path.exists() and target_path.is_dir() and any(target_path.iterdir())
    except Exception:
        return False


def _resolve_runtime_root() -> Path | None:
    """向上查找包含 adapter.py 与 adapters/ 的 runtime 根目录。"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "adapter.py").exists() and (parent / "adapters").exists():
            return parent
    return None


def ensure_runtime_adapter_imports() -> Path | None:
    """Ensure local runtime adapter modules are importable for dynamic env code."""
    runtime_root = _resolve_runtime_root()
    if runtime_root is None:
        return None
    runtime_root_str = str(runtime_root)
    if runtime_root_str not in sys.path:
        sys.path.insert(0, runtime_root_str)
    return runtime_root


def init_env_class(env_class_code: str, env_class_name: str):
    """
    Initialize an environment class from source code string.
    
    :param env_class_code: Python source code string of the environment class
    :param env_class_name: Name of the class (must be defined in the code)
    :return: Environment class object
    """
    runtime_root = ensure_runtime_adapter_imports()
    module = types.ModuleType("dynamic_env")
    if runtime_root is not None:
        module.__dict__["__file__"] = str(runtime_root / "dynamic_env.py")
    exec(env_class_code, module.__dict__)
    
    if not hasattr(module, env_class_name):
        raise ValueError(f"Class '{env_class_name}' not found in provided env_class_code.")
    
    return getattr(module, env_class_name)


def init_env_instance(env_class, init_config=None):
    """
    Create an environment instance from class and apply initial config.
    
    :param env_class: Environment class object
    :param init_config: Optional dict for initial attribute configuration
    :return: Environment instance object
    """
    init_config = deepcopy(init_config)
    try:
        # Try constructor with dict argument
        if init_config and isinstance(init_config, dict):
            env_instance = env_class(init_config)
        else:
            env_instance = env_class({})
    except TypeError:
        # Constructor with no arguments
        env_instance = env_class()
    
    if init_config:
        for key, value in init_config.items():
            setattr(env_instance, key, value)
    
    return env_instance


def prepare_env_instance_runtime(env_instance, task_item=None, env_item=None, env_id: str = ""):
    """
    Attach runtime context for adapter-backed strategy tools.

    This keeps OmniaBench evaluation aligned with the shared FS / code runtime:
    - `from adapter import ...` remains importable;
    - fs tools get sandbox config and optional bootstrap files;
    - code strategy can call `python_executor` through the shared adapter shim.
    """
    ensure_runtime_adapter_imports()
    task_obj = task_item if isinstance(task_item, dict) else {}
    env_obj = env_item if isinstance(env_item, dict) else {}

    normalized_env_id = str(env_id or task_obj.get("env_id") or env_obj.get("env_id") or "").strip()
    task_id = str(task_obj.get("task_id") or "").strip()
    if normalized_env_id:
        setattr(env_instance, "env_id", normalized_env_id)
    if task_id:
        setattr(env_instance, "task_id", task_id)

    runtime_cfg = getattr(env_instance, "_adapter_runtime", None)
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}

    raw_strategy = task_obj.get("strategy_config")
    if not isinstance(raw_strategy, dict):
        raw_strategy = env_obj.get("strategy_config")
    if not isinstance(raw_strategy, dict):
        raw_strategy = {}

    import os
    import json
    override_str = os.getenv("OMNIABENCH_STRATEGY_OVERRIDE", "")
    fs_bundle_root = str(os.getenv("OMNIABENCH_FS_BUNDLE_ROOT", "")).strip()
    fs_tmp_root_override = str(os.getenv("OMNIABENCH_FS_TMP_ROOT", "")).strip()
    if override_str:
        try:
            override_dict = json.loads(override_str)
            if isinstance(override_dict, dict):
                for k, v in override_dict.items():
                    if k not in raw_strategy:
                        raw_strategy[k] = {}
                    if isinstance(v, dict):
                        raw_strategy[k].update(v)
        except Exception as e:
            print(f"[Warning] Failed to parse OMNIABENCH_STRATEGY_OVERRIDE: {e}")

    normalized_strategy = {}
    try:
        from adapters.strategy_config import normalize_strategy_config

        normalized_strategy = normalize_strategy_config({"strategy_config": raw_strategy})
    except Exception:
        normalized_strategy = deepcopy(raw_strategy) if isinstance(raw_strategy, dict) else {}

    fs_inputs = task_obj.get("fs_inputs", [])
    fs_bootstrap_meta = task_obj.get("fs_runtime_bootstrap")
    if not isinstance(fs_bootstrap_meta, dict):
        fs_bootstrap_meta = task_obj.get("bootstrap_info")
    fs_tool_present = False
    for tool in env_obj.get("tools", []) if isinstance(env_obj.get("tools", []), list) else []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", {}) if isinstance(tool.get("function", {}), dict) else {}
        tool_name = str(fn.get("name") or tool.get("name") or "").strip()
        if tool_name.startswith("fs_"):
            fs_tool_present = True
            break

    code_tool_present = False
    for tool in env_obj.get("tools", []) if isinstance(env_obj.get("tools", []), list) else []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", {}) if isinstance(tool.get("function", {}), dict) else {}
        tool_name = str(fn.get("name") or tool.get("name") or "").strip()
        if tool_name == "python_executor":
            code_tool_present = True
            break

    fs_cfg = normalized_strategy.get("fs", {}) if isinstance(normalized_strategy.get("fs", {}), dict) else {}
    fs_enabled = bool(fs_cfg.get("enabled", False)) or bool(fs_tool_present) or bool(fs_inputs) or isinstance(fs_bootstrap_meta, dict)
    if fs_enabled:
        sandbox_root = ""
        if fs_bundle_root and normalized_env_id and task_id:
            sandbox_root = str(_resolve_task_scoped_fs_root(fs_bundle_root, normalized_env_id, task_id))
        elif isinstance(fs_bootstrap_meta, dict):
            sandbox_root = str(
                fs_bootstrap_meta.get("sandbox_root_dir")
                or fs_bootstrap_meta.get("fixture_root_dir")
                or ""
            ).strip()
        if not sandbox_root:
            tmp_root = str(fs_tmp_root_override or fs_cfg.get("tmp_root", "")).strip()
            if tmp_root and normalized_env_id and task_id:
                sandbox_root = str(_resolve_task_scoped_fs_root(tmp_root, normalized_env_id, task_id))
            elif tmp_root:
                sandbox_root = str(Path(tmp_root).expanduser().resolve())

        runtime_fs_cfg = runtime_cfg.setdefault("fs", {})
        if sandbox_root:
            runtime_fs_cfg["tmp_root"] = sandbox_root
            runtime_fs_cfg["bundle_root"] = fs_bundle_root
        if fs_cfg:
            runtime_fs_cfg["max_file_size_bytes"] = int(fs_cfg.get("max_file_size_bytes", 262144))
            runtime_fs_cfg["max_list_entries"] = int(fs_cfg.get("max_list_entries", 200))

        if sandbox_root and bool(fs_inputs):
            try:
                from adapters.fs.fixture_builder import bootstrap_fs_inputs_for_task

                if fs_bundle_root and _directory_has_entries(sandbox_root):
                    bootstrap_info = {
                        "enabled": True,
                        "build_success": True,
                        "build_skipped": True,
                        "reused_existing_bundle": True,
                        "fixture_root_dir": sandbox_root,
                        "sandbox_root_dir": sandbox_root,
                        "generated_files": [],
                        "build_logs": ["reused_existing_bundle_root"],
                    }
                else:
                    bootstrap_info = bootstrap_fs_inputs_for_task(
                        task_obj=task_obj,
                        sandbox_root_dir=sandbox_root,
                        overwrite=True,
                    )
                runtime_fs_cfg["bootstrap_info"] = bootstrap_info if isinstance(bootstrap_info, dict) else {}
            except Exception as e:
                runtime_fs_cfg["bootstrap_info"] = {
                    "enabled": True,
                    "build_skipped": True,
                    "reason": f"omniabench_runtime_bootstrap_failed: {e}",
                    "generated_files": [],
                }

    code_cfg = normalized_strategy.get("code", {}) if isinstance(normalized_strategy.get("code", {}), dict) else {}
    if bool(code_cfg.get("enabled", False)) or bool(code_tool_present):
        runtime_cfg["code"] = deepcopy(code_cfg)

    if normalized_strategy:
        runtime_cfg["strategy_config"] = deepcopy(normalized_strategy)
    setattr(env_instance, "_adapter_runtime", runtime_cfg)



def get_state_diff(old_state: dict, new_state: dict, ignore_keys: list = []) -> dict:
    """
    Compare two state dictionaries and return differences:
    - Added keys
    - Removed keys
    - Changed values
    
    Recursively compares dict values.
    """
    old_state = deepcopy(old_state)
    new_state = deepcopy(new_state)
    diff_result = {}

    # Find union of all keys
    all_keys = set(old_state.keys()) | set(new_state.keys())

    for key in all_keys:
        old_val = old_state.get(key)
        new_val = new_state.get(key)

        if key not in old_state: # Added key
            diff_result[key] = {"added": new_val}
        elif key not in new_state: # Removed key
            diff_result[key] = {"removed": old_val}
        else: # Both exist, compare values
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                # Recursive comparison for dicts
                sub_diff = get_state_diff(old_val, new_val)
                if sub_diff:  # Record only if there are changes
                    diff_result[key] = sub_diff
            else:
                # Simple type comparison
                if old_val != new_val:
                    diff_result[key] = {"changed": {"old":old_val, "new":new_val}}
                    
    # Remove ignored keys
    for key in ignore_keys:
        if key in diff_result:
            del diff_result[key]

    return deepcopy(diff_result)


def get_state_info(env_instance):
    """Return state dictionary of environment instance (excluding built-in attributes)."""
    return deepcopy({
        k: v for k, v in vars(env_instance).items()
        if not (k.startswith("__") and k.endswith("__"))
    })


def run_check_function(func_code: str, init_state: dict, final_state: dict):
    """
    Dynamically execute a verification function defined in func_code.
    """
    safe_globals = {
        '__builtins__': __builtins__,
    }
    safe_globals.update({"initial_state": deepcopy(init_state)})

    try:
        # Execute in safe_globals, function will retain this global scope
        exec(func_code, safe_globals)

        if 'check_func' not in safe_globals:
            return False, None, "Function 'check_func' not found."

        result = safe_globals['check_func'](final_state)

        if not isinstance(result, bool):
            print("Function did not return a boolean. Result: {result}")
            return False, None, "Function did not return a boolean."

        return True, result, None
    except Exception as e:
        print("Error:", e)
        return False, None, str(e)
