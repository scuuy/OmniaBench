"""final_eval 的统一评测入口。

特点：
- 只依赖 final_eval 本地整理后的 runtime 与 data 目录；
- 用 CLI 参数替代旧版脚本中的硬编码路径；
- 支持按 batch 名或直接 task_items_path 运行；
- 支持 run_and_eval / eval_only 两种模式；
- 默认输出到 final_eval/results。
"""

from __future__ import annotations

import argparse
import datetime
import inspect
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_EVAL_ROOT = SCRIPT_DIR.parent
RUNTIME_DIR = FINAL_EVAL_ROOT / "runtime"
CONFIG_DIR = FINAL_EVAL_ROOT / "configs"
RESULTS_DIR = FINAL_EVAL_ROOT / "results"

if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from agent.task_solve_agent import TaskSolveAgent  # noqa: E402
from omniabench_env import (  # noqa: E402
    OmniaBenchConvRLEnv,
    OmniaBenchConvSFTEnv,
    OmniaBenchNonConvRLEnv,
    OmniaBenchNonConvSFTEnv,
)
from omniabench_env.utils.user_agent import (  # noqa: E402
    build_agent_followup_hint,
    normalize_user_difficulty_config,
)
from result_evaluator import (  # noqa: E402
    augment_result_with_evaluations,
    evaluate_result_file,
    load_task_lookup,
    summarize_pass_k_group,
)


ENV_CLS_MAP = {
    "omniabench_conversation_rl": OmniaBenchConvRLEnv,
    "omniabench_non_conversation_rl": OmniaBenchNonConvRLEnv,
    "omniabench_conversation_sft": OmniaBenchConvSFTEnv,
    "omniabench_non_conversation_sft": OmniaBenchNonConvSFTEnv,
}

MAX_STEPS_MAP = {
    "omniabench_conversation_rl": 200,
    "omniabench_non_conversation_rl": 200,
    "omniabench_conversation_sft": 200,
    "omniabench_non_conversation_sft": 200,
}


# ============ Colored Logging Utilities ============
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"
    BG_DARK_GRAY = "\033[100m"

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        cls.RESET = cls.BOLD = cls.DIM = cls.UNDERLINE = ""
        cls.BLACK = cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = ""
        cls.MAGENTA = cls.CYAN = cls.WHITE = ""
        cls.BRIGHT_RED = cls.BRIGHT_GREEN = cls.BRIGHT_YELLOW = ""
        cls.BRIGHT_BLUE = cls.BRIGHT_MAGENTA = cls.BRIGHT_CYAN = ""
        cls.BG_RED = cls.BG_GREEN = cls.BG_BLUE = cls.BG_DARK_GRAY = ""


# Check if output supports colors
if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
    Colors.disable()


# ASCII icons (compatible with all terminals)
ICONS = {
    'success': '[OK]',
    'error': '[ERR]',
    'warning': '[!]',
    'info': '[i]',
    'running': '[:]',
    'pending': '[ ]',
    'complete': '[*]',
    'star': '*',
    'bullet': '-',
    'arrow': '->',
    'clock': '',
    'check': '[V]',
    'cross': '[X]',
    'fire': '',
    'target': '',
    'chart': '',
    'rocket': '',
}


def c(text: str, *colors: str) -> str:
    """Apply colors to text."""
    return "".join(colors) + str(text) + Colors.RESET


def format_box(title: str, content: str, color: str = Colors.BRIGHT_BLUE, width: int = 80) -> str:
    """Format content in a colored box."""
    lines = content.split('\n')
    max_line_len = max(len(line) for line in lines) if lines else 0
    box_width = max(width, min(max_line_len + 4, 100))

    horizontal = "─" * (box_width - 2)
    corners = {"tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│"}

    result = []
    # Top border with title
    if title:
        title_str = f" {title} "
        title_len = len(title_str)
        left_len = (box_width - title_len) // 2
        right_len = box_width - title_len - left_len - 2
        result.append(f"{color}{corners['tl']}{horizontal * left_len}{title_str}{horizontal * right_len}{corners['tr']}{Colors.RESET}")
    else:
        result.append(f"{color}{corners['tl']}{horizontal}{corners['tr']}{Colors.RESET}")

    # Content lines
    for line in lines:
        padded = line.ljust(box_width - 2)
        result.append(f"{color}{corners['v']}{Colors.RESET} {padded} {color}{corners['v']}{Colors.RESET}")

    # Bottom border
    result.append(f"{color}{corners['bl']}{horizontal}{corners['br']}{Colors.RESET}")

    return '\n'.join(result)


def format_progress_bar(current: int, total: int, width: int = 30, color: str = Colors.BRIGHT_GREEN) -> str:
    """Format a progress bar with percentage."""
    percentage = min(100, int(100 * current / max(1, total)))
    filled = int(width * current / max(1, total))
    empty = width - filled

    bar = f"{color}{'█' * filled}{Colors.DIM}{'░' * empty}{Colors.RESET}"
    return f"{bar} {percentage:3d}%"


def format_key_value(key: str, value: str, key_color: str = Colors.CYAN, value_color: str = Colors.WHITE) -> str:
    """Format a key-value pair."""
    return f"{key_color}{key}:{Colors.RESET} {value_color}{value}{Colors.RESET}"


def get_current_time():
    """返回当前时间字符串。"""
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def get_current_utc_timestamp():
    """返回 UTC 时间戳字符串（naive ISO format）。"""
    return (
        datetime.datetime.now(tz=datetime.timezone.utc)
        .replace(tzinfo=None)
        .isoformat()
    )


def _resolve_config_value(raw: str, fallback_env_var: str = "", default: str = "") -> str:
    """解析直填值或环境变量名。"""
    text = str(raw or "").strip()
    if not text:
        return os.getenv(fallback_env_var, "") or default
    if text.startswith(("http://", "https://", "sk-")):
        return text
    from_env = os.getenv(text, "")
    if from_env:
        return from_env
    return os.getenv(fallback_env_var, "") or default


def _normalize_openai_base_url(url: str) -> str:
    """将可能带 /chat/completions 的 base_url 规整到 /v1。"""
    text = str(url or "").strip()
    if not text:
        return ""
    suffix = "/chat/completions"
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text


def _resolve_optional_path(raw_path: str) -> str:
    text = str(raw_path or "").strip()
    if not text:
        return ""
    return str(Path(text).expanduser().resolve())


def _apply_runtime_path_overrides(args) -> dict:
    applied = {}
    fs_bundle_root = _resolve_optional_path(getattr(args, "fs_bundle_root", ""))
    fs_tmp_root = _resolve_optional_path(getattr(args, "fs_tmp_root", ""))

    if fs_bundle_root:
        os.environ["OMNIABENCH_FS_BUNDLE_ROOT"] = fs_bundle_root
        applied["fs_bundle_root"] = fs_bundle_root
    if fs_tmp_root:
        os.environ["OMNIABENCH_FS_TMP_ROOT"] = fs_tmp_root
        applied["fs_tmp_root"] = fs_tmp_root
    return applied


def _make_json_serializable(obj, max_depth=10):
    """递归转换为可 JSON 序列化结构。"""
    if max_depth <= 0:
        return str(type(obj).__name__)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if callable(obj):
        module = str(getattr(obj, "__module__", "") or "")
        name = str(getattr(obj, "__name__", "") or getattr(obj, "__qualname__", "") or type(obj).__name__)
        return f"<callable:{module}.{name}>".strip(".")
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v, max_depth=max_depth - 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v, max_depth=max_depth - 1) for v in obj]
    if isinstance(obj, set):
        return [_make_json_serializable(v, max_depth=max_depth - 1) for v in sorted(obj, key=lambda x: str(x))]
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def save_json(path, data):
    """原子写入 JSON。"""
    target_path = str(path)
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{target_path}.tmp"
    serializable_data = _make_json_serializable(data)
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(serializable_data, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp_path, target_path)


def _extract_env_states(env):
    """从环境对象提取 init/final state。"""
    init_state = getattr(env, "init_state", None)
    final_state = getattr(env, "pred_final_state", None)
    if init_state is not None:
        init_state = _make_json_serializable(deepcopy(init_state))
    if final_state is not None:
        final_state = _make_json_serializable(deepcopy(final_state))
    return init_state, final_state


def _load_task_items(task_items_path):
    with open(task_items_path, "r", encoding="utf-8") as file:
        task_items = json.load(file)
    if not isinstance(task_items, list):
        raise ValueError(f"task_items_path 应为 list，实际为 {type(task_items)}")
    return task_items


def _parse_global_id_selector(selector: str):
    """解析 global_id 区间字符串，支持 all / 1-10 / 1-10,21-30 / 5。"""
    text = str(selector or "").strip()
    if not text:
        text = "1-10"
    if text.lower() == "all":
        return None
    selected_ids = set()
    for part in text.split(","):
        chunk = str(part or "").strip()
        if not chunk:
            continue
        if "-" in chunk:
            left, right = chunk.split("-", 1)
            start = int(left)
            end = int(right)
            if start > end:
                raise ValueError(f"global_id 区间非法: {chunk}")
            selected_ids.update(range(start, end + 1))
        else:
            selected_ids.add(int(chunk))
    if not selected_ids:
        raise ValueError("global_id 范围为空。")
    return selected_ids


def _resolve_task_ids(
    task_items_path,
    selected_task_ids=None,
    selected_task_keys=None,
    global_id_range="1-10",
    lang_filter="all",
):
    """解析最终要运行的 task 索引。"""
    task_items = _load_task_items(task_items_path)
    normalized_lang_filter = str(lang_filter or "all").strip().lower()
    if normalized_lang_filter not in {"all", "cn", "en"}:
        raise ValueError(f"lang_filter 非法: {lang_filter}")

    def _lang_matches(item):
        if normalized_lang_filter == "all":
            return True
        return str(item.get("lang", "")).strip().lower() == normalized_lang_filter

    all_task_ids = list(range(len(task_items)))
    if selected_task_ids is None and selected_task_keys is None:
        selected_global_ids = _parse_global_id_selector(global_id_range)
        if selected_global_ids is None:
            return [idx for idx, item in enumerate(task_items) if _lang_matches(item)]
        matched_task_ids = []
        for idx, item in enumerate(task_items):
            if not _lang_matches(item):
                continue
            try:
                global_id = int(item.get("global_id"))
            except Exception:
                continue
            if global_id in selected_global_ids:
                matched_task_ids.append(idx)
        return matched_task_ids

    normalized_task_ids = []
    seen = set()
    if selected_task_ids is not None:
        for raw_idx in selected_task_ids:
            task_idx = int(raw_idx)
            if task_idx < 0 or task_idx >= len(task_items):
                raise ValueError(f"selected_task_ids 越界: {task_idx}")
            if not _lang_matches(task_items[task_idx]):
                continue
            if task_idx in seen:
                continue
            seen.add(task_idx)
            normalized_task_ids.append(task_idx)

    if selected_task_keys is not None:
        key_to_indexes = {}
        for idx, item in enumerate(task_items):
            if not isinstance(item, dict):
                continue
            for raw_key in [item.get("task_key"), item.get("task_id"), item.get("source_task_id")]:
                key = str(raw_key or "").strip()
                if key:
                    key_to_indexes.setdefault(key, []).append(idx)
        for raw_key in selected_task_keys:
            key = str(raw_key or "").strip()
            matched_indexes = key_to_indexes.get(key, [])
            if not matched_indexes:
                raise ValueError(f"selected_task_keys 未命中: {key}")
            for task_idx in matched_indexes:
                if not _lang_matches(task_items[task_idx]):
                    continue
                if task_idx not in seen:
                    seen.add(task_idx)
                    normalized_task_ids.append(task_idx)
    return normalized_task_ids


def _build_task_index_to_task_id(task_items_path):
    """建立 task_index -> task_id 映射。"""
    with open(task_items_path, "r", encoding="utf-8") as file:
        task_items = json.load(file)
    index_to_id = {}
    if not isinstance(task_items, list):
        return index_to_id
    for idx, item in enumerate(task_items):
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id", "")).strip()
        if task_id:
            index_to_id[idx] = task_id
    return index_to_id


def _build_task_index_metadata(task_items_path):
    """建立 task_index -> metadata 映射。"""
    task_items = _load_task_items(task_items_path)
    metadata = {}
    for idx, item in enumerate(task_items):
        if not isinstance(item, dict):
            continue
        metadata[idx] = {
            "task_id": str(item.get("task_id", "")).strip(),
            "env_id": str(item.get("env_id", "")).strip(),
            "global_id": item.get("global_id"),
            "batch": str(item.get("batch", "")).strip(),
            "lang": str(item.get("lang", "")).strip(),
        }
    return metadata


def _build_single_run_payload(task_ids, result_by_run_key):
    """构造 pass_k=1 的输出。"""
    payload = []
    for task_index in task_ids:
        result_item = result_by_run_key.get((int(task_index), 1))
        if isinstance(result_item, dict):
            payload.append(deepcopy(result_item))
    return payload[0] if len(task_ids) == 1 and payload else payload


def _group_results_by_task_index(result_by_run_key):
    """将结果按 task_index 聚合。"""
    grouped = {}
    for (task_index, _sample_idx), result_item in result_by_run_key.items():
        grouped.setdefault(int(task_index), []).append(deepcopy(result_item))
    return grouped


def _build_grouped_result_payload(task_ids, task_index_to_id, result_groups, task_lookup, pass_k):
    """构造 pass_k>1 的输出。"""
    payload = []
    for task_index in task_ids:
        samples = result_groups.get(int(task_index), [])
        if not samples:
            continue
        first_sample = samples[0]
        task_info = first_sample.get("task_info", {}) if isinstance(first_sample.get("task_info"), dict) else {}
        actual_task_id = str(task_info.get("task_id") or task_index_to_id.get(task_index, "")).strip()
        task_item = deepcopy((task_lookup or {}).get(actual_task_id, {}))
        group_item = {
            "task_index": task_index,
            "task_id": actual_task_id,
            "env_id": task_info.get("env_id") or task_item.get("env_id"),
            "task": task_info.get("task") or task_item.get("task", ""),
            "task_info": deepcopy(task_info),
            "k": int(pass_k),
            "samples": deepcopy(samples),
        }
        payload.append(summarize_pass_k_group(group_item))
    return payload[0] if len(task_ids) == 1 and payload else payload


def _format_progress_bar(done_count, total_count, width=24):
    """格式化简单文本进度条。"""
    total = max(1, int(total_count))
    done = max(0, min(int(done_count), total))
    filled = int(width * done / total)
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def _count_completed_tasks(task_ids, completed_run_keys, pass_k):
    """统计已经完成全部 pass 轮次的 task 数量。"""
    completed_task_count = 0
    max_pass_k = max(1, int(pass_k))
    for task_index in task_ids:
        sample_set = {
            sample_idx
            for (completed_task_index, sample_idx) in completed_run_keys
            if int(completed_task_index) == int(task_index)
        }
        if len(sample_set) >= max_pass_k:
            completed_task_count += 1
    return completed_task_count


def _extract_run_key_from_result_item(item, fallback_task_index=None, fallback_sample_idx=None):
    """从结果 item 中稳健提取 (task_index, sample_idx)。

    兼容三种结构：
    1. 顶层字段：item["task_index"], item["sample_idx"]；
    2. task_info 字段：item["task_info"]["task_index"], item["task_info"]["sample_idx"]；
    3. pass_k 分组外层提供的 fallback_task_index。
    """
    if not isinstance(item, dict):
        return None

    task_info = item.get("task_info", {})
    if not isinstance(task_info, dict):
        task_info = {}

    task_index = item.get("task_index")
    if task_index is None:
        task_index = task_info.get("task_index")
    if task_index is None:
        task_index = fallback_task_index

    sample_idx = item.get("sample_idx")
    if sample_idx is None:
        sample_idx = task_info.get("sample_idx")
    if sample_idx is None:
        sample_idx = fallback_sample_idx
    if sample_idx is None:
        sample_idx = 1

    if task_index is None:
        return None

    try:
        return int(task_index), int(sample_idx)
    except Exception:
        return None


def _is_resume_completed_result(item, retry_failed=True):
    """判断历史结果是否应视作已完成。

    默认 retry_failed=True：
    - result_status == failed 的样本不会被跳过，会进入 pending_runs 重新跑；
    - termination_reason == INFRA_ERROR 的样本不会被跳过，会进入 pending_runs 重新跑。

    如果命令行传 --resume-keep-failed，则 retry_failed=False，失败样本也会被视为已完成。
    """
    if not isinstance(item, dict):
        return False

    if retry_failed:
        if str(item.get("result_status", "")).strip().lower() == "failed":
            return False
        if str(item.get("termination_reason", "")).strip().upper() == "INFRA_ERROR":
            return False

    return True


def _result_jsonl_path_from_save_file(save_file_path):
    """由最终聚合 JSON 路径推导运行中增量 JSONL 路径。

    例如：xxx.json -> xxx.runs.jsonl
    """
    path = Path(str(save_file_path)).expanduser().resolve()
    if str(path).endswith(".runs.jsonl"):
        return str(path)
    return str(path.with_suffix(".runs.jsonl"))


def _save_file_path_from_result_jsonl(runs_jsonl_path):
    """由 xxx.runs.jsonl 推导最终聚合 JSON 路径 xxx.json。"""
    path = Path(str(runs_jsonl_path)).expanduser().resolve()
    text = str(path)
    if text.endswith(".runs.jsonl"):
        return text[: -len(".runs.jsonl")] + ".json"
    return str(path.with_suffix(".json"))


def _get_shard_path(incremental_dir, task_id, sample_idx, num_shards=16):
    """根据任务 ID 生成分片文件路径，使同一任务总是写入同一文件。

    使用 task_id 的哈希来分散写入，减少单个文件的大小。
    """
    shard_id = int(task_id) % num_shards
    shard_filename = f"incremental_shard_{shard_id:03d}.jsonl"
    return os.path.join(incremental_dir, shard_filename)


def _append_to_shard_file(shard_path, incremental_data):
    """追加写入到分片文件。

    Args:
        shard_path: 分片文件路径
        incremental_data: 要写入的数据
    """
    with open(shard_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(incremental_data, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _merge_incremental_files(incremental_dir, output_path, num_shards=16):
    """合并所有分片文件到单个 JSONL 文件。

    Args:
        incremental_dir: 分片文件所在目录
        output_path: 输出文件路径
        num_shards: 分片文件数量
    """
    merged_records = []
    shard_files = []

    for i in range(num_shards):
        shard_path = os.path.join(incremental_dir, f"incremental_shard_{i:03d}.jsonl")
        if os.path.exists(shard_path):
            shard_files.append(shard_path)

    for shard_path in sorted(shard_files):
        try:
            with open(shard_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            merged_records.append(record)
                    except Exception:
                        pass
        except Exception as e:
            print(f"{Colors.BRIGHT_YELLOW}[!]{Colors.RESET} 警告：合并分片文件失败 {shard_path}: {e}")

    # 写入合并后的文件
    if merged_records:
        output_path_parent = Path(output_path).parent
        output_path_parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in merged_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}[V]{Colors.RESET} 已合并 {len(merged_records)} 条记录到 {output_path}")

    return merged_records


def append_jsonl(path, item):
    """追加写入单条 JSONL，并 fsync，适合断点续跑。"""
    target_path = Path(str(path)).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_item = _make_json_serializable(item)
    with target_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(serializable_item, ensure_ascii=False, separators=(",", ":")))
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _iter_result_items_from_payload(payload):
    """把普通 result JSON payload 展开为单条 run item。

    兼容：
    - pass_k=1: [result, ...] 或 result
    - pass_k>1: [{samples:[...]}, ...] 或 {samples:[...]}
    """
    if not payload:
        return
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("samples"), list):
            group_task_index = item.get("task_index")
            for sample in item.get("samples", []):
                if isinstance(sample, dict):
                    if sample.get("task_index") is None and group_task_index is not None:
                        sample = deepcopy(sample)
                        sample["task_index"] = group_task_index
                    yield sample
        else:
            yield item


def _load_result_runs_from_json_payload(
    payload,
    task_index_to_id=None,
    retry_failed=True,
):
    """从旧版聚合 JSON 中恢复 result_by_run_key / completed_run_keys。"""
    result_by_run_key = {}
    completed_run_keys = set()
    stats = {
        "loaded_completed_runs": 0,
        "skipped_failed_runs_to_retry": 0,
        "skipped_invalid_items": 0,
        "duplicated_run_keys": 0,
        "source_items": 0,
    }
    inv_task_id_to_index = {
        str(v).strip(): k
        for k, v in (task_index_to_id or {}).items()
        if str(v).strip()
    }

    for item in _iter_result_items_from_payload(payload):
        stats["source_items"] += 1
        run_key = _extract_run_key_from_result_item(item)
        if run_key is None:
            item_task_info = item.get("task_info", {}) if isinstance(item.get("task_info"), dict) else {}
            actual_task_id = str(item.get("task_id") or item_task_info.get("task_id") or "").strip()
            fallback_task_index = inv_task_id_to_index.get(actual_task_id) if actual_task_id else None
            run_key = _extract_run_key_from_result_item(
                item,
                fallback_task_index=fallback_task_index,
                fallback_sample_idx=1,
            )
        if run_key is None:
            stats["skipped_invalid_items"] += 1
            continue
        if run_key in result_by_run_key:
            stats["duplicated_run_keys"] += 1
        if not _is_resume_completed_result(item, retry_failed=retry_failed):
            stats["skipped_failed_runs_to_retry"] += 1
            continue
        result_by_run_key[run_key] = item
        completed_run_keys.add(run_key)
        stats["loaded_completed_runs"] += 1
    return result_by_run_key, completed_run_keys, stats


def load_result_runs_from_files(
    save_file_path,
    runs_jsonl_path,
    task_index_to_id=None,
    retry_failed=True,
):
    """从聚合 JSON 和增量 JSONL 中恢复结果。

    读取顺序：
    1. 先读旧版/阶段性聚合 JSON；
    2. 再读 runs.jsonl，后写入的同 run_key 结果覆盖前者。

    这样可以兼容旧结果，也可以兼容运行中尚未聚合但已经追加到 JSONL 的结果。
    """
    merged_result_by_run_key = {}
    merged_completed_run_keys = set()
    total_stats = {
        "loaded_completed_runs": 0,
        "skipped_failed_runs_to_retry": 0,
        "skipped_invalid_items": 0,
        "duplicated_run_keys": 0,
        "json_source_items": 0,
        "jsonl_source_lines": 0,
        "jsonl_bad_lines": 0,
        "sources": [],
    }

    def _merge(part_result_by_run_key, part_stats, source_name):
        total_stats["sources"].append(source_name)
        total_stats["loaded_completed_runs"] += int(part_stats.get("loaded_completed_runs", 0) or 0)
        total_stats["skipped_failed_runs_to_retry"] += int(part_stats.get("skipped_failed_runs_to_retry", 0) or 0)
        total_stats["skipped_invalid_items"] += int(part_stats.get("skipped_invalid_items", 0) or 0)
        total_stats["duplicated_run_keys"] += int(part_stats.get("duplicated_run_keys", 0) or 0)
        for run_key, result_item in part_result_by_run_key.items():
            if run_key in merged_result_by_run_key:
                total_stats["duplicated_run_keys"] += 1
            merged_result_by_run_key[run_key] = result_item
        merged_completed_run_keys.clear()
        merged_completed_run_keys.update(merged_result_by_run_key.keys())

    save_path = Path(str(save_file_path)).expanduser().resolve() if save_file_path else None
    if save_path and save_path.exists():
        with save_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        part_result, _part_keys, part_stats = _load_result_runs_from_json_payload(
            payload,
            task_index_to_id=task_index_to_id,
            retry_failed=retry_failed,
        )
        total_stats["json_source_items"] += int(part_stats.get("source_items", 0) or 0)
        _merge(part_result, part_stats, "json")

    runs_path = Path(str(runs_jsonl_path)).expanduser().resolve() if runs_jsonl_path else None
    if runs_path and runs_path.exists():
        line_payloads = []
        with runs_path.open("r", encoding="utf-8") as file:
            for _line_no, line in enumerate(file, start=1):
                raw = line.strip()
                if not raw:
                    continue
                total_stats["jsonl_source_lines"] += 1
                try:
                    obj = json.loads(raw)
                except Exception:
                    total_stats["jsonl_bad_lines"] += 1
                    continue
                if isinstance(obj, dict):
                    line_payloads.append(obj)
                else:
                    total_stats["jsonl_bad_lines"] += 1
        part_result, _part_keys, part_stats = _load_result_runs_from_json_payload(
            line_payloads,
            task_index_to_id=task_index_to_id,
            retry_failed=retry_failed,
        )
        _merge(part_result, part_stats, "jsonl")

    total_stats["unique_completed_run_keys"] = len(merged_completed_run_keys)
    return merged_result_by_run_key, merged_completed_run_keys, total_stats


def build_result_payload_from_runs(task_ids, task_index_to_id, result_by_run_key, task_lookup, pass_k):
    """从 result_by_run_key 构造最终聚合 JSON payload。"""
    if int(pass_k) > 1:
        return _build_grouped_result_payload(
            task_ids=task_ids,
            task_index_to_id=task_index_to_id,
            result_groups=_group_results_by_task_index(result_by_run_key),
            task_lookup=task_lookup,
            pass_k=int(pass_k),
        )
    return _build_single_run_payload(task_ids=task_ids, result_by_run_key=result_by_run_key)


def save_aggregated_result_json(save_file_path, task_ids, task_index_to_id, result_by_run_key, task_lookup, pass_k):
    """将内存中的 JSONL run 结果聚合写成最终 JSON。"""
    payload = build_result_payload_from_runs(
        task_ids=task_ids,
        task_index_to_id=task_index_to_id,
        result_by_run_key=result_by_run_key,
        task_lookup=task_lookup,
        pass_k=pass_k,
    )
    save_json(save_file_path, payload)
    return payload


def _format_seconds(seconds):
    total = max(0, int(seconds or 0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_progress_snapshot(task_count, total_run_count, completed_task_count, completed_run_count, elapsed_seconds=0.0):
    """格式化总体进度快照。"""
    remaining_task_count = max(0, int(task_count) - int(completed_task_count))
    remaining_run_count = max(0, int(total_run_count) - int(completed_run_count))
    progress_bar = _format_progress_bar(completed_run_count, total_run_count)
    avg_run_seconds = float(elapsed_seconds or 0.0) / max(1, int(completed_run_count or 0))
    eta_seconds = avg_run_seconds * remaining_run_count if completed_run_count > 0 else 0.0
    return (
        f"{progress_bar} runs {completed_run_count}/{total_run_count} "
        f"(remaining {remaining_run_count}) | "
        f"tasks {completed_task_count}/{task_count} "
        f"(remaining {remaining_task_count}) | "
        f"elapsed {_format_seconds(elapsed_seconds)} | "
        f"eta {_format_seconds(eta_seconds)}"
    )


class _LiveProgressReporter:
    """仅在进度变化时更新单行进度条，不持续刷新。"""

    def __init__(self, refresh_interval=0.5):
        self.refresh_interval = float(refresh_interval)
        self._lock = threading.Lock()
        self._last_render_len = 0
        self._last_line = ""
        self._error_logs = []

    def update(self, line: str):
        """仅在进度条内容变化时更新显示。"""
        with self._lock:
            new_line = str(line or "")
            if new_line == self._last_line:
                return
            self._last_line = new_line
            clear_line = "\r" + (" " * self._last_render_len) + "\r"
            sys.stdout.write(clear_line)
            sys.stdout.write(new_line)
            sys.stdout.flush()
            self._last_render_len = len(new_line)

    def log_error(self, task_id: int, sample_idx: int, error: str):
        """保存错误日志，避免被进度条淹没。"""
        with self._lock:
            self._error_logs.append({
                "task_id": task_id,
                "sample_idx": sample_idx,
                "error": error,
                "time": time.time(),
            })

    def get_error_logs(self) -> list:
        """返回所有错误日志。"""
        with self._lock:
            return list(self._error_logs)

    def log(self, *args, **kwargs):
        """打印日志消息（会清除进度条后打印）。"""
        with self._lock:
            clear_line = "\r" + (" " * self._last_render_len) + "\r"
            sys.stdout.write(clear_line)
            sys.stdout.flush()
            print(*args, **kwargs, flush=True)

    def stop(self, final_line: str = ""):
        """停止进度条，显示最终行。"""
        with self._lock:
            if final_line:
                clear_line = "\r" + (" " * self._last_render_len) + "\r"
                sys.stdout.write(clear_line)
                sys.stdout.write(final_line + "\n")
                sys.stdout.flush()
            else:
                sys.stdout.write("\n")
                sys.stdout.flush()


def solve_task(
    env_name,
    env_config,
    agent_model,
    agent_model_provider,
    infer_mode,
    enable_thinking,
    task_id,
    agent_api_key=None,
    agent_base_url=None,
    user_model=None,
    user_provider=None,
    user_api_key=None,
    user_base_url=None,
    task_lookup=None,
    enable_checklist_eval=False,
    enable_rubric_eval=True,
    force_recompute_checklist_eval=False,
    force_recompute_rubric_eval=False,
    rubric_judge_config=None,
    sample_idx=1,
    pass_k=1,
    user_difficulty_config=None,
    agent_system_hint="",
    lang="cn",
    agent_reasoning_effort="high",
    agent_effort="high",
    agent_force_reasoning_effort=False,
    agent_use_responses_api=None,
    agent_omit_temperature=False,
    temperature=0.5,
    max_steps=0,
    task_timeout_seconds=3600,
):
    """运行单个 task。"""
    task_start_time = get_current_utc_timestamp()
    task_start_perf = time.time()
    env_kwargs = dict(env_config or {})
    if str(env_name or "").startswith("omniabench_") and "verbose" not in env_kwargs:
        env_kwargs["verbose"] = False
    if env_name in ["omniabench_conversation_rl", "omniabench_conversation_sft"]:
        if "user_model" not in env_kwargs and user_model:
            env_kwargs["user_model"] = user_model
        if "provider" not in env_kwargs and user_provider:
            env_kwargs["provider"] = user_provider
        if user_api_key:
            env_kwargs["api_key"] = user_api_key
        if user_base_url:
            env_kwargs["base_url"] = user_base_url
        if user_difficulty_config:
            env_kwargs["user_difficulty_config"] = deepcopy(user_difficulty_config)
        env_kwargs["lang"] = lang
        # Pass enable_thinking to conversation environments
        if "enable_thinking" not in env_kwargs:
            env_kwargs["enable_thinking"] = env_config.get("enable_thinking", False)

    env_cls = ENV_CLS_MAP[env_name]
    try:
        env_sig = inspect.signature(env_cls.__init__)
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in env_sig.parameters.values()
        )
        if not accepts_var_kwargs:
            allowed_env_keys = {
                name
                for name, p in env_sig.parameters.items()
                if name != "self"
                and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            env_kwargs = {k: v for k, v in env_kwargs.items() if k in allowed_env_keys}
    except (TypeError, ValueError):
        pass

    env = env_cls(**env_kwargs)
    # Use CLI-provided max_steps or fall back to env-specific default
    effective_max_steps = max_steps if max_steps > 0 else MAX_STEPS_MAP[env_name]

    # Agent thinking takes precedence from CLI --enable-thinking flag
    # If not set, check env_config for user_enable_thinking
    agent_thinking = enable_thinking
    if not agent_thinking and "enable_thinking" in (env_config or {}):
        agent_thinking = env_config.get("enable_thinking", False)

    agent = TaskSolveAgent(
        env_name=env_name,
        env=env,
        model=agent_model,
        provider=agent_model_provider,
        infer_mode=infer_mode,
        temperature=temperature,
        max_steps=effective_max_steps,
        enable_thinking=agent_thinking,
        api_key=agent_api_key,
        base_url=agent_base_url,
        extra_system_prompt=agent_system_hint,
        lang=lang,
        reasoning_effort=agent_reasoning_effort,
        effort=agent_effort,
        force_reasoning_effort=agent_force_reasoning_effort,
        use_responses_api=agent_use_responses_api,
        omit_temperature=agent_omit_temperature,
    )

    result = agent.run(task_index=task_id, timeout_seconds=task_timeout_seconds)
    init_state, final_state = _extract_env_states(env)
    if init_state is not None:
        result["init_state"] = init_state
    if final_state is not None:
        result["final_state"] = final_state
    result = augment_result_with_evaluations(
        result_item=result,
        task_lookup=task_lookup if isinstance(task_lookup, dict) else {},
        enable_checklist_eval=bool(enable_checklist_eval),
        enable_rubric_eval=bool(enable_rubric_eval),
        force_recompute_checklist_eval=bool(force_recompute_checklist_eval),
        force_recompute_rubric_eval=bool(force_recompute_rubric_eval),
        rubric_judge_config=rubric_judge_config if isinstance(rubric_judge_config, dict) else None,
        lang=lang,
    )
    task_info = result.get("task_info", {}) if isinstance(result.get("task_info"), dict) else {}
    task_info["task_index"] = int(task_id)
    task_info["sample_idx"] = int(sample_idx)
    task_info["pass_k"] = int(pass_k)
    enabled_difficulties = []
    if isinstance(user_difficulty_config, dict):
        enabled_difficulties = list(user_difficulty_config.get("enabled_atomic_difficulties") or [])
    if enabled_difficulties:
        task_info["enabled_atomic_difficulties"] = enabled_difficulties
    result["task_info"] = task_info
    result["task_index"] = int(task_id)
    result["sample_idx"] = int(sample_idx)
    result["pass_k"] = int(pass_k)
    result["run_key"] = f"task_{int(task_id)}__sample_{int(sample_idx)}"
    result["result_status"] = "completed"
    result["start_time"] = task_start_time
    result["finish_time"] = get_current_utc_timestamp()
    result["elapsed_seconds"] = round(max(0.0, time.time() - task_start_perf), 6)
    result["termination_reason"] = str(
        (result.get("final_info", {}) if isinstance(result.get("final_info"), dict) else {}).get("termination_reason")
        or (result.get("final_info", {}) if isinstance(result.get("final_info"), dict) else {}).get("terminated_by")
        or "COMPLETED"
    )
    if hasattr(env, "trajectory"):
        result["env_trajectory"] = _make_json_serializable(deepcopy(getattr(env, "trajectory")))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Run OmniaBench evaluation benchmark.")
    parser.add_argument("--execution-mode", choices=["run_and_eval", "eval_only"], default="run_and_eval")
    parser.add_argument("--env-name", choices=sorted(ENV_CLS_MAP.keys()), default="omniabench_conversation_rl")
    parser.add_argument("--batch", default="", help="Batch name defined in configs/datasets.json")
    parser.add_argument("--task-items-path", default="", help="Optional: direct path to prepared task_items json")
    parser.add_argument("--result-file-path", default="", help="Result file used in eval_only mode")
    parser.add_argument("--save-file-path", default="", help="Optional custom save path")
    parser.add_argument("--resume", action="store_true", help="Automatically resume from the latest matching result file in out-dir")
    parser.add_argument("--prefix", default="", help="")
    parser.add_argument(
        "--resume-keep-failed",
        action="store_true",
        help="When resuming, keep failed/INFRA_ERROR runs as completed instead of retrying them.",
    )
    parser.add_argument(
        "--aggregate-every-runs",
        type=int,
        default=0,
        help=(
            "How often to rebuild the final aggregated JSON from runs.jsonl. "
            "0 means only aggregate at the end; running progress is still saved to .runs.jsonl."
        ),
    )
    parser.add_argument("--out-dir", default=str(RESULTS_DIR), help="Result output directory")
    parser.add_argument("--agent-model", default="gpt-5.4")
    parser.add_argument("--agent-provider", default="openai")
    parser.add_argument("--user-model", default="gpt-4.1")
    parser.add_argument("--user-provider", default="openai")
    parser.add_argument("--rubric-judge-model", default="gpt-4.1")
    parser.add_argument("--rubric-judge-provider", default="openai")
    parser.add_argument("--infer-mode", choices=["prompt", "fc"], default="fc")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-rubric-eval", action="store_true")
    parser.add_argument("--enable-checklist-eval", action="store_true")
    parser.add_argument("--force-recompute-rubric-eval", action="store_true")
    parser.add_argument("--force-recompute-checklist-eval", action="store_true")
    parser.add_argument("--pass-k", type=int, default=1)
    parser.add_argument("--max-task-workers", type=int, default=8)
    parser.add_argument("--selected-task-ids", nargs="*", default=None)
    parser.add_argument("--selected-task-keys", nargs="*", default=None)
    parser.add_argument(
        "--global-id-range",
        default="1-10",
        help="global_id 范围，如 1-10 / 11-20 / 1-10,21-30 / all。默认前 10 条。",
    )
    parser.add_argument(
        "--lang-filter",
        choices=["all", "cn", "en"],
        default="all",
        help="语言过滤：cn / en / all。会与 global_id 区间取交集。",
    )
    parser.add_argument(
        "--prompt-lang",
        choices=["auto", "cn", "en"],
        default="auto",
        help="强制指定系统 Prompt 语言(中文/英文)。默认 auto 根据任务实际 lang 字段判断。",
    )
    parser.add_argument("--user-difficulty-ids", default="", help="Comma-separated ids, e.g. 2.b,6,8")
    parser.add_argument("--agent-api-key", default="")
    parser.add_argument("--agent-base-url", default="")
    parser.add_argument("--user-api-key", default="")
    parser.add_argument("--user-base-url", default="")
    parser.add_argument("--rubric-judge-api-key", default="")
    parser.add_argument("--rubric-judge-base-url", default="")
    parser.add_argument("--agent-reasoning-effort", default="high", help="Reasoning effort for gpt-5.x responses API: low, medium, high")
    parser.add_argument("--agent-effort", default="high", help="Effort for Claude API: low, medium, high, xhigh, max")
    parser.add_argument(
        "--agent-force-reasoning-effort",
        action="store_true",
        help="Force-enable reasoning_effort even when --agent-base-url isn't recognized as an official "
             "OpenAI endpoint. Use this if your self-hosted/third-party proxy supports the parameter.",
    )
    parser.add_argument(
        "--agent-use-responses-api",
        choices=["auto", "true", "false"],
        default="auto",
        help="Force on/off the OpenAI Responses API path for the agent model, overriding the "
             "model-based default (auto).",
    )
    parser.add_argument(
        "--agent-responses-api-omit-temperature",
        action="store_true",
        help="Drop the `temperature` field when calling the OpenAI Responses API, for third-party "
             "deployments that reject it.",
    )
    parser.add_argument("--thinking-config", default="", help="JSON config for thinking settings, e.g. '{\"user_enable_thinking\": false, \"rubric_judge_enable_thinking\": false}'")
    parser.add_argument(
        "--fs-bundle-root",
        default="",
        help="Optional: bundled fs root for relocation. Supports <root>/fs_fixtures/<env_id>/<task_id> or <root>/<env_id>/<task_id>.",
    )
    parser.add_argument(
        "--fs-tmp-root",
        default="",
        help="Optional: override fs tmp root. Runtime resolves each task to <root>/<env_id>/<task_id>.",
    )
    parser.add_argument(
        "--incremental-dir",
        default="",
        help="高速盘路径，用于存放增量分片文件（可选，默认在 out-dir 同级）",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=16,
        help="分片文件数量，默认 16 个",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for agent inference. Default 0.5.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Max steps per task. 0 means use env-specific default (rl=100, non-conv-rl=30, conv-sft=40, non-conv-sft=30).",
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=int,
        default=3600,
        help="Per-task wall-clock timeout in seconds before forced TIMEOUT termination. Default 3600 (1 hour).",
    )
    parser.add_argument(
        "--agent-temperature",
        type=float,
        default=0.5,
        help="Alias for --temperature. Temperature for agent inference.",
    )
    return parser


def load_dataset_registry():
    """读取 batch 注册表。"""
    datasets_path = CONFIG_DIR / "datasets.json"
    if not datasets_path.exists():
        raise FileNotFoundError(f"Dataset registry not found: {datasets_path}")
    with datasets_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_task_items_path(batch: str, task_items_path: str) -> str:
    """根据 batch 或显式路径解析任务文件。"""
    if task_items_path:
        return str(Path(task_items_path).expanduser().resolve())
    if not batch:
        raise ValueError("Please provide --batch or --task-items-path.")
    for item in load_dataset_registry():
        if str(item.get("batch")) == batch:
            rel_path = item.get("relative_paths", {}).get("prepared_task_items_path", "")
            if not rel_path:
                break
            return str((FINAL_EVAL_ROOT / rel_path).resolve())
    raise ValueError(f"Unknown batch: {batch}")


def main():
    load_dotenv()
    args = build_parser().parse_args()
    runtime_path_overrides = _apply_runtime_path_overrides(args)

    task_items_path = resolve_task_items_path(args.batch, args.task_items_path)
    task_lookup = load_task_lookup(task_items_path)

    agent_api_key = _resolve_config_value(args.agent_api_key, fallback_env_var="OPENAI_API_KEY", default="")
    agent_base_url = _normalize_openai_base_url(
        _resolve_config_value(args.agent_base_url, fallback_env_var="OPENAI_BASE_URL", default="")
    )
    user_api_key = _resolve_config_value(args.user_api_key, fallback_env_var="", default="") or agent_api_key
    user_base_url = _normalize_openai_base_url(
        _resolve_config_value(args.user_base_url, fallback_env_var="", default="")
    ) or agent_base_url
    rubric_judge_api_key = _resolve_config_value(
        args.rubric_judge_api_key,
        fallback_env_var="OPENAI_API_KEY",
        default="",
    ) or agent_api_key
    rubric_judge_base_url = _normalize_openai_base_url(
        _resolve_config_value(
            args.rubric_judge_base_url,
            fallback_env_var="OPENAI_BASE_URL",
            default="",
        )
    ) or agent_base_url

    if not agent_api_key:
        raise RuntimeError("未检测到 agent API key。请设置 api_key 或 OPENAI_API_KEY。")

    agent_use_responses_api = {"auto": None, "true": True, "false": False}[args.agent_use_responses_api]

    user_difficulty_ids = [item.strip() for item in str(args.user_difficulty_ids or "").split(",") if item.strip()]
    user_difficulty_config = normalize_user_difficulty_config(user_difficulty_ids)
    agent_system_hint = build_agent_followup_hint(user_difficulty_config)

    # Parse thinking config from JSON argument
    import json
    thinking_config = {}
    if args.thinking_config:
        try:
            thinking_config = json.loads(args.thinking_config)
        except json.JSONDecodeError:
            print(f"Warning: Invalid thinking-config JSON: {args.thinking_config}")

    env_config = {
        "mode": "train",
        "task_items_path": task_items_path,
        "user_model": args.user_model,
        "provider": args.user_provider,
        "user_difficulty_config": user_difficulty_config,
        "enable_thinking": thinking_config.get("user_enable_thinking", False),
    }

    rubric_judge_config = {
        "provider": args.rubric_judge_provider,
        "model": args.rubric_judge_model,
        "api_key": rubric_judge_api_key,
        "base_url": rubric_judge_base_url,
        "enable_thinking": thinking_config.get("rubric_judge_enable_thinking", False),
    }

    if args.execution_mode == "eval_only":
        if not args.result_file_path:
            raise RuntimeError("execution-mode=eval_only 时必须提供 --result-file-path。")
        output_path = args.save_file_path or args.result_file_path
        eval_lang = args.prompt_lang if args.prompt_lang != "auto" else (
            args.lang_filter if args.lang_filter in ["cn", "en"] else "cn"
        )
        evaluate_result_file(
            result_file_path=args.result_file_path,
            task_items_path=task_items_path,
            save_file_path=output_path,
            num_workers=max(1, int(args.max_task_workers)),
            enable_checklist_eval=bool(args.enable_checklist_eval),
            enable_rubric_eval=not bool(args.disable_rubric_eval),
            force_recompute_checklist_eval=bool(args.force_recompute_checklist_eval),
            force_recompute_rubric_eval=bool(args.force_recompute_rubric_eval),
            rubric_judge_config=rubric_judge_config,
            lang=eval_lang,
        )
        print("eval_save_file_path:", output_path)
        return

    task_ids = _resolve_task_ids(
        task_items_path,
        selected_task_ids=args.selected_task_ids,
        selected_task_keys=args.selected_task_keys,
        global_id_range=args.global_id_range,
        lang_filter=args.lang_filter,
    )
    task_index_to_id = _build_task_index_to_task_id(task_items_path)
    task_index_metadata = _build_task_index_metadata(task_items_path)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file_path = args.save_file_path
    batch_name = args.batch or Path(task_items_path).stem
    if not save_file_path:
        prefix = f"{args.env_name}-{batch_name}-{args.agent_model}-{args.infer_mode}_{args.user_model}_"
        suffix = f"{'_passk' + str(int(args.pass_k)) if int(args.pass_k) > 1 else ''}.json"

        if args.resume:
            import glob
            existing_json_files = glob.glob(str(out_dir / f"{prefix}*{suffix}"))
            existing_jsonl_files = glob.glob(str(out_dir / f"{prefix}*{suffix[:-5]}.runs.jsonl"))
            existing_files = []
            for path in existing_json_files:
                existing_files.append((os.path.getmtime(path), "json", path))
            for path in existing_jsonl_files:
                existing_files.append((os.path.getmtime(path), "jsonl", path))
            if existing_files:
                existing_files.sort()
                _, file_type, found_path = existing_files[-1]
                save_file_path = found_path if file_type == "json" else _save_file_path_from_result_jsonl(found_path)
                print(f"[run_eval] Found existing file to resume: {found_path}")

        if not save_file_path:
            save_file_path = str(out_dir / f"{prefix}{get_current_time()}{suffix}")

    save_file_path = str(Path(save_file_path).expanduser().resolve())
    if save_file_path.endswith(".runs.jsonl"):
        save_file_path = _save_file_path_from_result_jsonl(save_file_path)
    runs_jsonl_path = _result_jsonl_path_from_save_file(save_file_path)

    # 初始化增量分片文件目录
    # 格式：<output_file_path_without_suffix>_incremental_shards/
    output_file_stem = Path(save_file_path).stem
    if args.incremental_dir:
        incremental_base = Path(args.incremental_dir).expanduser().resolve()
        incremental_base.mkdir(parents=True, exist_ok=True)
        incremental_dir = incremental_base / f"{output_file_stem}_incremental_shards"
    else:
        incremental_dir = Path(runs_jsonl_path).parent / f"{output_file_stem}_incremental_shards"
    incremental_dir.mkdir(parents=True, exist_ok=True)
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}[INFO]{Colors.RESET} 增量分片文件目录：{incremental_dir}")

    # Resume logic: 从分片增量文件恢复（快速，不阻塞启动）
    result_by_run_key = {}
    completed_run_keys = set()
    resume_loaded = False

    if args.resume:
        import traceback
        import glob as glob_module
        loaded_count = 0

        # 从分片文件加载（使用 glob 遍历所有 shard）
        try:
            # 提取 base_name 用于匹配（移除时间戳）
            import re
            base_name = re.sub(r'_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$', '', output_file_stem)

            # 分片目录的父目录必须与本次创建 incremental_dir 时使用的父目录一致
            # （若指定了 --incremental-dir，历史分片目录也位于该目录下，而不是 runs_jsonl_path 的父目录）
            shard_dir_parent = incremental_base if args.incremental_dir else Path(runs_jsonl_path).parent

            # 查找所有匹配的分片目录（不同时间戳的历史运行都可能包含互补的已完成结果）。
            # 注意：不排除本次的 incremental_dir 本身——它可能是刚新建的空目录（merge 无害），
            # 也可能是复用了已有历史时间戳的非空目录（此时必须保留，否则会丢数据）。
            shard_dir_pattern = str(shard_dir_parent / f"{base_name}*_incremental_shards")
            matching_shard_dirs = glob_module.glob(shard_dir_pattern)

            if matching_shard_dirs:
                # 按目录名（含时间戳）升序排列，确保后面处理的时间戳更新的目录
                # 在合并时能覆盖较早目录中的同一 (task_index, sample_idx) 结果
                matching_shard_dirs = sorted(matching_shard_dirs)
                print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}[V]{Colors.RESET} {Colors.CYAN}Found {len(matching_shard_dirs)} historical shard directories, merging all (oldest to newest):{Colors.RESET}")
                for d in matching_shard_dirs:
                    print(f"    - {d}")
                target_incremental_dirs = [Path(d) for d in matching_shard_dirs]
            else:
                target_incremental_dirs = [incremental_dir]

            def load_shard(shard_path):
                records = []
                with open(shard_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            result = item.get("result", {})
                            task_index = item.get("task_index")
                            sample_idx = item.get("sample_idx")
                            if task_index is not None and sample_idx is not None:
                                records.append((int(task_index), int(sample_idx), result, item))
                        except Exception:
                            pass
                return records

            retry_failed = not bool(getattr(args, "resume_keep_failed", False))
            shard_files_total = 0
            # 按目录顺序（旧→新）依次合并，目录内部用线程池并发加载分片文件，
            # 保证跨目录的覆盖顺序确定（新目录结果覆盖旧目录同 key 结果）
            for target_dir in target_incremental_dirs:
                shard_pattern = os.path.join(target_dir, "incremental_shard_*.jsonl")
                dir_shard_files = sorted(glob_module.glob(shard_pattern))
                shard_files_total += len(dir_shard_files)
                if not dir_shard_files:
                    continue
                with ThreadPoolExecutor(max_workers=min(16, len(dir_shard_files))) as executor:
                    futures = {executor.submit(load_shard, sp): sp for sp in dir_shard_files}
                    dir_records = []
                    for future in as_completed(futures):
                        dir_records.extend(future.result())
                for task_index, sample_idx, result, item in dir_records:
                    # 检查是否是失败任务，需要重跑
                    if retry_failed:
                        status = str(result.get("result_status", "")).strip().lower()
                        termination = str(result.get("termination_reason", "")).strip().upper()
                        if status == "failed" or termination == "INFRA_ERROR":
                            # 失败任务，不加入 completed，会重跑；同时清除之前该 key 可能存在的旧完成记录
                            result_by_run_key.pop((task_index, sample_idx), None)
                            completed_run_keys.discard((task_index, sample_idx))
                            continue
                    result_by_run_key[(task_index, sample_idx)] = result
                    completed_run_keys.add((task_index, sample_idx))

            loaded_count = len(completed_run_keys)

            if shard_files_total:
                print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}[V]{Colors.RESET} {Colors.CYAN}Loaded from shard files:{Colors.RESET} {len(completed_run_keys)} results ({shard_files_total} shards) [快速加载]")
                resume_loaded = True
        except Exception as e:
            traceback.print_exc()
            print(f"{Colors.BRIGHT_RED}[X]{Colors.RESET} {Colors.RED}Error loading resume data:{Colors.RESET} {e}")

    # Fallback: 从旧的单文件 incremental.jsonl 格式恢复
    if not resume_loaded and args.resume and (os.path.exists(save_file_path) or os.path.exists(runs_jsonl_path)):
        import traceback
        try:
            retry_failed = not bool(getattr(args, "resume_keep_failed", False))
            (
                result_by_run_key,
                completed_run_keys,
                resume_stats,
            ) = load_result_runs_from_files(
                save_file_path=save_file_path,
                runs_jsonl_path=runs_jsonl_path,
                task_index_to_id=task_index_to_id,
                retry_failed=retry_failed,
            )
        except Exception as e:
            traceback.print_exc()
            print(f"{Colors.BRIGHT_RED}[X]{Colors.RESET} {Colors.RED}Error loading resume files:{Colors.RESET} {e}")

    # 计算待运行的任务
    pending_runs = [
        (task_id, sample_idx)
        for task_id in task_ids
        for sample_idx in range(1, max(1, int(args.pass_k)) + 1)
    ]
    pending_runs = [r for r in pending_runs if r not in completed_run_keys]

    # 打印 resume 信息
    if args.resume:
        print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}[V]{Colors.RESET} {Colors.CYAN}Resumed:{Colors.RESET} {len(completed_run_keys)} completed, {len(pending_runs)} pending")
        print("")

    task_count = len(task_ids)
    total_run_count = len(task_ids) * max(1, int(args.pass_k))

    # Print startup banner with colors
    print("")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}╔{'═'*78}╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}║{' '*30}EVALUATION STARTING{' '*29}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}╚{'═'*78}╝{Colors.RESET}")
    print("")

    # Config section
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}[CONFIG] Configuration:{Colors.RESET}")
    print(f"  {Colors.CYAN}Environment:{Colors.RESET} {args.env_name}")
    print(f"  {Colors.CYAN}Task Items:{Colors.RESET} {task_items_path}")
    print(f"  {Colors.CYAN}Output:{Colors.RESET} {save_file_path}")
    print("")

    # Task config
    print(f"{Colors.BOLD}{Colors.BRIGHT_BLUE}[TASKS] Task Configuration:{Colors.RESET}")
    print(f"  {Colors.CYAN}Total Tasks:{Colors.RESET} {task_count}")
    print(f"  {Colors.CYAN}Pass-K:{Colors.RESET} {args.pass_k}")
    print(f"  {Colors.CYAN}Global ID Range:{Colors.RESET} {args.global_id_range}")
    print(f"  {Colors.CYAN}Lang Filter:{Colors.RESET} {args.lang_filter}")
    print(f"  {Colors.CYAN}Thinking:{Colors.RESET} {bool(args.enable_thinking)}")
    print("")

    effective_workers = min(max(1, int(args.max_task_workers)), max(1, total_run_count))
    run_start_perf = time.time()

    # Initial progress
    initial_completed = _count_completed_tasks(task_ids, completed_run_keys, args.pass_k)
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}[PROGRESS] Initial Progress:{Colors.RESET}")
    print(f"  {format_progress_bar(initial_completed, total_run_count, 30, Colors.BRIGHT_GREEN)}")
    print(f"  {Colors.CYAN}Completed:{Colors.RESET} {initial_completed}/{total_run_count} runs ({initial_completed}/{task_count} tasks)")
    print(f"  {Colors.CYAN}Pending:{Colors.RESET} {len(pending_runs)} runs")
    print("")
    print(f"{Colors.DIM}{'─'*80}{Colors.RESET}")
    print("")

    progress_reporter = _LiveProgressReporter()
    print(f"{Colors.DIM}Starting execution...{Colors.RESET}")
    print("")

    try:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_meta = {
                executor.submit(
                    solve_task,
                    env_name=args.env_name,
                    env_config=env_config,
                    agent_model=args.agent_model,
                    agent_model_provider=args.agent_provider,
                    infer_mode=args.infer_mode,
                    enable_thinking=bool(args.enable_thinking),
                    task_id=task_id,
                    agent_api_key=agent_api_key,
                    agent_base_url=agent_base_url,
                    user_model=args.user_model,
                    user_provider=args.user_provider,
                    user_api_key=user_api_key,
                    user_base_url=user_base_url,
                    task_lookup=task_lookup,
                    enable_checklist_eval=bool(args.enable_checklist_eval),
                    enable_rubric_eval=not bool(args.disable_rubric_eval),
                    force_recompute_checklist_eval=bool(args.force_recompute_checklist_eval),
                    force_recompute_rubric_eval=bool(args.force_recompute_rubric_eval),
                    rubric_judge_config=rubric_judge_config,
                    sample_idx=sample_idx,
                    pass_k=int(args.pass_k),
                    user_difficulty_config=user_difficulty_config,
                    agent_system_hint=agent_system_hint,
                    lang=args.prompt_lang if args.prompt_lang != "auto" else task_index_metadata.get(int(task_id), {}).get("lang", args.lang_filter if args.lang_filter in ["cn", "en"] else "cn"),
                    agent_reasoning_effort=args.agent_reasoning_effort,
                    agent_effort=args.agent_effort,
                    agent_force_reasoning_effort=bool(args.agent_force_reasoning_effort),
                    agent_use_responses_api=agent_use_responses_api,
                    agent_omit_temperature=bool(args.agent_responses_api_omit_temperature),
                    temperature=args.agent_temperature if args.agent_temperature != 0.5 else args.temperature,
                    max_steps=args.max_steps,
                    task_timeout_seconds=args.task_timeout_seconds,
                ): (task_id, sample_idx)
                for task_id, sample_idx in pending_runs
            }

            for future in as_completed(future_to_meta):
                task_id, sample_idx = future_to_meta[future]
                task_meta = task_index_metadata.get(int(task_id), {})
                task_global_id = task_meta.get("global_id")
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "task_info": {
                            "task_id": str(task_index_to_id.get(task_id, "")).strip(),
                            "task_index": int(task_id),
                            "global_id": task_global_id,
                            "sample_idx": int(sample_idx),
                            "pass_k": int(args.pass_k),
                        },
                        "task_index": int(task_id),
                        "sample_idx": int(sample_idx),
                        "pass_k": int(args.pass_k),
                        "run_key": f"task_{int(task_id)}__sample_{int(sample_idx)}",
                        "result_status": "failed",
                        "runtime_error": {
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                        "start_time": get_current_utc_timestamp(),
                        "finish_time": get_current_utc_timestamp(),
                        "elapsed_seconds": 0.0,
                        "termination_reason": "INFRA_ERROR",
                    }
                    error_type = type(exc).__name__
                    error_msg = str(exc)

                    # 判断是否为模型生成内容导致的错误（预期内）vs 框架 bug（意外）
                    infra_errors = {
                        "FileNotFoundError", "PermissionError", "ConnectionError",
                        "TimeoutError", "ConnectionRefusedError", "SocketError",
                        "SSLError", "HTTPError", "URLError",
                    }

                    model_content_patterns = [
                        "dict' object has no attribute",
                        "list index out of range",
                        "KeyError",
                        "required field",
                        "JSON",
                    ]

                    # 环境加载阶段（reset）的错误属于框架/数据 bug，不是模型输出问题，
                    # 不能计入 model_output_error（否则会把数据/框架 bug 误算成模型能力）。
                    env_load_patterns = [
                        "not found in provided env_class_code",
                        "not found in env_items",
                        "Invalid env_id",
                    ]
                    is_env_load_issue = any(pattern in error_msg for pattern in env_load_patterns)

                    # SyntaxError 通常是模型生成代码有语法问题，属于预期内的模型输出错误
                    error_types_model_issue = {"AttributeError", "KeyError", "ValueError", "IndexError", "JSONDecodeError", "SyntaxError"}
                    is_model_issue = (
                        not is_env_load_issue
                        and (
                            error_type in error_types_model_issue
                            or any(pattern in error_msg for pattern in model_content_patterns)
                        )
                    )
                    is_infra_issue = is_env_load_issue or error_type in infra_errors

                    if is_model_issue:
                        result = {
                            "task_info": {
                                "task_id": str(task_index_to_id.get(task_id, "")).strip(),
                                "task_index": int(task_id),
                                "global_id": task_global_id,
                                "sample_idx": int(sample_idx),
                                "pass_k": int(args.pass_k),
                            },
                            "task_index": int(task_id),
                            "sample_idx": int(sample_idx),
                            "pass_k": int(args.pass_k),
                            "result_status": "model_output_error",
                            "model_error": {
                                "error_type": error_type,
                                "message": error_msg[:200],
                            },
                            "start_time": get_current_utc_timestamp(),
                            "finish_time": get_current_utc_timestamp(),
                            "elapsed_seconds": 0.0,
                            "termination_reason": "MODEL_OUTPUT_ERROR",
                        }
                        progress_reporter.log(
                            f"{Colors.BRIGHT_YELLOW}[~]{Colors.RESET} task_index={task_id} global_id={task_global_id} sample={sample_idx} "
                            f"{Colors.YELLOW}model output error{Colors.RESET}: {error_msg[:60]}"
                        )
                    else:
                        result = {
                            "task_info": {
                                "task_id": str(task_index_to_id.get(task_id, "")).strip(),
                                "task_index": int(task_id),
                                "global_id": task_global_id,
                                "sample_idx": int(sample_idx),
                                "pass_k": int(args.pass_k),
                            },
                            "task_index": int(task_id),
                            "sample_idx": int(sample_idx),
                            "pass_k": int(args.pass_k),
                            "result_status": "failed",
                            "runtime_error": {
                                "error_type": error_type,
                                "message": error_msg,
                            },
                            "start_time": get_current_utc_timestamp(),
                            "finish_time": get_current_utc_timestamp(),
                            "elapsed_seconds": 0.0,
                            "termination_reason": "INFRA_ERROR",
                        }
                        progress_reporter.log_error(task_id, sample_idx, f"{error_type}: {error_msg}")
                        progress_reporter.log(
                            f"{Colors.BRIGHT_RED}[X]{Colors.RESET} task_index={task_id} global_id={task_global_id} sample={sample_idx} "
                            f"{Colors.RED}infra error{Colors.RESET}: {error_msg[:60]}"
                        )
                else:
                    actual_task_id = str(
                        (result.get("task_info", {}) if isinstance(result.get("task_info"), dict) else {}).get("task_id")
                        or task_index_to_id.get(task_id, "")
                    ).strip()
                    actual_global_id = (
                        (result.get("task_info", {}) if isinstance(result.get("task_info"), dict) else {}).get("global_id")
                        or task_global_id
                    )
                    result_duration = result.get("elapsed_seconds", 0.0)
                    # Determine status icon and color based on result
                    # Try to get rubric_score from multiple sources
                    rubric_score = result.get("rubric_score")
                    if rubric_score is None:
                        rubric_eval = result.get("rubric_eval", {})
                        if isinstance(rubric_eval, dict):
                            rubric_score = rubric_eval.get("avg_result", 0.0)
                        else:
                            rubric_score = 0.0
                    if rubric_score is None:
                        rubric_score = 0.0

                    rubric_passed = result.get("rubric_passed")
                    if rubric_passed is None:
                        rubric_passed = rubric_score >= 1.0

                    if rubric_passed:
                        status_str = f"{Colors.BRIGHT_GREEN}[V]{Colors.RESET}"
                    elif rubric_score >= 0.5:
                        status_str = f"{Colors.BRIGHT_YELLOW}[~]{Colors.RESET}"
                    else:
                        status_str = f"{Colors.BRIGHT_RED}[X]{Colors.RESET}"
                    progress_reporter.log(
                        f"{status_str} task_index={task_id} global_id={actual_global_id} sample={sample_idx} "
                        f"done: {actual_task_id or '-'} | took {round(float(result_duration or 0.0), 2)}s "
                        f"| termination={result.get('termination_reason', '-')} | score={rubric_score:.2f}"
                    )

                env_trajectory = result.pop("env_trajectory", None) if isinstance(result, dict) else None
                if env_trajectory:
                    state_diff_dir = Path(save_file_path).with_suffix("")
                    state_diff_dir = state_diff_dir.parent / f"{state_diff_dir.name}_state_diffs"
                    state_diff_dir.mkdir(parents=True, exist_ok=True)
                    state_diff_path = state_diff_dir / (
                        f"task_index_{int(task_id):04d}_global_{str(task_global_id)}_sample_{int(sample_idx):02d}.json"
                    )
                    save_json(
                        state_diff_path,
                        {
                            "task_index": int(task_id),
                            "global_id": task_global_id,
                            "sample_idx": int(sample_idx),
                            "env_trajectory": env_trajectory,
                        },
                    )
                    result["state_diff_path"] = str(state_diff_path)

                run_key_tuple = (int(task_id), int(sample_idx))
                result_by_run_key[run_key_tuple] = result
                completed_run_keys.add(run_key_tuple)

                # 增量写入：使用分片文件避免并发竞争
                # 每个 task_id 总是写入同一个分片文件，减少文件数量
                incremental_shard_path = _get_shard_path(incremental_dir, task_id, sample_idx, num_shards=int(args.num_shards))
                incremental_data = {
                    "task_index": int(task_id),
                    "sample_idx": int(sample_idx),
                    "result": _make_json_serializable(result),
                    "timestamp": get_current_utc_timestamp(),
                }
                append_start_perf = time.time()
                _append_to_shard_file(incremental_shard_path, incremental_data)
                append_elapsed = time.time() - append_start_perf

                completed_run_count = len(completed_run_keys)
                completed_task_count = _count_completed_tasks(task_ids, completed_run_keys, args.pass_k)
                remaining_run_count = max(0, total_run_count - completed_run_count)
                remaining_task_count = max(0, task_count - completed_task_count)
                elapsed_seconds = time.time() - run_start_perf
                elapsed_str = _format_seconds(int(elapsed_seconds))
                progress_bar = format_progress_bar(completed_run_count, total_run_count, 25, Colors.BRIGHT_GREEN)
                progress_reporter.update(
                    f"{Colors.BRIGHT_CYAN}[RUNNING]{Colors.RESET} "
                    + f"{progress_bar} "
                    + f"{completed_run_count}/{total_run_count} runs "
                    + f"({completed_task_count}/{task_count} tasks) "
                    + f"| time: {elapsed_str}"
                )
    finally:
        final_completed_run_count = len(completed_run_keys)
        final_completed_task_count = _count_completed_tasks(task_ids, completed_run_keys, args.pass_k)
        final_elapsed_seconds = time.time() - run_start_perf
        final_progress_bar = format_progress_bar(final_completed_run_count, total_run_count, 30, Colors.BRIGHT_GREEN)
        progress_reporter.stop(
            final_line=(
                f"{Colors.BRIGHT_GREEN}[DONE]{Colors.RESET} "
                + f"{final_progress_bar} "
                + f"{final_completed_run_count}/{total_run_count} runs "
                + f"({final_completed_task_count}/{task_count} tasks) "
                + f"| time: {_format_seconds(int(final_elapsed_seconds))}"
            )
        )

    # 合并所有分片增量文件到单个 JSONL 文件
    print("")
    print(f"{Colors.DIM}Merging incremental shard files...{Colors.RESET}")
    incremental_output_path = str(Path(save_file_path).with_suffix('.runs.jsonl'))
    _merge_incremental_files(incremental_dir, incremental_output_path, num_shards=int(args.num_shards))

    final_aggregate_start_perf = time.time()
    save_aggregated_result_json(
        save_file_path=save_file_path,
        task_ids=task_ids,
        task_index_to_id=task_index_to_id,
        result_by_run_key=result_by_run_key,
        task_lookup=task_lookup,
        pass_k=int(args.pass_k),
    )
    final_aggregate_elapsed = time.time() - final_aggregate_start_perf
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}✓{Colors.RESET} {Colors.CYAN}Results saved to:{Colors.RESET} {save_file_path}")


if __name__ == "__main__":
    main()
