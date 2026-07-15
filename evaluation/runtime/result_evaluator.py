"""
Result evaluation helpers for rubric-first scoring.
"""

from __future__ import annotations

import json
import ast
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from agent.agent_llm_inference import openai_inference_prompt
from envscaler_env.utils.env_util import run_check_function


RUBRIC_FIELDS = {
    "rubric_count",
    "rubric_total_score",
    "general_rubric_score",
    "task_specific_rubric_score",
    "rubrics_text",
    "rubric_explanation",
    "rubric_items",
    "rubrics",
}


RUBRIC_JUDGE_TEMPLATE = """你是一名严格的任务执行结果评测员。

你将获得：
1. task：任务描述
2. init_state：初始状态
3. final_state：最终状态
4. final_assistant_response：智能体最后给用户的回答（如果有）
5. trajectory_summary：执行轨迹摘要，包含工具调用、参数与关键观察
6. rubric_items：需要逐条判定的 rubric 列表

你的目标：
基于这些证据，对每条 rubric 做二值判断：
- 满足则 passed=true，earned_score=该条 points
- 不满足则 passed=false，earned_score=0

要求：
1. 只能依据输入证据判断，不要臆造未出现的信息。
2. 若证据不足，应判为不满足，并在 reason 中简短说明证据不足。
3. 对于一些名字的别称如果在情境中本来就是口头表述不是特定词句，就要结合所有提供的信息判断是否完成对应语义的特定任务。
4. 如果有的 rubric 内容需要在有前提的情况下判断，请严格按照前提是否成立来判断后续内容，如果前提不成立应该直接判断满足条件。
5. 输出必须是严格 JSON，不要输出 markdown，不要输出额外解释。
6. 请先完整输出 items，再严格根据 items 自动计算顶层 earned_score、total_score、avg_result，不要手工估算。

输出 JSON 格式：
{{
  "earned_score": 0,
  "total_score": 0,
  "avg_result": 0.0,
  "summary": "简短总结",
  "items": [
    {{
      "tag": "G1",
      "category": "general",
      "points": 1,
      "passed": false,
      "earned_score": 0,
      "reason": "简短理由"
    }}
  ]
}}

校验要求：
1. items 数量必须与 rubric_items 数量一致
2. tag 必须与 rubric_items 对应条目一致
3. points 必须与 rubric_items 对应条目一致
4. earned_score 之和必须等于顶层 earned_score
5. total_score 必须等于 rubric_items points 总和
6. avg_result = earned_score / total_score，保留 4 位小数

# task
{task}

# init_state
{init_state}

# final_state
{final_state}

# final_assistant_response
{final_assistant_response}

# trajectory_summary
{trajectory_summary}

# rubric_items
{rubric_items}
"""

RUBRIC_JUDGE_TEMPLATE_EN = """You are a strict evaluator of task execution results.

You will be given:
1. task: the task description
2. init_state: the initial state
3. final_state: the final state
4. final_assistant_response: the assistant's final response to the user, if any
5. trajectory_summary: a summary of the execution trajectory, including tool calls, parameters, and key observations
6. rubric_items: the list of rubric items to judge one by one

Your goal:
Based on this evidence, make a binary judgment for each rubric item:
- If satisfied, set passed=true and earned_score=the item's points
- If not satisfied, set passed=false and earned_score=0

Requirements:
1. Judge only based on the input evidence. Do not fabricate information that does not appear.
2. If the evidence is insufficient, judge it as not satisfied and briefly state in the reason that the evidence is insufficient.
3. For certain name aliases that are inherently oral expressions rather than specific terms in context, it is necessary to integrate all provided information to determine whether the specific task corresponding to the intended semantic meaning has been completed.
4. If a rubric item needs to be judged under certain prerequisites, strictly follow whether the prerequisite holds. If the prerequisite does not hold, the item should be judged as satisfied directly.
5. The output must be strict JSON. Do not output markdown or any additional explanation.
6. First output all items completely, then automatically calculate the top-level earned_score, total_score, and avg_result strictly based on the items. Do not estimate them manually.

Output JSON format:
{{
  "earned_score": 0,
  "total_score": 0,
  "avg_result": 0.0,
  "summary": "Brief summary",
  "items": [
    {{
      "tag": "G1",
      "category": "general",
      "points": 1,
      "passed": false,
      "earned_score": 0,
      "reason": "Brief reason"
    }}
  ]
}}

Validation requirements:
1. The number of items must match the number of rubric_items
2. Each tag must match the corresponding item in rubric_items
3. Each points value must match the corresponding item in rubric_items
4. The sum of earned_score across items must equal the top-level earned_score
5. total_score must equal the sum of points across rubric_items
6. avg_result = earned_score / total_score, rounded to 4 decimal places

# task
{task}

# init_state
{init_state}

# final_state
{final_state}

# final_assistant_response
{final_assistant_response}

# trajectory_summary
{trajectory_summary}

# rubric_items
{rubric_items}
"""


def _read_json(file_path):
    file_path_str = str(file_path)
    
    # 针对 .jsonl 文件的逐行解析
    if file_path_str.endswith('.jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    data.append(json.loads(line))
        return data
        
    # 针对标准 .json 文件的解析
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def _save_json(path: str, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _normalize_openai_base_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    suffix = "/chat/completions"
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text


def _normalize_task_path(task_items_path: str) -> Path:
    return Path(str(task_items_path)).expanduser().resolve()


def _strip_prepared_suffix(task_path: Path) -> Path:
    stem = task_path.stem
    for token in ("_single_file_ready_", "_envscaler_ready_"):
        if token in stem:
            stem = stem.replace(token, "_")
            return task_path.with_name(stem + task_path.suffix)
    for suffix in ("_single_file_ready", "_envscaler_ready"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            return task_path.with_name(stem + task_path.suffix)
    return task_path


def _detect_rubric_companion_path(task_path: Path) -> Path | None:
    base_task_path = _strip_prepared_suffix(task_path)
    name = base_task_path.name
    if base_task_path.exists() and ("rubric_all_step1_" in name or "rubric_step1_" in name):
        return base_task_path
    prefix = "checker_step1_check_funcs_"
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    candidates = [
        base_task_path.with_name(f"rubric_all_step1_{suffix}"),
        base_task_path.with_name(f"rubric_step1_{suffix}"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _merge_task_item(base_item: Dict[str, Any], extra_item: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base_item)
    priority_keys = RUBRIC_FIELDS | {"task", "task_source"}
    for key, value in (extra_item or {}).items():
        if key in priority_keys:
            merged[key] = deepcopy(value)
            continue
        if key not in merged or merged.get(key) in (None, "", [], {}):
            merged[key] = deepcopy(value)
    return merged


def load_task_lookup(task_items_path: str) -> Dict[str, Dict[str, Any]]:
    task_path = _normalize_task_path(task_items_path)
    primary_data = _read_json(str(task_path))
    if not isinstance(primary_data, list):
        raise ValueError(f"task_items_path 应为 list，实际为 {type(primary_data)}")

    task_lookup: Dict[str, Dict[str, Any]] = {}
    for item in primary_data:
        if not isinstance(item, dict):
            continue
        # 使用 global_id 作为键而不是 task_id，因为 task_id 不唯一
        global_id = item.get("global_id")
        if global_id is not None:
            key = str(global_id)
            task_lookup[key] = deepcopy(item)
        else:
            # 降级：如果没有 global_id，使用 task_id
            task_id = str(item.get("task_id", "")).strip()
            if task_id:
                task_lookup[task_id] = deepcopy(item)

    companion_path = _detect_rubric_companion_path(task_path)
    if companion_path is None:
        return task_lookup

    companion_data = _read_json(str(companion_path))
    if not isinstance(companion_data, list):
        return task_lookup

    for item in companion_data:
        if not isinstance(item, dict):
            continue
        # 使用 global_id 作为键
        global_id = item.get("global_id")
        if global_id is not None:
            key = str(global_id)
            if key in task_lookup:
                task_lookup[key] = _merge_task_item(task_lookup[key], item)
            else:
                task_lookup[key] = deepcopy(item)
        else:
            # 降级：使用 task_id
            task_id = str(item.get("task_id", "")).strip()
            if not task_id:
                continue
            if task_id in task_lookup:
                task_lookup[task_id] = _merge_task_item(task_lookup[task_id], item)
            else:
                task_lookup[task_id] = deepcopy(item)
    return task_lookup


def _parse_rubric_lines(rubrics_text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    current_category = ""
    line_pattern = re.compile(r"^(G\d+|T\d+)\s*\|\s*([123])分\s*\|\s*(.+)$")
    for raw_line in str(rubrics_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "【General】":
            current_category = "general"
            continue
        if line == "【Task-Specific】":
            current_category = "task_specific"
            continue
        match = line_pattern.match(line)
        if not match or not current_category:
            continue
        tag, points, text = match.groups()
        out.append(
            {
                "tag": tag,
                "category": current_category,
                "points": int(points),
                "text": text.strip(),
            }
        )
    return out


def _normalize_rubric_item(item: Dict[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    tag = str(item.get("tag") or item.get("id") or "").strip()
    category = str(item.get("category") or item.get("section") or "").strip()
    text = str(item.get("text") or item.get("criteria") or "").strip()
    try:
        points = int(item.get("points", 0) or 0)
    except Exception:
        return None
    if not tag or not category or points <= 0:
        return None
    return {
        "tag": tag,
        "category": category,
        "points": points,
        "text": text,
    }


def _normalize_rubric_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items or []:
        normalized = _normalize_rubric_item(item)
        if normalized:
            out.append(normalized)
    return out


def _extract_rubric_items(task_item: Dict[str, Any], result_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    for source in [result_item.get("task_info", {}), task_item or {}, result_item]:
        if not isinstance(source, dict):
            continue
        # 兼容两种键名：rubric_items（旧）与 rubrics（route2/3 单文件数据）。
        for key in ("rubric_items", "rubrics"):
            rubric_items = source.get(key)
            if isinstance(rubric_items, list) and rubric_items:
                normalized = _normalize_rubric_items(rubric_items)
                if normalized:
                    return normalized
        rubrics_text = source.get("rubrics_text")
        if isinstance(rubrics_text, str) and rubrics_text.strip():
            parsed = _parse_rubric_lines(rubrics_text)
            if parsed:
                return _normalize_rubric_items(parsed)
    return []


def _extract_checklist_with_func(task_item: Dict[str, Any], result_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    for source in [result_item.get("task_info", {}), task_item or {}]:
        if isinstance(source, dict):
            checklist = source.get("checklist_with_func")
            if isinstance(checklist, list) and checklist:
                return deepcopy(checklist)
    return []


def _find_last_assistant_response(messages: List[Dict[str, Any]]) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _collect_candidate_answer_texts(result_item: Dict[str, Any]) -> List[str]:
    """收集可能承载最终结构化答案的文本片段（按优先级）。

    优先顺序：
    1. 最后一条 assistant 文本 content
    2. assistant 的 chat_with_user 工具调用 arguments.content（倒序）
    3. final_observation 文本
    """
    texts: List[str] = []
    messages = result_item.get("messages", [])
    if isinstance(messages, list):
        last_text = _find_last_assistant_response(messages)
        if last_text:
            texts.append(last_text)
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function", {}) if isinstance(call.get("function"), dict) else {}
                arguments = fn.get("arguments")
                if isinstance(arguments, str) and arguments.strip():
                    # FC 模式下 arguments 常是 JSON 字符串，真正答案多在 content 字段内，
                    # 优先解出内层 content，再用整串兜底。
                    try:
                        parsed_args = json.loads(arguments)
                    except Exception:
                        parsed_args = None
                    if isinstance(parsed_args, dict):
                        inner = parsed_args.get("content")
                        if isinstance(inner, str) and inner.strip():
                            texts.append(inner.strip())
                    texts.append(arguments.strip())
                elif isinstance(arguments, dict):
                    content = arguments.get("content")
                    if isinstance(content, str) and content.strip():
                        texts.append(content.strip())

    final_observation = result_item.get("final_observation")
    if isinstance(final_observation, str) and final_observation.strip():
        texts.append(final_observation.strip())
    elif isinstance(final_observation, dict):
        content = final_observation.get("content")
        if isinstance(content, str) and content.strip():
            texts.append(content.strip())

    seen = set()
    deduped: List[str] = []
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _extract_largest_json_obj(text: str) -> Any:
    """从文本中提取最大的可解析 JSON 对象/数组（用于候选答案）。"""
    raw = str(text or "").strip()
    if not raw:
        return None
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(raw)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass

    candidates: List[str] = []
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start_stack: List[int] = []
        for idx, ch in enumerate(raw):
            if ch == open_ch:
                start_stack.append(idx)
            elif ch == close_ch and start_stack:
                start = start_stack.pop()
                candidates.append(raw[start : idx + 1])
    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(candidate)
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                continue
    return None


def _extract_candidate_answer(result_item: Dict[str, Any]) -> tuple[Any, str | None]:
    """从结果中提取候选答案 JSON。返回 (candidate_answer, source_text)。"""
    for text in _collect_candidate_answer_texts(result_item):
        obj = _extract_largest_json_obj(text)
        if obj is not None:
            return obj, text
    return None, None


def run_verifier_function(
    verifier_code: str,
    candidate_answer: Any,
    ground_truth_answer: Any,
) -> tuple[bool, float | None, str | None]:
    """执行 verifier_code 中定义的 `verify(candidate_answer, ground_truth_answer) -> float`。

    返回 (success, score, error)。score 会被裁剪到 [0, 1]。
    """
    safe_globals = {"__builtins__": __builtins__}
    try:
        exec(str(verifier_code or ""), safe_globals)
        verify_fn = safe_globals.get("verify")
        if not callable(verify_fn):
            return False, None, "Function 'verify' not found in verifier_code."
        raw_score = verify_fn(deepcopy(candidate_answer), deepcopy(ground_truth_answer))
        score = float(raw_score)
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        return True, round(score, 6), None
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{type(exc).__name__}: {exc}"


def _extract_verifier_spec(
    task_item: Dict[str, Any], result_item: Dict[str, Any]
) -> tuple[str, Any]:
    """提取 verifier_code 与 ground_truth_answer。任一缺失则返回 ("", None)。"""
    verifier_code = ""
    ground_truth_answer: Any = None
    for source in [result_item.get("task_info", {}), task_item or {}, result_item]:
        if not isinstance(source, dict):
            continue
        if not verifier_code:
            code = source.get("verifier_code")
            if isinstance(code, str) and code.strip():
                verifier_code = code
        if ground_truth_answer is None:
            gt = source.get("ground_truth_answer")
            if gt is not None:
                ground_truth_answer = gt
    return verifier_code, ground_truth_answer


def evaluate_verifier(
    task_item: Dict[str, Any],
    result_item: Dict[str, Any],
) -> Dict[str, Any]:
    """对单条结果运行 verifier_code，得到程序化分数。无 verifier 时返回 {}。"""
    verifier_code, ground_truth_answer = _extract_verifier_spec(task_item, result_item)
    if not verifier_code or ground_truth_answer is None:
        return {}

    candidate_answer, source_text = _extract_candidate_answer(result_item)
    if candidate_answer is None:
        return {
            "score": 0.0,
            "parse_success": False,
            "verify_success": False,
            "candidate_answer": None,
            "error": "no_candidate_answer_json_extracted",
        }

    success, score, error = run_verifier_function(
        verifier_code=verifier_code,
        candidate_answer=candidate_answer,
        ground_truth_answer=ground_truth_answer,
    )
    return {
        "score": float(score) if score is not None else 0.0,
        "parse_success": True,
        "verify_success": bool(success),
        "candidate_answer": deepcopy(candidate_answer),
        "candidate_source_preview": str(source_text or "")[:500],
        "error": error,
    }


def _truncate_text(text: Any, max_chars: int = 1200) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = _safe_json_dumps(text)
        except Exception:
            text = str(text)
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(truncated)"


def _make_json_serializable(obj: Any, max_depth: int = 10) -> Any:
    if max_depth <= 0:
        return str(type(obj).__name__)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if callable(obj):
        module = str(getattr(obj, "__module__", "") or "")
        name = str(getattr(obj, "__name__", "") or getattr(obj, "__qualname__", "") or type(obj).__name__)
        return f"<callable:{module}.{name}>".strip(".")
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for key, value in obj.items():
            out[str(key)] = _make_json_serializable(value, max_depth=max_depth - 1)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_serializable(x, max_depth=max_depth - 1) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def _safe_json_dumps(obj: Any, indent: int = 2) -> str:
    return json.dumps(_make_json_serializable(obj), ensure_ascii=False, indent=indent)


def _build_trajectory_summary(result_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    trajectory = result_item.get("trajectory", [])
    if not isinstance(trajectory, list):
        return out
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        if "action" not in step and step.get("step") == 0:
            continue
        action = step.get("action", {})
        observation = step.get("observation", {})
        out.append(
            {
                "step": step.get("step"),
                "action": action,
                "observation_type": observation.get("type") if isinstance(observation, dict) else "",
                "observation_content": _truncate_text(
                    observation.get("content") if isinstance(observation, dict) else observation
                ),
                "reward": step.get("reward"),
                "terminated": step.get("terminated"),
                "truncated": step.get("truncated"),
            }
        )
    return out


def evaluate_checklist(
    checklist_with_func: List[Dict[str, Any]],
    init_state: Dict[str, Any],
    final_state: Dict[str, Any],
) -> Dict[str, Any]:
    checklist_with_func_result = []
    for check_item in checklist_with_func:
        if not isinstance(check_item, dict):
            continue
        check_func_str = check_item.get("check_func", "")
        success, result, error = run_check_function(
            func_code=str(check_func_str or ""),
            init_state=init_state if isinstance(init_state, dict) else {},
            final_state=final_state if isinstance(final_state, dict) else {},
        )
        new_item = deepcopy(check_item)
        new_item["check_func_result"] = {"success": success, "result": result, "error": error}
        checklist_with_func_result.append(new_item)

    valid_results = [
        item["check_func_result"]["result"]
        for item in checklist_with_func_result
        if item["check_func_result"]["result"] is not None
    ]
    if len(checklist_with_func_result) == 0:
        avg_result = 0.0
    else:
        avg_result = round(sum(valid_results) / len(checklist_with_func_result), 4)
    return {
        "count": len(checklist_with_func_result),
        "avg_result": avg_result,
        "items": checklist_with_func_result,
    }


def _extract_json_obj(text: str) -> Dict[str, Any] | None:
    raw = str(text or "").strip()
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    parsed_candidates: List[Dict[str, Any]] = []
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            parsed_candidates.append(obj)
    except Exception:
        try:
            obj = ast.literal_eval(raw)
            if isinstance(obj, dict):
                parsed_candidates.append(obj)
        except Exception:
            pass
    candidates: List[str] = []
    start_stack: List[int] = []
    for idx, ch in enumerate(raw):
        if ch == "{":
            start_stack.append(idx)
        elif ch == "}" and start_stack:
            start = start_stack.pop()
            candidates.append(raw[start : idx + 1])
    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            try:
                obj = ast.literal_eval(candidate)
            except Exception:
                continue
        if isinstance(obj, dict):
            parsed_candidates.append(obj)
    if not parsed_candidates:
        return None
    return _pick_rubric_like_obj(parsed_candidates)


def _normalize_category(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in ("task_specific", "taskspecific"):
        return "task_specific"
    if text == "general":
        return "general"
    return text


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("bool is not valid int")
    if isinstance(value, (int, float)):
        return int(float(value))
    text = str(value or "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"cannot parse int from {value!r}")
    return int(float(match.group(0)))


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("bool is not valid float")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if text.endswith("%"):
        return float(text[:-1].strip()) / 100.0
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"cannot parse float from {value!r}")
    return float(match.group(0))


def _coerce_bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "false"):
            return text == "true"
    raise ValueError(f"cannot parse bool from {value!r}")


def _pick_rubric_like_obj(candidates: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    preferred_keys = ("items", "earned_score", "total_score")
    for obj in candidates:
        if all(k in obj for k in preferred_keys):
            return obj
    for obj in candidates:
        for key in ("rubric_eval", "result", "data", "output"):
            nested = obj.get(key)
            if isinstance(nested, dict) and all(k in nested for k in preferred_keys):
                return nested
    return candidates[0] if candidates else None


def _normalize_rubric_eval_obj(
    obj: Dict[str, Any], rubric_items: List[Dict[str, Any]]
) -> tuple[Dict[str, Any] | None, str]:
    if not isinstance(obj, dict):
        return None, "parsed obj is not dict"
    items = obj.get("items")
    if not isinstance(items, list) or len(items) != len(rubric_items):
        return None, "items length mismatch"

    expected_total = sum(_coerce_int(item.get("points", 0) or 0) for item in rubric_items)

    expected_by_tag = {str(item.get("tag", "")): item for item in rubric_items}
    if len(expected_by_tag) != len(rubric_items):
        return None, "duplicated expected tags"

    actual_by_tag: Dict[str, Dict[str, Any]] = {}
    for actual in items:
        if not isinstance(actual, dict):
            return None, "item is not dict"
        actual_tag = str(actual.get("tag") or actual.get("id") or "")
        if not actual_tag:
            return None, "missing tag/id in item"
        actual_by_tag[actual_tag] = actual

    if set(actual_by_tag.keys()) != set(expected_by_tag.keys()):
        return None, "item tags set mismatch"

    item_earned_sum = 0
    normalized_items: List[Dict[str, Any]] = []
    for tag, expected in expected_by_tag.items():
        actual = actual_by_tag[tag]
        if _normalize_category(actual.get("category") or actual.get("section")) != _normalize_category(
            expected.get("category")
        ):
            return None, f"category mismatch: expected {expected.get('category')}, got {actual.get('category')}"
        try:
            actual_points = _coerce_int(actual.get("points", 0) or 0)
            expected_points = _coerce_int(expected.get("points", 0) or 0)
        except Exception:
            return None, f"points not numeric for {tag}"
        if actual_points != expected_points:
            return None, f"points mismatch for {tag}"
        try:
            passed = _coerce_bool_like(actual.get("passed"))
        except Exception:
            return None, f"passed is not bool-like for {tag}"
        try:
            earned = _coerce_int(actual.get("earned_score", -1))
        except Exception:
            return None, f"earned_score not numeric for {tag}"
        if earned not in (0, expected_points):
            return None, f"earned_score invalid for {tag}"
        if passed and earned == 0:
            return None, f"passed=true but earned_score=0 for {tag}"
        if (not passed) and earned != 0:
            return None, f"passed=false but earned_score>0 for {tag}"
        item_earned_sum += earned
        normalized_items.append(
            {
                "tag": tag,
                "category": expected.get("category"),
                "points": expected_points,
                "passed": passed,
                "earned_score": earned,
                "reason": str(actual.get("reason") or "").strip(),
            }
        )

    expected_avg = round((item_earned_sum / expected_total), 4) if expected_total > 0 else 0.0
    normalized_obj = {
        "earned_score": item_earned_sum,
        "total_score": expected_total,
        "avg_result": expected_avg,
        "summary": str(obj.get("summary") or "").strip(),
        "items": normalized_items,
    }

    top_level_mismatches: List[str] = []
    try:
        raw_earned_score = _coerce_int(obj.get("earned_score"))
        if raw_earned_score != item_earned_sum:
            top_level_mismatches.append(
                f"earned_score mismatch: expected sum {item_earned_sum}, got {raw_earned_score}"
            )
    except Exception:
        top_level_mismatches.append("earned_score invalid")

    try:
        raw_total_score = _coerce_int(obj.get("total_score"))
        if raw_total_score != expected_total:
            top_level_mismatches.append(
                f"total_score mismatch: expected {expected_total}, got {raw_total_score}"
            )
    except Exception:
        top_level_mismatches.append("total_score invalid")

    try:
        raw_avg_result = _coerce_float(obj.get("avg_result"))
        if abs(raw_avg_result - expected_avg) > 1e-4:
            top_level_mismatches.append(
                f"avg_result mismatch: expected {expected_avg}, got {raw_avg_result}"
            )
    except Exception:
        top_level_mismatches.append("avg_result invalid")

    return normalized_obj, "; ".join(top_level_mismatches)


def evaluate_rubric(
    task_text: str,
    init_state: Dict[str, Any],
    final_state: Dict[str, Any],
    result_item: Dict[str, Any],
    rubric_items: List[Dict[str, Any]],
    judge_config: Dict[str, Any],
    lang: str = "cn",
) -> Dict[str, Any]:
    rubric_items = _normalize_rubric_items(rubric_items)
    if not rubric_items:
        return {}

    messages = [
        {
            "role": "user",
            "content": (RUBRIC_JUDGE_TEMPLATE_EN if lang == "en" else RUBRIC_JUDGE_TEMPLATE).format(
                task=str(task_text or ""),
                init_state=_safe_json_dumps(init_state if isinstance(init_state, dict) else {}),
                final_state=_safe_json_dumps(final_state if isinstance(final_state, dict) else {}),
                final_assistant_response=_find_last_assistant_response(result_item.get("messages", [])),
                trajectory_summary=_safe_json_dumps(_build_trajectory_summary(result_item)),
                rubric_items=_safe_json_dumps(rubric_items),
            ),
        }
    ]

    max_try = 3
    cur_try = 0
    parsed_obj: Dict[str, Any] = {}
    attempt_logs: List[Dict[str, Any]] = []
    while cur_try < max_try:
        cur_try += 1
        api_response = openai_inference_prompt(
            model=str(judge_config.get("model", "")),
            messages=messages,
            temperature=0.0,
            enable_thinking=bool(judge_config.get("enable_thinking", False)),
            api_key=str(judge_config.get("api_key", "") or ""),
            base_url=_normalize_openai_base_url(str(judge_config.get("base_url", "") or "")),
        )
        output_text = api_response.get("content", "") if isinstance(api_response, dict) else str(api_response)
        obj = _extract_json_obj(output_text)
        if obj:
            normalized_obj, validation_error = _normalize_rubric_eval_obj(obj, rubric_items)
            is_valid = normalized_obj is not None
        else:
            normalized_obj = None
            is_valid, validation_error = False, "no valid json object extracted"
        attempt_logs.append(
            {
                "try_index": cur_try,
                "raw_response": str(output_text or ""),
                "parsed_obj": deepcopy(obj) if isinstance(obj, dict) else None,
                "normalized_obj": deepcopy(normalized_obj) if isinstance(normalized_obj, dict) else None,
                "validation_error": validation_error,
            }
        )
        if normalized_obj and is_valid:
            parsed_obj = normalized_obj
            break

    if not parsed_obj:
        return {
            "count": len(rubric_items),
            "earned_score": 0,
            "total_score": sum(int(item.get("points", 0) or 0) for item in rubric_items),
            "avg_result": 0.0,
            "items": [],
            "summary": "",
            "judge_meta": {
                "parse_success": False,
                "llm_calls": cur_try,
                "judge_model": str(judge_config.get("model", "")),
                "attempt_logs": attempt_logs,
            },
        }

    parsed_obj["count"] = len(rubric_items)
    parsed_obj["judge_meta"] = {
        "parse_success": True,
        "llm_calls": cur_try,
        "judge_model": str(judge_config.get("model", "")),
        "attempt_logs": attempt_logs,
    }
    return parsed_obj


def augment_result_with_evaluations(
    result_item: Dict[str, Any],
    task_lookup: Dict[str, Dict[str, Any]] | None = None,
    enable_checklist_eval: bool = False,
    enable_rubric_eval: bool = True,
    force_recompute_checklist_eval: bool = False,
    force_recompute_rubric_eval: bool = False,
    rubric_judge_config: Dict[str, Any] | None = None,
    lang: str = "cn",
    enable_verifier_eval: bool = True,
    force_recompute_verifier_eval: bool = False,
) -> Dict[str, Any]:
    out = deepcopy(result_item if isinstance(result_item, dict) else {})
    out.pop("checklist_eval", None)
    task_info = out.get("task_info", {}) if isinstance(out.get("task_info"), dict) else {}

    # 优先使用 global_id 查找 task_item，因为 task_id 可能不唯一
    global_id = task_info.get("global_id")
    if global_id is not None:
        lookup_key = str(global_id)
    else:
        lookup_key = str(task_info.get("task_id", "")).strip()

    task_item = deepcopy((task_lookup or {}).get(lookup_key, {}))
    if not isinstance(task_item, dict):
        task_item = {}

    final_info = deepcopy(out.get("final_info", {}) if isinstance(out.get("final_info"), dict) else {})

    init_state = out.get("init_state", {})
    final_state = out.get("final_state", {})
    if not isinstance(init_state, dict):
        init_state = deepcopy(task_item.get("init_config", {})) if isinstance(task_item.get("init_config"), dict) else {}
    if not isinstance(final_state, dict):
        final_state = {}

    final_info.pop("checklist_eval", None)
    final_info.pop("checklist_eval_error", None)

    if enable_rubric_eval and rubric_judge_config:
        need_rubric = force_recompute_rubric_eval or not isinstance(out.get("rubric_eval"), dict)
        if need_rubric:
            # 当强制重新计算时，从 out 中移除所有可能包含旧 rubric 的字段
            if force_recompute_rubric_eval:
                out.pop("task_info", None)
                out.pop("rubric_items", None)
                out.pop("rubrics", None)
                out.pop("rubrics_text", None)
            rubric_items = _extract_rubric_items(task_item, out)
            if rubric_items:
                try:
                    rubric_eval = evaluate_rubric(
                        task_text=str(task_info.get("task") or task_item.get("task") or ""),
                        init_state=init_state,
                        final_state=final_state,
                        result_item=out,
                        rubric_items=rubric_items,
                        judge_config=rubric_judge_config,
                        lang=lang,
                    )
                    out["rubric_eval"] = rubric_eval
                    out["rubric_reward"] = float(rubric_eval.get("avg_result", 0.0) or 0.0)
                    out["total_reward"] = out["rubric_reward"]
                    final_info["rubric_eval"] = deepcopy(rubric_eval)
                except Exception as exc:
                    final_info["rubric_eval_error"] = {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }

    if enable_verifier_eval:
        need_verifier = force_recompute_verifier_eval or not isinstance(out.get("verifier_eval"), dict)
        if need_verifier:
            try:
                verifier_eval = evaluate_verifier(task_item=task_item, result_item=out)
                if verifier_eval:
                    out["verifier_eval"] = verifier_eval
                    out["verifier_reward"] = float(verifier_eval.get("score", 0.0) or 0.0)
                    # 第三路：verifier 为主信号，total_reward 取 verifier 分数。
                    out["total_reward"] = out["verifier_reward"]
                    final_info["verifier_eval"] = deepcopy(verifier_eval)
            except Exception as exc:
                final_info["verifier_eval_error"] = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }

    out["final_info"] = final_info
    return out


def _sample_sort_key(item: Dict[str, Any]) -> tuple[int, int]:
    task_info = item.get("task_info", {}) if isinstance(item.get("task_info"), dict) else {}
    raw_idx = task_info.get("sample_idx", item.get("sample_idx", 0))
    try:
        sample_idx = int(raw_idx)
    except Exception:
        sample_idx = 0
    return sample_idx, 0


def _extract_rubric_score(result_item: Dict[str, Any]) -> float | None:
    rubric_eval = result_item.get("rubric_eval")
    if not isinstance(rubric_eval, dict):
        return None
    avg_result = rubric_eval.get("avg_result")
    if avg_result is None:
        return None
    try:
        return float(avg_result)
    except Exception:
        return None


def _extract_verifier_score(result_item: Dict[str, Any]) -> float | None:
    verifier_eval = result_item.get("verifier_eval")
    if not isinstance(verifier_eval, dict):
        return None
    score = verifier_eval.get("score")
    if score is None:
        return None
    try:
        return float(score)
    except Exception:
        return None


def _score_to_full_pass(score: float | None) -> bool | None:
    if score is None:
        return None
    return abs(float(score) - 1.0) < 1e-6


def _extract_combined_score_for_pass_k(result_item: Dict[str, Any]) -> tuple[str | None, float | None]:
    """综合 pass@k 口径：第三路有 verifier 时以 verifier 为准，否则用 rubric。"""
    verifier_score = _extract_verifier_score(result_item)
    if verifier_score is not None:
        return "verifier", verifier_score
    rubric_score = _extract_rubric_score(result_item)
    if rubric_score is not None:
        return "rubric", rubric_score
    return None, None


def _build_pass_symbol_sequence(pass_sequence: List[bool | None]) -> str:
    symbol_map = {
        True: "✓",
        False: "✗",
        None: "·",
    }
    return "".join(symbol_map[item] for item in pass_sequence)


def summarize_pass_k_group(group_item: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(group_item if isinstance(group_item, dict) else {})
    samples = out.get("samples")
    if not isinstance(samples, list):
        return out

    normalized_samples = [deepcopy(item) for item in samples if isinstance(item, dict)]
    normalized_samples.sort(key=_sample_sort_key)
    out["samples"] = normalized_samples
    out["k"] = len(normalized_samples)

    if normalized_samples:
        first_sample = normalized_samples[0]
        task_info = first_sample.get("task_info", {}) if isinstance(first_sample.get("task_info"), dict) else {}
        out["task_id"] = str(out.get("task_id") or task_info.get("task_id") or "").strip()
        out["env_id"] = str(out.get("env_id") or task_info.get("env_id") or "").strip()
        out["task"] = str(out.get("task") or task_info.get("task") or "")
        out["task_info"] = deepcopy(task_info)

    sample_idx_sequence: List[int | None] = []
    rubric_pass_sequence: List[bool | None] = []
    verifier_pass_sequence: List[bool | None] = []
    combined_source_sequence: List[str | None] = []
    combined_pass_sequence: List[bool | None] = []
    rubric_scores: List[float] = []
    verifier_scores: List[float] = []
    combined_scores: List[float] = []

    for sample in normalized_samples:
        task_info = sample.get("task_info", {}) if isinstance(sample.get("task_info"), dict) else {}
        raw_sample_idx = task_info.get("sample_idx", sample.get("sample_idx"))
        try:
            sample_idx = int(raw_sample_idx)
        except Exception:
            sample_idx = None
        sample_idx_sequence.append(sample_idx)

        rubric_score = _extract_rubric_score(sample)
        verifier_score = _extract_verifier_score(sample)
        combined_source, combined_score = _extract_combined_score_for_pass_k(sample)

        if rubric_score is not None:
            rubric_scores.append(rubric_score)
        if verifier_score is not None:
            verifier_scores.append(verifier_score)
        if combined_score is not None:
            combined_scores.append(combined_score)

        rubric_pass_sequence.append(_score_to_full_pass(rubric_score))
        verifier_pass_sequence.append(_score_to_full_pass(verifier_score))
        combined_source_sequence.append(combined_source)
        combined_pass_sequence.append(_score_to_full_pass(combined_score))

    rubric_pass_count = sum(1 for score in rubric_scores if abs(score - 1.0) < 1e-6)
    verifier_pass_count = sum(1 for score in verifier_scores if abs(score - 1.0) < 1e-6)
    combined_pass_count = sum(1 for score in combined_scores if abs(score - 1.0) < 1e-6)

    if verifier_scores and rubric_scores:
        combined_rule = "verifier_over_rubric"
    elif verifier_scores:
        combined_rule = "verifier_only"
    else:
        combined_rule = "rubric_only"

    out["pass_k_summary"] = {
        "k": len(normalized_samples),
        "sample_idx_sequence": sample_idx_sequence,
        "rubric_valid_run_count": len(rubric_scores),
        "rubric_pass_count": rubric_pass_count,
        "rubric_pass_at_k": (rubric_pass_count > 0) if rubric_scores else None,
        "rubric_best_avg_result": round(max(rubric_scores), 4) if rubric_scores else None,
        "rubric_mean_avg_result": round(sum(rubric_scores) / len(rubric_scores), 4) if rubric_scores else None,
        "rubric_pass_sequence": rubric_pass_sequence,
        "rubric_pass_pattern": _build_pass_symbol_sequence(rubric_pass_sequence),
        "verifier_valid_run_count": len(verifier_scores),
        "verifier_pass_count": verifier_pass_count,
        "verifier_pass_at_k": (verifier_pass_count > 0) if verifier_scores else None,
        "verifier_best_score": round(max(verifier_scores), 4) if verifier_scores else None,
        "verifier_mean_score": round(sum(verifier_scores) / len(verifier_scores), 4) if verifier_scores else None,
        "verifier_pass_sequence": verifier_pass_sequence,
        "verifier_pass_pattern": _build_pass_symbol_sequence(verifier_pass_sequence),
        "combined_rule": combined_rule,
        "combined_valid_run_count": len(combined_scores),
        "combined_pass_count": combined_pass_count,
        "combined_pass_at_k": (combined_pass_count > 0) if combined_scores else None,
        "combined_best_avg_result": round(max(combined_scores), 4) if combined_scores else None,
        "combined_mean_avg_result": round(sum(combined_scores) / len(combined_scores), 4) if combined_scores else None,
        "combined_source_sequence": combined_source_sequence,
        "combined_pass_sequence": combined_pass_sequence,
        "combined_pass_pattern": _build_pass_symbol_sequence(combined_pass_sequence),
    }
    return out


def augment_group_with_evaluations(
    group_item: Dict[str, Any],
    task_lookup: Dict[str, Dict[str, Any]] | None = None,
    enable_checklist_eval: bool = False,
    enable_rubric_eval: bool = True,
    force_recompute_checklist_eval: bool = False,
    force_recompute_rubric_eval: bool = False,
    rubric_judge_config: Dict[str, Any] | None = None,
    lang: str = "cn",
    enable_verifier_eval: bool = True,
    force_recompute_verifier_eval: bool = False,
) -> Dict[str, Any]:
    out = deepcopy(group_item if isinstance(group_item, dict) else {})
    samples = out.get("samples")
    if not isinstance(samples, list):
        return out
    out["samples"] = [
        augment_result_with_evaluations(
            result_item=item,
            task_lookup=task_lookup,
            enable_checklist_eval=enable_checklist_eval,
            enable_rubric_eval=enable_rubric_eval,
            force_recompute_checklist_eval=force_recompute_checklist_eval,
            force_recompute_rubric_eval=force_recompute_rubric_eval,
            rubric_judge_config=rubric_judge_config,
            lang=lang,
            enable_verifier_eval=enable_verifier_eval,
            force_recompute_verifier_eval=force_recompute_verifier_eval,
        )
        for item in samples
        if isinstance(item, dict)
    ]
    return summarize_pass_k_group(out)


def evaluate_result_file(
    result_file_path: str,
    task_items_path: str,
    save_file_path: str | None = None,
    num_workers: int = 1,
    enable_checklist_eval: bool = False,
    enable_rubric_eval: bool = True,
    force_recompute_checklist_eval: bool = False,
    force_recompute_rubric_eval: bool = False,
    rubric_judge_config: Dict[str, Any] | None = None,
    lang: str = "cn",
    enable_verifier_eval: bool = True,
    force_recompute_verifier_eval: bool = False,
):
    result_data = _read_json(result_file_path)
    task_lookup = load_task_lookup(task_items_path)

    if isinstance(result_data, dict) and isinstance(result_data.get("samples"), list):
        evaluated = augment_group_with_evaluations(
            group_item=result_data,
            task_lookup=task_lookup,
            enable_checklist_eval=enable_checklist_eval,
            enable_rubric_eval=enable_rubric_eval,
            force_recompute_checklist_eval=force_recompute_checklist_eval,
            force_recompute_rubric_eval=force_recompute_rubric_eval,
            rubric_judge_config=rubric_judge_config,
            lang=lang,
            enable_verifier_eval=enable_verifier_eval,
            force_recompute_verifier_eval=force_recompute_verifier_eval,
        )
        _save_json(save_file_path or result_file_path, evaluated)
        return evaluated

    if isinstance(result_data, dict):
        evaluated = augment_result_with_evaluations(
            result_item=result_data,
            task_lookup=task_lookup,
            enable_checklist_eval=enable_checklist_eval,
            enable_rubric_eval=enable_rubric_eval,
            force_recompute_checklist_eval=force_recompute_checklist_eval,
            force_recompute_rubric_eval=force_recompute_rubric_eval,
            rubric_judge_config=rubric_judge_config,
            lang=lang,
            enable_verifier_eval=enable_verifier_eval,
            force_recompute_verifier_eval=force_recompute_verifier_eval,
        )
        _save_json(save_file_path or result_file_path, evaluated)
        return evaluated

    if not isinstance(result_data, list):
        raise ValueError(f"result_file_path 应为 list 或 dict，实际为 {type(result_data)}")

    if result_data and all(isinstance(item, dict) and isinstance(item.get("samples"), list) for item in result_data):
        group_result_dict: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
            future_to_index = {
                executor.submit(
                    augment_group_with_evaluations,
                    item,
                    task_lookup,
                    enable_checklist_eval,
                    enable_rubric_eval,
                    force_recompute_checklist_eval,
                    force_recompute_rubric_eval,
                    rubric_judge_config,
                ): idx
                for idx, item in enumerate(result_data)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                group_result_dict[idx] = future.result()

        evaluated = [group_result_dict[i] for i in sorted(group_result_dict.keys())]
        _save_json(save_file_path or result_file_path, evaluated)
        return evaluated

    result_dict: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
        future_to_index = {
            executor.submit(
                augment_result_with_evaluations,
                item,
                task_lookup,
                enable_checklist_eval,
                enable_rubric_eval,
                force_recompute_checklist_eval,
                force_recompute_rubric_eval,
                rubric_judge_config,
            ): idx
            for idx, item in enumerate(result_data)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            result_dict[idx] = future.result()

    evaluated = [result_dict[i] for i in sorted(result_dict.keys())]
    _save_json(save_file_path or result_file_path, evaluated)
    return evaluated
