"""
code 策略：
- 提供 Python 执行器工具；
- 为 step4 提供 prompt 附加规则与 DAG 插入辅助；
- 执行器保证“任何报错都作为返回值返回，不抛异常中断”。
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Dict, List


CODE_CAPABILITY_TAG = "python_executor"
PYTHON_EXECUTOR_NAME = "python_executor"


def _normalize_language_mode(language_mode: str | None) -> str:
    text = str(language_mode or "").strip().lower()
    if text.startswith("en"):
        return "en"
    return "zh"


def build_code_operation_spec(language_mode: str = "zh") -> Dict[str, Any]:
    english_mode = _normalize_language_mode(language_mode) == "en"
    operation_description = (
        "Execute a block of Python code and return stdout, stderr, return_value, and error information. "
        "Useful for aggregation, complex filtering, structure transformation, local validation, and programmatic reasoning."
        if english_mode
        else "执行一段 Python 代码并返回 stdout、stderr、return_value 与错误信息。适合做聚合计算、复杂筛选、结构转换、局部验证与程序化推导。"
    )
    docstring_line = (
        "Execute Python code and always return stdout/stderr/exceptions in the response payload."
        if english_mode
        else "执行 Python 代码，并把 stdout / stderr / 异常全部作为返回值返回。"
    )
    return {
        "operation_name": PYTHON_EXECUTOR_NAME,
        "operation_description": operation_description,
        "operation_type": "state_change",
        "capability_tags": ["python_executor", "code"],
        "adapter_backed": True,
        "system_tool_kind": "python_executor",
        "code": """def python_executor(
    self,
    code: str,
    args: dict | None = None,
    timeout_seconds: int = 8,
) -> dict:
    \"\"\"
    """ + docstring_line + """
    \"\"\"
    try:
        from adapter import run_python_executor
        return run_python_executor(self, code=code, args=args or {}, timeout_seconds=timeout_seconds)
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "error": str(e),
            "return_value": None,
        }
""",
    }


def run_python_executor(self_obj, code: str, args: Dict[str, Any] | None = None, timeout_seconds: int = 8) -> Dict[str, Any]:
    local_args = dict(args or {})
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    result_value = None
    error_message = ""
    safe_globals = {"__builtins__": __builtins__}
    safe_locals = {"args": local_args, "result": None}
    try:
        compiled = compile(str(code or ""), "<python_executor>", "exec")
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(compiled, safe_globals, safe_locals)
        result_value = safe_locals.get("result")
    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
    return {
        "success": (error_message == ""),
        "stdout": stdout_buffer.getvalue()[:12000],
        "stderr": stderr_buffer.getvalue()[:12000],
        "error": error_message,
        "return_value": result_value,
        "args": local_args,
        "timeout_seconds": int(timeout_seconds),
    }


def build_code_chain_prompt_addendum(min_python_steps: int, max_python_steps: int, language_mode: str = "zh") -> str:
    if _normalize_language_mode(language_mode) == "en":
        return f"""

Code strategy addendum (apply only when python_executor is enabled):
1. You may insert `python_executor` as a real tool node between adjacent tools instead of a plain reasoning node;
2. Insert python_executor only when one of the following is truly needed:
   - programmatic filtering / ranking / aggregation over multiple candidates
   - complex structure transformation, field reshaping, or temporary statistics
   - a block of code is required to convert upstream outputs into inputs consumable by downstream tools
3. If code strategy is enabled, prefer constructing at least {int(min_python_steps)} python_executor step(s);
4. A single DAG should usually include no more than {int(max_python_steps)} python_executor step(s);
5. In step4, do not output final runnable code; only explain why it is necessary and where it should be inserted.

When outputting JSON, you may additionally include:
"python_executor_insertions": [
  {{
    "after_tool_name": "...",
    "before_tool_name": "...",
    "purpose": "...",
    "necessity_reason": "..."
  }}
]
"""
    return f"""

Code 策略附加要求（仅在启用 python_executor 时执行）：
1. 你可以在相邻工具之间插入 `python_executor` 这一个真实工具节点，而不是普通 reasoning 节点；
2. 仅当存在以下需求时才应插入 python_executor：
   - 多候选结果的程序化筛选/排序/聚合
   - 复杂结构转换、字段重组、临时统计
   - 只有执行一段代码后才能把上游输出转成下游可消费输入
3. 若启用 code 策略，优先构造并至少插入 {int(min_python_steps)} 个 python_executor 步骤；
4. 单条 DAG 中建议 python_executor 插入总数不超过 {int(max_python_steps)} 个；
5. step4 阶段不要给出最终可运行代码，只需要说明它为什么必要以及插在什么位置。

输出 JSON 时，可额外输出：
"python_executor_insertions": [
  {{
    "after_tool_name": "...",
    "before_tool_name": "...",
    "purpose": "...",
    "necessity_reason": "..."
  }}
]
"""


def parse_python_executor_name(node: Dict[str, Any]) -> bool:
    if not isinstance(node, dict):
        return False
    name = str(node.get("name", "")).strip()
    if name == PYTHON_EXECUTOR_NAME:
        return True
    tags = {str(x).strip() for x in (node.get("capability_tags", []) or [])}
    return CODE_CAPABILITY_TAG in tags


def build_python_executor_placeholder_args(purpose: str = "") -> Dict[str, Any]:
    return {
        "code": "# 请根据当前任务与前序输出生成可运行 Python 代码\nresult = {}",
        "args": {"purpose": str(purpose or "").strip()},
        "timeout_seconds": 8,
    }
