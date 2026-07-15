#!/usr/bin/env python3
"""Self-contained per-route score extraction from aggregated result JSON.

No heavy deps (no openai / envscaler imports) so it can run anywhere, including
the orchestrator's post-run summary and the standalone analyze step.

Combined-score precedence per item (mirrors result_evaluator):
    verifier_eval.score  ->  rubric_eval.avg_result  ->  total_reward
A task "passes" when its sample-1 combined score == 1.0 (within tol).

Result JSON shapes handled:
- pass_k == 1: a single result-item dict, or a list of result-item dicts.
- pass_k  > 1: a list of group dicts (each has 'samples' + 'pass_k_summary'),
  or a single group dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TOL = 1e-6


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _combined_score_from_item(item: dict) -> tuple[str | None, float | None]:
    """verifier -> rubric -> total_reward. Returns (source, score)."""
    if not isinstance(item, dict):
        return None, None
    verifier_eval = item.get("verifier_eval")
    if isinstance(verifier_eval, dict):
        score = _coerce_float(verifier_eval.get("score"))
        if score is not None:
            return "verifier", score
    rubric_eval = item.get("rubric_eval")
    if isinstance(rubric_eval, dict):
        score = _coerce_float(rubric_eval.get("avg_result"))
        if score is not None:
            return "rubric", score
    total_reward = _coerce_float(item.get("total_reward"))
    if total_reward is not None:
        return "total_reward", total_reward
    return None, None


def _is_full_pass(score: float | None) -> bool:
    return score is not None and abs(score - 1.0) < TOL


def _normalize_payload(payload: Any) -> list[dict]:
    """Return a list of top-level entries (group dicts or result-item dicts)."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _is_group(entry: dict) -> bool:
    return isinstance(entry.get("samples"), list) or isinstance(entry.get("pass_k_summary"), dict)


def _sample1_item(entry: dict) -> dict:
    """For a group, pick the sample with sample_idx==1 (fallback: first)."""
    samples = entry.get("samples")
    if isinstance(samples, list) and samples:
        for s in samples:
            if isinstance(s, dict):
                info = s.get("task_info") if isinstance(s.get("task_info"), dict) else {}
                if int(s.get("sample_idx") or info.get("sample_idx") or 0) == 1:
                    return s
        for s in samples:
            if isinstance(s, dict):
                return s
    return entry


def _entry_global_id(entry: dict) -> Any:
    if entry.get("global_id") is not None:
        return entry.get("global_id")
    info = entry.get("task_info") if isinstance(entry.get("task_info"), dict) else {}
    if info.get("global_id") is not None:
        return info.get("global_id")
    s1 = _sample1_item(entry)
    if isinstance(s1, dict):
        if s1.get("global_id") is not None:
            return s1.get("global_id")
        si = s1.get("task_info") if isinstance(s1.get("task_info"), dict) else {}
        return si.get("global_id")
    return None


def summarize_route(result_file_path: str) -> dict[str, Any]:
    """Compute pass@1, mean score, and (if present) pass@k for one route file."""
    payload = _read_json(result_file_path)
    entries = _normalize_payload(payload)

    task_count = 0
    pass1_count = 0
    score_sum = 0.0
    scored_count = 0
    source_counter: dict[str, int] = {}

    passk_available = False
    passk_pass_count = 0
    passk_task_count = 0
    k_value = 1

    for entry in entries:
        task_count += 1
        if _is_group(entry):
            summary = entry.get("pass_k_summary") if isinstance(entry.get("pass_k_summary"), dict) else {}
            k_value = max(k_value, int(entry.get("k") or summary.get("k") or 1))
            s1 = _sample1_item(entry)
            source, score = _combined_score_from_item(s1)
            # pass@k from summary if available
            combined_pass_at_k = summary.get("combined_pass_at_k")
            if combined_pass_at_k is not None:
                passk_available = True
                passk_task_count += 1
                if bool(combined_pass_at_k):
                    passk_pass_count += 1
        else:
            source, score = _combined_score_from_item(entry)

        if source:
            source_counter[source] = source_counter.get(source, 0) + 1
        if score is not None:
            score_sum += score
            scored_count += 1
        if _is_full_pass(score):
            pass1_count += 1

    pass_at_1 = (pass1_count / task_count) if task_count else 0.0
    mean_score = (score_sum / scored_count) if scored_count else 0.0
    dominant_source = max(source_counter, key=source_counter.get) if source_counter else None

    summary: dict[str, Any] = {
        "result_file_path": str(result_file_path),
        "task_count": task_count,
        "scored_count": scored_count,
        "pass_at_1": round(pass_at_1, 4),
        "pass_at_1_count": pass1_count,
        "score": round(mean_score, 4),
        "score_source": dominant_source,
        "k": k_value,
    }
    if passk_available and passk_task_count:
        summary["pass_at_k"] = round(passk_pass_count / passk_task_count, 4)
        summary["pass_at_k_count"] = passk_pass_count
    return summary


def print_route_table(route_summaries: list[dict[str, Any]]) -> None:
    """Pretty-print the per-route pass@1 / score table."""
    print("")
    print("=" * 78)
    print("PER-ROUTE SCORES")
    print("=" * 78)
    header = f"{'route':<10} {'name':<18} {'tasks':>6} {'pass@1':>8} {'score':>8} {'src':>12}"
    print(header)
    print("-" * 78)
    ok_rows = []
    for row in route_summaries:
        passk_str = ""
        if row.get("pass_at_k") is not None:
            passk_str = f"  pass@{row.get('k', '?')}={row['pass_at_k']:.4f}"
        if row.get("status") and row["status"] != "ok":
            print(
                f"{row.get('route_id', '-'):<10} {row.get('name', '-'):<18} "
                f"{'-':>6} {'-':>8} {'-':>8} {'-':>12}   [{row['status']}]"
            )
            continue
        ok_rows.append(row)
        print(
            f"{row.get('route_id', '-'):<10} {row.get('name', '-'):<18} "
            f"{row.get('task_count', 0):>6} {row.get('pass_at_1', 0.0):>8.4f} "
            f"{row.get('score', 0.0):>8.4f} {str(row.get('score_source') or '-'):>12}"
            f"{passk_str}"
        )
    # 四路平权 overall：各路 pass@1 / score 的算术平均（每路权重相同，与各路条数无关）
    if ok_rows:
        print("-" * 78)
        n = len(ok_rows)
        overall_pass1 = sum(r.get("pass_at_1", 0.0) for r in ok_rows) / n
        overall_score = sum(r.get("score", 0.0) for r in ok_rows) / n
        passk_rows = [r for r in ok_rows if r.get("pass_at_k") is not None]
        overall_passk_str = ""
        if passk_rows:
            overall_passk = sum(r["pass_at_k"] for r in passk_rows) / len(passk_rows)
            k_disp = passk_rows[0].get("k", "?")
            overall_passk_str = f"  pass@{k_disp}={overall_passk:.4f}"
        print(
            f"{'OVERALL':<10} {'(equal-weighted)':<18} "
            f"{n:>6} {overall_pass1:>8.4f} {overall_score:>8.4f} {'-':>12}"
            f"{overall_passk_str}"
        )
    print("=" * 78)
    print("")


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Summarize per-route scores from an aggregated result JSON.")
    parser.add_argument("--result-file-path", required=True)
    parser.add_argument("--route-id", default="")
    parser.add_argument("--name", default="")
    return parser


def main():
    args = build_parser().parse_args()
    summary = summarize_route(args.result_file_path)
    summary["route_id"] = args.route_id or "route"
    summary["name"] = args.name or Path(args.result_file_path).stem
    summary["status"] = "ok"
    print_route_table([summary])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
