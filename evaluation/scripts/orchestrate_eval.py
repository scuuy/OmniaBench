#!/usr/bin/env python3
"""Three-route eval orchestrator.

For each selected route:
  1. resolve its data file (data_file, else fallback_data_files);
  2. assign deterministic global_ids -> write a prepared task_items json;
  3. if the route needs a sandbox, copy a FRESH sandbox from the fixed
     runtime_assets bundle into a per-run target dir and point
     ENVSCALER_FS_BUNDLE_ROOT at it (whole framework stays portable);
  4. invoke scripts/run_eval.py with the route's env_name + the chosen profile;
  5. summarize per-route pass@1 + score.

After all routes finish, print one combined per-route table (pass@1 AND score),
which is the behavior required after every eval run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

def _get_beijing_timestamp() -> str:
    """返回北京时间的时间戳字符串 (YYYYMMDD-HHMMSS)。"""
    now = time.time()
    beijing_time = time.gmtime(now + 8 * 3600)
    return time.strftime("%Y%m%d-%H%M%S", beijing_time)

SCRIPT_DIR = Path(__file__).resolve().parent
FRAMEWORK_ROOT = SCRIPT_DIR.parent
CONFIGS_DIR = FRAMEWORK_ROOT / "configs"
RUNTIME_DIR = FRAMEWORK_ROOT / "runtime"
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIXED_SANDBOX_BUNDLE = FRAMEWORK_ROOT / "runtime_assets" / "fs_bundle"
SANDBOX_RUN_ROOT = RUNTIME_DIR / "tmp" / "sandbox_runs"
PREPARED_DIR = FRAMEWORK_ROOT / "data" / "routes" / "_prepared"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from assign_route_ids import prepare_route_file  # noqa: E402
from route_scores import summarize_route, print_route_table  # noqa: E402


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _merge_all_route_runs_jsonl(result_files: dict[str, str], summaries: list[dict], merged_path: Path) -> None:
    """Merge all route .runs.jsonl files into one combined file."""
    merged_count = 0
    with open(merged_path, "w", encoding="utf-8") as out_f:
        for route_id, result_path in sorted(result_files.items()):
            runs_jsonl_path = Path(result_path).with_suffix('.runs.jsonl')
            if runs_jsonl_path.exists():
                with open(runs_jsonl_path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if line:
                            out_f.write(line + "\n")
                            merged_count += 1

    # 合并 state_diffs 到统一目录（放在 base_output_dir 内）
    base_dir = merged_path.parent.parent  # <profile>-<run_tag>/
    combined_state_diff_dir = base_dir / "state_diffs"
    combined_state_diff_dir.mkdir(parents=True, exist_ok=True)

    import glob
    for summary in summaries:
        if summary.get("status") == "ok":
            route_save_path = summary.get("save_file_path", "")
            if route_save_path:
                route_state_diff_dir = Path(route_save_path).with_suffix("")
                route_state_diff_dir = route_state_diff_dir.parent / f"{route_state_diff_dir.name}_state_diffs"
                if route_state_diff_dir.exists():
                    for sd_file in glob.glob(str(route_state_diff_dir / "*.json")):
                        shutil.copy2(sd_file, combined_state_diff_dir)

    print(f"[orchestrate] Merged {merged_count} runs from {len(result_files)} routes")


def _resolve_route_data_file(route: dict) -> str:
    """Return an existing source data file for the route, or '' if none found."""
    candidates = []
    if route.get("data_file"):
        candidates.append(route["data_file"])
    candidates.extend(route.get("fallback_data_files", []) or [])
    for raw in candidates:
        p = Path(raw)
        if not p.is_absolute():
            p = (FRAMEWORK_ROOT / raw).resolve()
        if p.exists() and p.stat().st_size > 0:
            return str(p)
    return ""


def _copy_fresh_sandbox(run_tag: str, route_id: str = None, profile_name: str = None, custom_prefix: str = None) -> str:
    """Copy the fixed sandbox bundle to a fresh per-run dir; return its path.

    If the sandbox already exists (e.g., during resume), return the existing path
    without copying again.
    """
    if not FIXED_SANDBOX_BUNDLE.exists():
        raise FileNotFoundError(
            f"Fixed sandbox bundle not found: {FIXED_SANDBOX_BUNDLE}. "
            "Expected runtime_assets/fs_bundle/fs_fixtures/<env_id>/<task_id>."
        )
    SANDBOX_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    # Use prefix + profile_name + run_tag + route_id to make sandbox unique per concurrent eval
    sandbox_name = "sandbox"
    if custom_prefix:
        sandbox_name += f"_{custom_prefix}"
    if profile_name:
        sandbox_name += f"_{profile_name}"
    sandbox_name += f"_{run_tag}"
    if route_id:
        sandbox_name += f"_{route_id}"
    target = SANDBOX_RUN_ROOT / sandbox_name

    # If sandbox already exists (resume case), return existing path
    if target.exists():
        return str(target.resolve())

    # Create fresh sandbox
    shutil.copytree(FIXED_SANDBOX_BUNDLE, target, symlinks=False)

    return str(target.resolve())


def _profile_args(profile: dict) -> list[str]:
    """Translate a profile dict into run_eval.py CLI flags."""
    mapping = {
        "agent_model": "--agent-model",
        "agent_provider": "--agent-provider",
        "agent_api_key": "--agent-api-key",
        "agent_base_url": "--agent-base-url",
        "agent_reasoning_effort": "--agent-reasoning-effort",
        "agent_effort": "--agent-effort",
        "user_model": "--user-model",
        "user_provider": "--user-provider",
        "user_api_key": "--user-api-key",
        "user_base_url": "--user-base-url",
        "rubric_judge_model": "--rubric-judge-model",
        "rubric_judge_provider": "--rubric-judge-provider",
        "rubric_judge_api_key": "--rubric-judge-api-key",
        "rubric_judge_base_url": "--rubric-judge-base-url",
        "infer_mode": "--infer-mode",
    }
    flags: list[str] = []
    for key, flag in mapping.items():
        value = str(profile.get(key) or "").strip()
        if value:
            flags.extend([flag, value])

    # Handle thinking configurations
    if profile.get("agent_enable_thinking"):
        flags.append("--enable-thinking")

    # Handle user/rubric thinking configurations (passed as extra kwargs to rubric_judge_config)
    user_thinking = profile.get("user_enable_thinking")
    rubric_thinking = profile.get("rubric_judge_enable_thinking")
    if user_thinking is not None or rubric_thinking is not None:
        # Store as JSON extra arg for run_eval.py to parse
        import json
        thinking_config = {}
        if user_thinking is not None:
            thinking_config["user_enable_thinking"] = user_thinking
        if rubric_thinking is not None:
            thinking_config["rubric_judge_enable_thinking"] = rubric_thinking
        flags.extend(["--thinking-config", json.dumps(thinking_config)])

    return flags


def run_single_route(
    route: dict,
    profile: dict,
    profile_name: str,
    pass_k: int,
    max_task_workers: int,
    lang_filter: str,
    extra_args: list[str],
    run_tag: str,
    route_global_id_range: str | None = None,
    custom_prefix: str = "",
) -> dict:
    route_id = route.get("id", "route")
    name = route.get("name", route_id)
    env_name = route["env_name"]

    source_path = _resolve_route_data_file(route)
    if not source_path:
        return {
            "route_id": route_id,
            "name": name,
            "status": "missing_data_file",
            "detail": f"no data file found (data_file={route.get('data_file')})",
        }

    prepared_path = str((PREPARED_DIR / f"{route_id}_prepared.json").resolve())
    id_info = prepare_route_file(
        source_path=source_path,
        output_path=prepared_path,
        id_base=int(route.get("id_base", 1)),
        id_passthrough=bool(route.get("id_passthrough", False)),
    )
    print(f"[orchestrate] {route_id} id-assignment: {json.dumps(id_info, ensure_ascii=False)}", flush=True)

    env = dict(os.environ)
    sandbox_root = ""
    if route.get("needs_sandbox"):
        sandbox_root = _copy_fresh_sandbox(run_tag, route_id, profile_name, custom_prefix)
        env["ENVSCALER_FS_BUNDLE_ROOT"] = sandbox_root
        print(f"[orchestrate] {route_id} fresh sandbox -> {sandbox_root}", flush=True)

    # 统一输出目录结构：<profile>-<run_tag>/，内含 route1/route2/route3 子目录，incremental_shards 和 state_diffs 在根目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_output_dir = RESULTS_DIR / f"{profile_name}-{run_tag}"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    route_output_dir = base_output_dir / route_id
    route_output_dir.mkdir(parents=True, exist_ok=True)

    # 统一的 incremental_shards 目录（所有 route 共享）
    incremental_dir = base_output_dir / "incremental_shards"
    incremental_dir.mkdir(parents=True, exist_ok=True)

    passk_suffix = f"_passk{int(pass_k)}" if int(pass_k) > 1 else ""
    save_file_path = str(
        (route_output_dir / f"{route_id}-{profile_name}-{run_tag}{passk_suffix}.json").resolve()
    )

    cmd = [
        sys.executable,
        str((SCRIPT_DIR / "run_eval.py").resolve()),
        "--execution-mode", "run_and_eval",
        "--env-name", env_name,
        "--task-items-path", prepared_path,
        "--save-file-path", save_file_path,
        "--pass-k", str(int(pass_k)),
        "--max-task-workers", str(int(max_task_workers)),
        "--batch", route_id,
        "--incremental-dir", str(incremental_dir),
    ]
    if lang_filter:
        cmd.extend(["--lang-filter", lang_filter])
    if route.get("needs_sandbox") and sandbox_root:
        cmd.extend(["--fs-bundle-root", sandbox_root])

    # Add per-route global-id-range: explicit arg > route config > none (use run_eval default)
    if route_global_id_range:
        cmd.extend(["--global-id-range", route_global_id_range])
    elif route.get("global_id_range"):
        cmd.extend(["--global-id-range", route["global_id_range"]])

    cmd.extend(_profile_args(profile))
    if extra_args:
        cmd.extend(extra_args)

    # Print expected task info for resume verification
    if "--resume" in (extra_args or []):
        import glob
        import json as json_module
        shard_pattern = str(incremental_dir / f"{route_id}-{profile_name}-{run_tag}*_incremental_shards" / "incremental_shard_*.jsonl")
        shard_files = glob.glob(shard_pattern)
        completed_count = 0
        if shard_files:
            for sf in shard_files:
                try:
                    with open(sf, 'r') as f:
                        completed_count += sum(1 for line in f if line.strip())
                except Exception:
                    pass
        total_tasks = id_info.get("item_count", 0)
        pending_count = total_tasks - completed_count
        print(f"[orchestrate] {route_id} RESUME: {completed_count}/{total_tasks} completed, {pending_count} pending tasks")
        if pending_count > 0:
            print(f"[orchestrate] {route_id} Expected global_id range: {route_global_id_range or route.get('global_id_range', 'N/A')}")

    print(f"[orchestrate] {route_id} running: {' '.join(cmd)}", flush=True)
    started = time.time()
    completed = subprocess.run(cmd, env=env)
    elapsed = round(time.time() - started, 2)

    if completed.returncode != 0:
        return {
            "route_id": route_id,
            "name": name,
            "status": f"run_eval_failed(rc={completed.returncode})",
            "save_file_path": save_file_path,
            "elapsed_seconds": elapsed,
        }
    if not Path(save_file_path).exists():
        return {
            "route_id": route_id,
            "name": name,
            "status": "no_result_file",
            "save_file_path": save_file_path,
            "elapsed_seconds": elapsed,
        }

    summary = summarize_route(save_file_path)
    summary.update(
        {
            "route_id": route_id,
            "name": name,
            "status": "ok",
            "env_name": env_name,
            "source_path": source_path,
            "prepared_path": prepared_path,
            "sandbox_root": sandbox_root,
            "elapsed_seconds": elapsed,
        }
    )
    return summary


def orchestrate(
    routes_config_path: str,
    profiles_config_path: str,
    profile_name: str,
    selected_route_ids: list[str] | None,
    pass_k: int,
    max_task_workers: int,
    lang_filter: str,
    extra_args: list[str],
    data_overrides: dict[str, str] | None = None,
    route_global_id_ranges: dict[str, str] | None = None,
) -> list[dict]:
    routes_config = _load_json(Path(routes_config_path))
    profiles_config = _load_json(Path(profiles_config_path))

    profile_name = profile_name or profiles_config.get("default_profile", "external")
    profiles = profiles_config.get("profiles", {})
    if profile_name not in profiles:
        raise KeyError(f"profile '{profile_name}' not in {profiles_config_path}. Available: {list(profiles)}")
    profile = profiles[profile_name]

    routes = routes_config.get("routes", [])
    if selected_route_ids:
        wanted = set(selected_route_ids)
        routes = [r for r in routes if r.get("id") in wanted]

    # 内网直改路径：--data-override route_id=/abs/path 直接指向已有数据，
    # 覆盖 routes.json 的 data_file，并清掉 fallback 避免回退到旧路径。
    if data_overrides:
        known_ids = {r.get("id") for r in routes_config.get("routes", [])}
        for rid, path in data_overrides.items():
            if rid not in known_ids:
                raise KeyError(f"--data-override 未知 route id '{rid}'. 可用：{sorted(known_ids)}")
        for route in routes:
            rid = route.get("id")
            if rid in data_overrides:
                route["data_file"] = data_overrides[rid]
                route["fallback_data_files"] = []
                print(f"[orchestrate] {rid} data_file 覆盖 -> {data_overrides[rid]}", flush=True)

    # 使用北京时间
    if _is_resume_requested(extra_args):
        run_tag = _find_latest_resume_run_tag(
            profile_name=profile_name,
            routes=routes,
            pass_k=pass_k,
        ) or _get_beijing_timestamp()
    else:
        run_tag = _get_beijing_timestamp()
    # 从 profile 中获取 user_model 用于文件名
    user_model = profile.get("user_model", "unknown")
    # 支持自定义 prefix (可通过环境变量 ORCHESTRATE_PREFIX 设置)
    custom_prefix = os.environ.get("ORCHESTRATE_PREFIX", "")

    # Print resume summary at start
    if _is_resume_requested(extra_args):
        print(f"[orchestrate] RESUME MODE: will check existing results and run pending tasks only", flush=True)
        for route in routes:
            rid = route.get("id", "route")
            expected_range = route_global_id_ranges.get(rid) if route_global_id_ranges else None
            if not expected_range:
                expected_range = route.get("global_id_range", "N/A")
            print(f"[orchestrate]   {rid}: expected global_id range = {expected_range}", flush=True)
    else:
        print(f"[orchestrate] FRESH RUN: will run all tasks from scratch", flush=True)

    print(f"[orchestrate] profile={profile_name} run_tag={run_tag} user_model={user_model} routes={[r.get('id') for r in routes]}", flush=True)

    # 统一输出目录结构：<profile>-<run_tag>/，内含 route1/route2/route3 子目录，incremental_shards 和 state_diffs 在根目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dirname = f"{profile_name}-{run_tag}"
    if custom_prefix:
        dirname = f"{custom_prefix}{dirname}"
    base_output_dir = RESULTS_DIR / dirname
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for route in routes:
        route_id = route.get("id")
        route_range = route_global_id_ranges.get(route_id) if route_global_id_ranges else None

        summary = run_single_route(
            route=route,
            profile=profile,
            profile_name=profile_name,
            pass_k=pass_k,
            max_task_workers=max_task_workers,
            lang_filter=lang_filter,
            extra_args=extra_args,
            run_tag=run_tag,
            route_global_id_range=route_range,
            custom_prefix=custom_prefix,
        )
        summaries.append(summary)
        print(f"[orchestrate] {route.get('id')} -> {json.dumps(summary, ensure_ascii=False)}", flush=True)

    print_route_table(summaries)

    # Generate comprehensive report with per-route and overall statistics
    try:
        from generate_eval_report import generate_combined_report, print_report

        # Build result_files dict from summaries
        result_files = {}
        for summary in summaries:
            if summary.get("status") == "ok" and summary.get("save_file_path"):
                route_id = summary.get("route_id", "unknown")
                result_files[route_id] = summary["save_file_path"]

        if result_files:
            print()
            print("=" * 80)
            print("COMPREHENSIVE EVALUATION REPORT")
            print("=" * 80)
            print()

            # Generate and print combined report
            report = generate_combined_report(result_files, output_path=None)
            print_report(report)

            # Merge all route .runs.jsonl files into one combined file (in base_output_dir)
            merged_runs_path = base_output_dir / f"combined-{custom_prefix}{profile_name}-{user_model}-{run_tag}.runs.jsonl"
            _merge_all_route_runs_jsonl(result_files, summaries, merged_runs_path)
            print(f"[orchestrate] Merged runs.jsonl -> {merged_runs_path}")

            # Generate Route1 capability/domain analysis report
            try:
                from analyze_eval import load_capability_map, analyze_route1, export_route1_analysis

                print()
                print("Generating Route1 capability analysis...")
                cap_map = load_capability_map()

                # Load route1 results for analysis
                route1_results = []
                if "route1" in result_files:
                    route1_path = Path(result_files["route1"])
                    if route1_path.exists():
                        with open(route1_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        item = json.loads(line)
                                        # Filter route1 data (global_id < 1000000)
                                        r = item.get("result", item)
                                        task_info = r.get("task_info") if isinstance(r.get("task_info"), dict) else {}
                                        gid = task_info.get("global_id") or r.get("global_id", 0)
                                        try:
                                            gid = int(gid)
                                        except (TypeError, ValueError):
                                            gid = 0
                                        if gid < 1000000:
                                            route1_results.append(item)
                                        elif gid >= 1000000:
                                            break
                                    except json.JSONDecodeError:
                                        continue

                if route1_results:
                    print(f"Route1 results loaded: {len(route1_results)} tasks")
                    stats = analyze_route1(route1_results, cap_map)

                    # Export to JSON in the same output directory
                    route1_analysis_path = base_output_dir / f"route1-analysis-{custom_prefix}{profile_name}-{user_model}-{run_tag}.json"
                    export_route1_analysis(stats, str(route1_analysis_path))
                    print(f"Route1 analysis saved to: {route1_analysis_path}")

                    # Also print summary to console
                    from analyze_eval import print_route1_analysis
                    print_route1_analysis(stats)
            except Exception as e:
                print(f"Warning: Failed to generate Route1 analysis: {e}")
    except Exception as e:
        print(f"[orchestrate] Warning: Failed to generate comprehensive report: {e}")

    summary_path = RESULTS_DIR / f"route_summary-{profile_name}-{run_tag}.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            {"profile": profile_name, "run_tag": run_tag, "routes": summaries},
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[orchestrate] summary -> {summary_path}", flush=True)

    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orchestrate multi-route eval.")
    parser.add_argument("--routes-config", default=str(CONFIGS_DIR / "routes.json"))
    parser.add_argument("--profiles-config", default=str(CONFIGS_DIR / "profiles.json"))
    parser.add_argument("--profile", default="", help="Profile name; defaults to profiles.json default_profile.")
    parser.add_argument("--routes", "--selected-routes", nargs="*", default=None, help="Subset of route ids, e.g. route1 route3.")
    parser.add_argument("--pass-k", type=int, default=1)
    parser.add_argument("--max-task-workers", type=int, default=8)
    parser.add_argument("--lang-filter", default="", help="Optional cn/en lang filter passed through to run_eval.")
    parser.add_argument(
        "--route-global-id-range",
        action="append",
        default=[],
        metavar="route_id=RANGE",
        help="Per-route global_id range, e.g. route2=1-10 route3=all. Can be repeated.",
    )
    parser.add_argument(
        "--data-override",
        action="append",
        default=[],
        metavar="route_id=PATH",
        help="直接指定某路数据文件，覆盖 routes.json。可重复，如 --data-override route2=/data/solver.json",
    )
    parser.add_argument(
        "--",
        dest="passthrough",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded verbatim to run_eval.py.",
    )
    return parser


def _is_resume_requested(extra_args: list[str]) -> bool:
    """判断透传参数里是否包含 --resume。"""
    return "--resume" in set(str(x).strip() for x in (extra_args or []))


def _find_latest_resume_run_tag(
    profile_name: str,
    routes: list[dict],
    pass_k: int,
) -> str:
    """在 results/<profile>-*/ 里找最近一次可 resume 的 run_tag。

    只找当前 profile 对应的目录。
    只要某个 route 下存在对应的 .json 或 .runs.jsonl，就认为可 resume。
    """
    passk_suffix = f"_passk{int(pass_k)}" if int(pass_k) > 1 else ""
    candidates = []

    if not RESULTS_DIR.exists():
        return ""

    prefix = f"{profile_name}-"

    for base_dir in RESULTS_DIR.glob(f"{profile_name}-*"):
        if not base_dir.is_dir():
            continue

        dirname = base_dir.name
        if not dirname.startswith(prefix):
            continue

        run_tag = dirname[len(prefix):]
        if not run_tag:
            continue

        matched = False

        for route in routes:
            route_id = str(route.get("id", "route"))
            route_dir = base_dir / route_id
            expected_json = route_dir / f"{route_id}-{profile_name}-{run_tag}{passk_suffix}.json"
            expected_jsonl = expected_json.with_suffix(".runs.jsonl")

            # 兼容增量分片目录
            expected_shard_dir = base_dir / "incremental_shards" / f"{expected_json.stem}_incremental_shards"

            if expected_json.exists() or expected_jsonl.exists() or expected_shard_dir.exists():
                matched = True
                break

        if matched:
            candidates.append((base_dir.stat().st_mtime, run_tag, base_dir))

    if not candidates:
        return ""

    candidates.sort(reverse=True)
    latest_mtime, latest_run_tag, latest_base_dir = candidates[0]
    print(f"[orchestrate] resume detected, reuse latest run_tag={latest_run_tag} from {latest_base_dir}", flush=True)
    return latest_run_tag

def main():
    args, unknown = build_parser().parse_known_args()
    extra_args = list(getattr(args, "passthrough", []) or []) + list(unknown or [])

    # Parse per-route global-id-range
    route_global_id_ranges: dict[str, str] = {}
    for raw in getattr(args, "route_global_id_range", []) or []:
        if "=" not in raw:
            raise ValueError(f"--route-global-id-range 格式应为 route_id=RANGE，收到：{raw!r}")
        rid, range_val = raw.split("=", 1)
        rid, range_val = rid.strip(), range_val.strip()
        if not rid or not range_val:
            raise ValueError(f"--route-global-id-range 格式应为 route_id=RANGE，收到：{raw!r}")
        route_global_id_ranges[rid] = range_val

    data_overrides: dict[str, str] = {}
    for raw in getattr(args, "data_override", []) or []:
        if "=" not in raw:
            raise ValueError(f"--data-override 格式应为 route_id=PATH，收到：{raw!r}")
        rid, path = raw.split("=", 1)
        rid, path = rid.strip(), path.strip()
        if not rid or not path:
            raise ValueError(f"--data-override 格式应为 route_id=PATH，收到：{raw!r}")
        data_overrides[rid] = path

    orchestrate(
        routes_config_path=args.routes_config,
        profiles_config_path=args.profiles_config,
        profile_name=args.profile,
        selected_route_ids=args.routes,
        pass_k=int(args.pass_k),
        max_task_workers=int(args.max_task_workers),
        lang_filter=args.lang_filter,
        extra_args=extra_args,
        data_overrides=data_overrides,
        route_global_id_ranges=route_global_id_ranges,
    )


if __name__ == "__main__":
    main()
