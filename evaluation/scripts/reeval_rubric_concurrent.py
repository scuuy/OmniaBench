#!/usr/bin/env python3
"""
并发重新评估 rubric
"""
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, 'runtime')
from result_evaluator import load_task_lookup, augment_result_with_evaluations


# 全局锁用于打印
print_lock = Lock()


def reeval_single_task(data, task_lookup, rubric_judge_config):
    """重新评估单个任务"""
    try:
        result = data.get('result', {})
        old_score = result.get('rubric_reward', 0)
        old_summary = result.get('rubric_eval', {}).get('summary', '')

        # 重新评估
        updated_result = augment_result_with_evaluations(
            result_item=result,
            task_lookup=task_lookup,
            enable_rubric_eval=True,
            force_recompute_rubric_eval=True,
            rubric_judge_config=rubric_judge_config,
            lang='cn',
            enable_checklist_eval=False,
            enable_verifier_eval=False
        )

        new_score = updated_result.get('rubric_reward', 0)
        new_summary = updated_result.get('rubric_eval', {}).get('summary', '')

        data['result'] = updated_result

        fixed = '不匹配' in old_summary and '不匹配' not in new_summary
        improved = new_score > old_score

        return data, fixed, improved, None
    except Exception as e:
        return data, False, False, str(e)


def main():
    if len(sys.argv) < 2:
        print("用法: python reeval_rubric_concurrent.py <result_dir> [max_workers]")
        sys.exit(1)

    result_dir = Path(sys.argv[1])
    max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print(f"并发数: {max_workers}")

    # 加载 task_lookup
    print("加载 task lookup...")
    task_lookup = load_task_lookup('data/routes/route1.json')
    print(f"✓ 加载了 {len(task_lookup)} 个任务\n")

    # Rubric judge 配置
    rubric_judge_config = {
        "model": os.environ.get("OMNIABENCH_JUDGE_MODEL", "YOUR_JUDGE_MODEL"),
        "provider": "openai",
        "api_key": os.environ.get("OMNIABENCH_JUDGE_API_KEY", ""),
        "base_url": os.environ.get("OMNIABENCH_JUDGE_BASE_URL", ""),
        "enable_thinking": False
    }

    # 找到所有 shard 文件
    shard_dir = result_dir / 'incremental_shards' / f'route1-{result_dir.name}_incremental_shards'

    if not shard_dir.exists():
        print(f"✗ 目录不存在: {shard_dir}")
        sys.exit(1)

    shard_files = sorted(list(shard_dir.glob('*.jsonl')))
    print(f"找到 {len(shard_files)} 个 shard 文件\n")

    total_tasks = 0
    total_fixed = 0
    total_improved = 0
    total_errors = 0

    # 逐个shard处理
    for shard_idx, shard_file in enumerate(shard_files, 1):
        print(f"[{shard_idx}/{len(shard_files)}] 处理 {shard_file.name}...")

        # 读取所有任务
        tasks = []
        with open(shard_file) as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))

        # 并发重新评估
        results = []
        fixed_count = 0
        improved_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(reeval_single_task, task, task_lookup, rubric_judge_config): idx
                for idx, task in enumerate(tasks)
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                data, fixed, improved, error = future.result()
                results.append((futures[future], data))

                if fixed:
                    fixed_count += 1
                if improved:
                    improved_count += 1
                if error:
                    error_count += 1
                    with print_lock:
                        print(f"  [错误] 任务 {futures[future]}: {error}")

                if completed % 5 == 0 or completed == len(tasks):
                    with print_lock:
                        print(f"  进度: {completed}/{len(tasks)}")

        # 按原顺序排序并写回
        results.sort(key=lambda x: x[0])
        with open(shard_file, 'w') as f:
            for _, data in results:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        total_tasks += len(tasks)
        total_fixed += fixed_count
        total_improved += improved_count
        total_errors += error_count

        print(f"  ✓ 完成: {len(tasks)} 个任务, 修复不匹配 {fixed_count}, 分数提升 {improved_count}, 错误 {error_count}\n")

    print("=" * 60)
    print(f"总计: {total_tasks} 个任务")
    print(f"修复不匹配: {total_fixed}")
    print(f"分数提升: {total_improved}")
    print(f"错误: {total_errors}")
    print("=" * 60)


if __name__ == '__main__':
    main()
