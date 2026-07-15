#!/usr/bin/env python3
"""
只重新评估 rubric，不重新运行任务
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, 'runtime')
from result_evaluator import load_task_lookup, augment_result_with_evaluations


def main():
    if len(sys.argv) < 2:
        print("用法: python reeval_rubric_only.py <result_dir>")
        sys.exit(1)

    result_dir = Path(sys.argv[1])

    # 加载 task_lookup
    print("加载 task lookup...")
    task_lookup = load_task_lookup('data/routes/route1.json')
    print(f"✓ 加载了 {len(task_lookup)} 个任务")

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
    print(f"\n找到 {len(shard_files)} 个 shard 文件")
    print("开始重新评估 rubric（不重新运行任务）...\n")

    total_tasks = 0
    total_fixed = 0
    total_score_improved = 0

    for i, shard_file in enumerate(shard_files, 1):
        try:
            results = []
            fixed_count = 0
            improved_count = 0

            with open(shard_file) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        result = data.get('result', {})

                        old_score = result.get('rubric_reward', 0)
                        old_summary = result.get('rubric_eval', {}).get('summary', '')

                        # 重新评估（只评估 rubric，不重新运行任务）
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

                        if '不匹配' in old_summary and '不匹配' not in new_summary:
                            fixed_count += 1

                        if new_score > old_score:
                            improved_count += 1

                        data['result'] = updated_result
                        results.append(data)

            # 写回文件
            with open(shard_file, 'w') as f:
                for data in results:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')

            total_tasks += len(results)
            total_fixed += fixed_count
            total_score_improved += improved_count

            print(f"[{i}/{len(shard_files)}] ✓ {shard_file.name}: {len(results)} 个任务, 修复不匹配 {fixed_count}, 分数提升 {improved_count}")

        except Exception as e:
            print(f"[{i}/{len(shard_files)}] ✗ {shard_file.name}: {e}")

    print(f"\n总计:")
    print(f"  重新评估: {total_tasks} 个任务")
    print(f"  修复不匹配: {total_fixed} 个任务")
    print(f"  分数提升: {total_score_improved} 个任务")


if __name__ == '__main__':
    main()
