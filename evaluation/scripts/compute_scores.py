#!/usr/bin/env python3
"""
计算OmniaBench评测结果的各维度分数：
- 路径级别 (route1/2/3/4)
- 一级domain级别 (90个level-1 domains)
- ToB/ToC/ToE级别
- 能力维度 (10 capabilities)

用法:
    python compute_scores.py --result-dir results/your_model_run/
    python compute_scores.py --route1 results/model/route1.json --route2 results/model/route2.json ...
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


# 10维能力分类
CAPABILITIES = [
    "task_understanding",
    "information_gathering",
    "planning_decision_making",
    "state_management",
    "tool_use",
    "code_programmatic_operations",
    "data_analysis",
    "office_document_handling",
    "interactive_collaboration",
    "reliability_safety"
]


def load_route_data(path: str) -> List[Dict[str, Any]]:
    """加载单个路径的JSON结果"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_pass_rate(items: List[Dict[str, Any]]) -> float:
    """计算pass@1 (score >= 0.6视为pass)"""
    if not items:
        return 0.0
    passed = sum(1 for item in items if item.get('score', 0) >= 0.6)
    return passed / len(items) * 100


def compute_avg_score(items: List[Dict[str, Any]]) -> float:
    """计算平均score"""
    if not items:
        return 0.0
    return sum(item.get('score', 0) for item in items) / len(items) * 100


def extract_level1_domain(task_info: Dict) -> str:
    """从task_info中提取level-1 domain"""
    # 优先从domain字段提取
    domain = task_info.get('domain', '')
    if domain:
        # 格式: "level1_domain > level2_domain" 或 "level1_domain"
        return domain.split('>')[0].strip()

    # 回退：从env_name提取
    env_name = task_info.get('env_name', '')
    # 格式: env_{id}_{domain}
    if env_name:
        parts = env_name.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[2:])

    return 'unknown'


def extract_tob_toc_toe(task_info: Dict) -> str:
    """提取ToB/ToC/ToE分类"""
    category = task_info.get('category', '').upper()
    if category in ['TOB', 'TOC', 'TOE']:
        return category
    # 回退方案：从domain推断（根据实际数据调整）
    return 'unknown'


def extract_capabilities(result: Dict) -> Dict[str, float]:
    """提取capability维度得分（如果存在）"""
    cap_scores = {}
    task_info = result.get('task_info', {})

    # 尝试从多个可能字段提取
    for cap in CAPABILITIES:
        score = task_info.get(f'{cap}_score') or task_info.get(f'capability_{cap}')
        if score is not None:
            cap_scores[cap] = float(score)

    return cap_scores


def aggregate_by_level1_domain(results: List[Dict]) -> Dict[str, Dict]:
    """按level-1 domain聚合"""
    domain_items = defaultdict(list)

    for item in results:
        task_info = item.get('task_info', {})
        domain = extract_level1_domain(task_info)
        domain_items[domain].append(item)

    domain_stats = {}
    for domain, items in domain_items.items():
        domain_stats[domain] = {
            'count': len(items),
            'pass_rate': compute_pass_rate(items),
            'avg_score': compute_avg_score(items)
        }

    return domain_stats


def aggregate_by_tob_toc_toe(results: List[Dict]) -> Dict[str, Dict]:
    """按ToB/ToC/ToE聚合"""
    category_items = defaultdict(list)

    for item in results:
        task_info = item.get('task_info', {})
        category = extract_tob_toc_toe(task_info)
        category_items[category].append(item)

    category_stats = {}
    for category, items in category_items.items():
        category_stats[category] = {
            'count': len(items),
            'pass_rate': compute_pass_rate(items),
            'avg_score': compute_avg_score(items)
        }

    return category_stats


def aggregate_by_capability(results: List[Dict]) -> Dict[str, float]:
    """按capability维度聚合（如果数据中有capability标注）"""
    cap_scores = defaultdict(list)

    for item in results:
        caps = extract_capabilities(item)
        for cap, score in caps.items():
            cap_scores[cap].append(score)

    cap_stats = {}
    for cap, scores in cap_scores.items():
        if scores:
            cap_stats[cap] = sum(scores) / len(scores) * 100

    return cap_stats


def compute_all_scores(route_files: Dict[str, str]) -> Dict:
    """计算所有维度的分数"""
    all_results = []
    route_stats = {}

    # 加载各路径数据
    for route_id, path in route_files.items():
        if not path or not Path(path).exists():
            continue

        items = load_route_data(path)
        all_results.extend(items)

        route_stats[route_id] = {
            'count': len(items),
            'pass_rate': compute_pass_rate(items),
            'avg_score': compute_avg_score(items)
        }

    # 整体统计
    overall_stats = {
        'count': len(all_results),
        'pass_rate': compute_pass_rate(all_results),
        'avg_score': compute_avg_score(all_results)
    }

    # 各维度聚合
    domain_stats = aggregate_by_level1_domain(all_results)
    tob_toc_toe_stats = aggregate_by_tob_toc_toe(all_results)
    capability_stats = aggregate_by_capability(all_results)

    return {
        'overall': overall_stats,
        'by_route': route_stats,
        'by_level1_domain': domain_stats,
        'by_tob_toc_toe': tob_toc_toe_stats,
        'by_capability': capability_stats
    }


def print_report(scores: Dict):
    """打印报告"""
    print("\n" + "="*80)
    print("OmniaBench Evaluation Report")
    print("="*80)

    # Overall
    print("\n[Overall]")
    overall = scores['overall']
    print(f"  Tasks: {overall['count']}")
    print(f"  Pass@1: {overall['pass_rate']:.2f}%")
    print(f"  Avg Score: {overall['avg_score']:.2f}")

    # By route
    print("\n[By Route]")
    for route_id, stats in sorted(scores['by_route'].items()):
        print(f"  {route_id}: {stats['count']} tasks, Pass@1={stats['pass_rate']:.2f}%, Score={stats['avg_score']:.2f}")

    # By ToB/ToC/ToE
    print("\n[By ToB/ToC/ToE]")
    for category, stats in sorted(scores['by_tob_toc_toe'].items()):
        if category != 'unknown':
            print(f"  {category}: {stats['count']} tasks, Pass@1={stats['pass_rate']:.2f}%, Score={stats['avg_score']:.2f}")

    # By Level-1 Domain (top 10)
    print("\n[Top 10 Level-1 Domains by Pass Rate]")
    domains = scores['by_level1_domain']
    top10 = sorted(domains.items(), key=lambda x: x[1]['pass_rate'], reverse=True)[:10]
    for domain, stats in top10:
        if domain != 'unknown':
            print(f"  {domain}: {stats['count']} tasks, Pass@1={stats['pass_rate']:.2f}%, Score={stats['avg_score']:.2f}")

    # By Capability
    if scores['by_capability']:
        print("\n[By Capability]")
        for cap, score in sorted(scores['by_capability'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cap}: {score:.2f}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Compute OmniaBench scores across multiple dimensions')
    parser.add_argument('--result-dir', type=str, help='Directory containing route1.json, route2.json, etc.')
    parser.add_argument('--route1', type=str, help='Path to route1.json')
    parser.add_argument('--route2', type=str, help='Path to route2.json')
    parser.add_argument('--route3', type=str, help='Path to route3.json')
    parser.add_argument('--route4', type=str, help='Path to route4.json')
    parser.add_argument('--output', type=str, help='Save JSON report to this file')

    args = parser.parse_args()

    # 收集路径文件
    route_files = {}
    if args.result_dir:
        result_dir = Path(args.result_dir)
        for i in range(1, 5):
            route_file = result_dir / f'route{i}.json'
            if route_file.exists():
                route_files[f'route{i}'] = str(route_file)
    else:
        for i in range(1, 5):
            route_arg = getattr(args, f'route{i}')
            if route_arg:
                route_files[f'route{i}'] = route_arg

    if not route_files:
        print("Error: No route files found. Use --result-dir or --route1/2/3/4")
        return

    # 计算分数
    scores = compute_all_scores(route_files)

    # 打印报告
    print_report(scores)

    # 保存JSON
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {args.output}")


if __name__ == '__main__':
    main()
