#!/usr/bin/env python3
"""
绘制OmniaBench评测结果的可视化图表：
- 路径对比柱状图
- ToB/ToC/ToE对比图
- Level-1 domain热力图
- Capability雷达图

用法:
    python plot_results.py --scores scores.json --output figures/
    python plot_results.py --result-dir results/your_model_run/ --output figures/
"""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_route_comparison(scores: Dict, output_dir: Path):
    """绘制路径对比柱状图"""
    route_stats = scores['by_route']
    if not route_stats:
        return

    routes = sorted(route_stats.keys())
    pass_rates = [route_stats[r]['pass_rate'] for r in routes]
    avg_scores = [route_stats[r]['avg_score'] for r in routes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pass@1
    ax1.bar(routes, pass_rates, color='steelblue', alpha=0.8)
    ax1.set_ylabel('Pass@1 (%)', fontsize=12)
    ax1.set_title('Pass@1 by Route', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    for i, v in enumerate(pass_rates):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)

    # Avg Score
    ax2.bar(routes, avg_scores, color='coral', alpha=0.8)
    ax2.set_ylabel('Average Score', fontsize=12)
    ax2.set_title('Average Score by Route', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    for i, v in enumerate(avg_scores):
        ax2.text(i, v + 2, f'{v:.1f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'route_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'route_comparison.png'}")


def plot_tob_toc_toe(scores: Dict, output_dir: Path):
    """绘制ToB/ToC/ToE对比图"""
    category_stats = scores['by_tob_toc_toe']
    if not category_stats:
        return

    # 过滤unknown
    valid_categories = {k: v for k, v in category_stats.items() if k != 'unknown'}
    if not valid_categories:
        print("⚠ No ToB/ToC/ToE category labels found, skipping tob_toc_toe plot (see evaluation/README.md#taxonomy-reference)")
        return

    categories = sorted(valid_categories.keys())
    pass_rates = [valid_categories[c]['pass_rate'] for c in categories]
    counts = [valid_categories[c]['count'] for c in categories]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pass@1对比
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax1.bar(categories, pass_rates, color=colors[:len(categories)], alpha=0.8)
    ax1.set_ylabel('Pass@1 (%)', fontsize=12)
    ax1.set_title('Pass@1 by Category (ToB/ToC/ToE)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    for i, v in enumerate(pass_rates):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)

    # 任务数量分布
    ax2.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors[:len(categories)],
            startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Task Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'tob_toc_toe.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'tob_toc_toe.png'}")


def plot_domain_heatmap(scores: Dict, output_dir: Path, top_n: int = 30):
    """绘制Level-1 domain热力图（top N）"""
    domain_stats = scores['by_level1_domain']
    if not domain_stats:
        return

    # 过滤unknown并排序
    valid_domains = {k: v for k, v in domain_stats.items() if k != 'unknown'}
    if not valid_domains:
        print("⚠ No level-1 domain labels found, skipping domain_heatmap plot (see evaluation/README.md#taxonomy-reference)")
        return

    # 取top N
    sorted_domains = sorted(valid_domains.items(), key=lambda x: x[1]['pass_rate'], reverse=True)[:top_n]
    domains = [d[0] for d in sorted_domains]
    pass_rates = [d[1]['pass_rate'] for d in sorted_domains]

    fig, ax = plt.subplots(figsize=(10, max(8, len(domains) * 0.3)))

    # 水平条形图
    y_pos = np.arange(len(domains))
    colors = plt.cm.RdYlGn(np.array(pass_rates) / 100)

    ax.barh(y_pos, pass_rates, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(domains, fontsize=9)
    ax.set_xlabel('Pass@1 (%)', fontsize=12)
    ax.set_title(f'Top {len(domains)} Level-1 Domains by Pass@1', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)

    # 添加数值标签
    for i, v in enumerate(pass_rates):
        ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'domain_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'domain_heatmap.png'}")


def plot_capability_radar(scores: Dict, output_dir: Path):
    """绘制Capability雷达图"""
    cap_stats = scores['by_capability']
    if not cap_stats:
        print("⚠ No capability data found, skipping radar plot")
        return

    # 准备数据
    capabilities = list(cap_stats.keys())
    values = [cap_stats[c] for c in capabilities]

    # 雷达图需要闭合
    capabilities_closed = capabilities + [capabilities[0]]
    values_closed = values + [values[0]]

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(capabilities), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    # 绘制
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    ax.plot(angles_closed, values_closed, 'o-', linewidth=2, color='steelblue')
    ax.fill(angles_closed, values_closed, alpha=0.25, color='steelblue')

    ax.set_xticks(angles)
    # 简化capability名称
    short_names = [c.replace('_', ' ').title()[:20] for c in capabilities]
    ax.set_xticklabels(short_names, fontsize=9)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
    ax.set_title('Capability Scores', fontsize=14, fontweight='bold', pad=20)

    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'capability_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'capability_radar.png'}")


def main():
    parser = argparse.ArgumentParser(description='Plot OmniaBench evaluation results')
    parser.add_argument('--scores', type=str, help='Path to scores JSON file (from compute_scores.py)')
    parser.add_argument('--result-dir', type=str, help='Directory containing route files (will compute scores first)')
    parser.add_argument('--output', type=str, default='figures/', help='Output directory for figures')
    parser.add_argument('--top-domains', type=int, default=30, help='Number of top domains to show in heatmap')

    args = parser.parse_args()

    # 加载或计算scores
    if args.scores:
        with open(args.scores, 'r', encoding='utf-8') as f:
            scores = json.load(f)
    elif args.result_dir:
        # 调用compute_scores模块
        from compute_scores import compute_all_scores
        result_dir = Path(args.result_dir)
        route_files = {}
        for i in range(1, 5):
            route_file = result_dir / f'route{i}.json'
            if route_file.exists():
                route_files[f'route{i}'] = str(route_file)
        if not route_files:
            print("Error: No route files found in result-dir")
            return
        scores = compute_all_scores(route_files)
    else:
        print("Error: Provide --scores or --result-dir")
        return

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 绘制各类图表
    print("\nGenerating plots...")
    plot_route_comparison(scores, output_dir)
    plot_tob_toc_toe(scores, output_dir)
    plot_domain_heatmap(scores, output_dir, top_n=args.top_domains)
    plot_capability_radar(scores, output_dir)

    print(f"\n✓ All plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
