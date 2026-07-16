"""
adapters：集中维护构造流程里的系统级增强能力。

当前包含三类可选策略：
- search：查询结果扩充与 search-like 工具标注
- fs：安全文件系统工具与文件沙箱
- code：Python 执行器工具
"""

from .strategy_config import (
    normalize_strategy_config,
    is_search_strategy_enabled,
    is_fs_strategy_enabled,
    is_code_strategy_enabled,
)

__all__ = [
    "normalize_strategy_config",
    "is_search_strategy_enabled",
    "is_fs_strategy_enabled",
    "is_code_strategy_enabled",
]
