"""兼容入口：Qwen APPS 错误子树重规划实验。

实际实现位于 ``tests/apps_qwen_subtree_retry.py``。后者会显式把
AuditCoder 各模块中的 LLM 工厂切换到 Qwen，避免仅修改环境变量却仍
调用 DeepSeek 客户端。
"""

from __future__ import annotations

from apps_qwen_subtree_retry import main


if __name__ == "__main__":
    main()
