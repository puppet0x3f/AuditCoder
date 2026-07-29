"""APPS 数据集批量测试脚本（Qwen：错误子树重规划消融实验）

这个脚本只负责把当前进程的 LLM provider 切换到 Qwen，然后复用
run_apps_subtree_replan.py 的完整测试逻辑。适合和原 GLM 脚本在不同终端
同时运行，输出目录默认分开。

Usage:
  python tests/run_apps_subtree_replan_qwen.py --offset 0 --count 10
  python tests/run_apps_subtree_replan_qwen.py --offset 10 --count 10 --output result/apps_ablation_subtree_replan_QWEN
  python tests/run_apps_subtree_replan_qwen.py --ids 0,5,10
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

os.environ["AUDITCODER_LLM_PROVIDER"] = "qwen"

import tests.apps_deepseek_subtree_retry as subtree_replan  # noqa: E402


DEFAULT_QWEN_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "result" / "apps_ablation_subtree_replan_QWEN"
)


def _get_qwen_max_tokens(override: int | None = None) -> int | None:
    if override is not None:
        return override
    raw = os.getenv("QWEN_MAX_TOKENS") or os.getenv("MAX_TOKENS")
    if not raw:
        return None
    try:
        return int(str(raw).strip().strip('"').strip("'"))
    except ValueError:
        return None


def _build_qwen_llm(
    temperature: float = 0.0,
    json_mode: bool = True,
    max_tokens: int | None = None,
):
    from langchain_openai import ChatOpenAI

    model_kwargs: dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    return ChatOpenAI(
        model=os.getenv("QWEN_MODEL", "qwen-plus"),
        base_url=os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        api_key=os.getenv("QWEN_API_KEY"),
        temperature=temperature,
        max_tokens=_get_qwen_max_tokens(max_tokens),
        model_kwargs=model_kwargs,
        extra_body={"enable_thinking": False},
    )


def _patch_auditcoder_llm_to_qwen() -> None:
    import auditcoder.common.llm as common_llm

    os.environ["AUDITCODER_LLM_PROVIDER"] = "qwen"
    common_llm.get_llm = _build_qwen_llm

    patched_modules = [
        "auditcoder.common",
        "auditcoder.plan_agent.llm",
        "auditcoder.plan_agent.nodes.algorithm_selector",
        "auditcoder.plan_agent.nodes.constraint_analyzer",
        "auditcoder.plan_agent.nodes.deep_thinker",
        "auditcoder.plan_agent.nodes.interface_designer",
        "auditcoder.plan_agent.nodes.ir_reviewer",
        "auditcoder.plan_agent.nodes.problem_parser",
        "auditcoder.plan_agent.nodes.task_decomposer",
        "auditcoder.plan_agent.nodes.test_case_analyzer",
        "auditcoder.assembler_agent.agent",
        "auditcoder.final_verifier_agent.nodes.task_replanner",
        "auditcoder.verifier_agent.nodes.bruteforce_verifier",
        "auditcoder.verifier_agent.nodes.weak_signal_eval",
    ]
    patched_modules.extend(
        f"auditcoder.algo_agents.agents.{name}_agent"
        for name in [
            "binary_search",
            "dp",
            "graph_bfs",
            "graph_dfs",
            "greedy",
            "math",
            "monotonic_queue",
            "segment_tree",
            "sorting",
            "string",
            "two_pointer",
            "union_find",
        ]
    )

    for module_name in patched_modules:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "get_llm"):
            setattr(module, "get_llm", _build_qwen_llm)


def main() -> None:
    subtree_replan.DEFAULT_OUTPUT_DIR = DEFAULT_QWEN_OUTPUT_DIR
    subtree_replan.LLM_PROVIDER_LABEL = "Qwen"
    subtree_replan.LLM_MODEL_ENV = "QWEN_MODEL"
    subtree_replan.LLM_DEFAULT_MODEL = "qwen-plus"
    subtree_replan._patch_auditcoder_llm_to_deepseek = _patch_auditcoder_llm_to_qwen
    print(f"[Qwen 测试] 模型：{os.getenv('QWEN_MODEL', 'qwen-plus')}")
    subtree_replan.main()


if __name__ == "__main__":
    main()
