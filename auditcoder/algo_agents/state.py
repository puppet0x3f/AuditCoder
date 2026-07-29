"""LangGraph state definition shared by all algorithm agents."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AlgoAgentState(TypedDict, total=False):
    """State passed into and out of each algorithm agent graph.

    task_node     — the single leaf TaskNode dict this agent must implement.
    task_plan_ir  — the full TaskPlanIR dict for context (problem, strategy, …).
    code_snippet  — Python source code produced by the agent.
    feedback      — optional verifier feedback dict injected on retry; contains
                    verdict, counterexample, contract_violations, error details.
    messages      — LangChain message history (auto-appended via add_messages).
    """

    task_node: dict[str, Any]
    task_plan_ir: dict[str, Any]
    code_snippet: str
    feedback: dict[str, Any] | None
    messages: Annotated[list, add_messages]
