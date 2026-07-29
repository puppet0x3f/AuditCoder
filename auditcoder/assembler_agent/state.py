"""LangGraph state definition for the Assembler Agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AssemblerState(TypedDict, total=False):
    """State passed into and out of the assembler agent graph.

    task_plan_ir  — the full TaskPlanIR dict from plan_agent (problem, strategy,
                    task_nodes, execution_order, etc.).
    code_snippets — dict mapping task_id → code_snippet string, produced by
                    dispatcher after running all leaf algorithm agents.
    final_code    — the complete, runnable Python solution assembled by the LLM.
    messages      — LangChain message history (auto-appended via add_messages).
    """

    task_plan_ir: dict[str, Any]
    code_snippets: dict[str, str]
    final_code: str
    assembly_contract_violations: list[str]
    assembly_contract_repair_attempted: bool
    messages: Annotated[list, add_messages]
