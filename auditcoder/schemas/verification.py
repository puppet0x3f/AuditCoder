"""Pydantic models for the Verifier Agent's input/output schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Input
# ──────────────────────────────────────────────

class CodeSnippet(BaseModel):
    task_id: str = Field(description="对应 AlgorithmTask.task_id")
    agent_type: str = Field(description="算法 Agent 类型，如 dp, greedy")
    source_code: str = Field(description="Python 源代码")
    explanation: str = Field(default="", description="Agent 自己的解释")


# ──────────────────────────────────────────────
# Signal sub-models
# ──────────────────────────────────────────────

class ResourceUsage(BaseModel):
    max_time_ms: float = Field(default=0.0)
    max_memory_kb: float = Field(default=0.0)


class StrongSignals(BaseModel):
    syntax_ok: bool = Field(default=False)
    compile_ok: bool = Field(default=False)
    sample_tests_passed: bool = Field(default=False)
    bruteforce_match: bool = Field(default=False)
    resource_ok: bool = Field(default=True)
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)


class MediumSignals(BaseModel):
    complexity_static_ok: bool = Field(default=False)
    complexity_dynamic_ok: bool = Field(default=False)
    estimated_complexity: str = Field(default="unknown")
    no_redundant_states: bool = Field(default=True)
    contract_ok: bool = Field(default=False)
    contract_violations: list[str] = Field(default_factory=list)


class WeakSignals(BaseModel):
    readability_score: int = Field(default=0, ge=0, le=5)
    style_score: int = Field(default=0, ge=0, le=5)
    explanation_consistent: bool = Field(default=False)
    notes: list[str] = Field(default_factory=list)


class Counterexample(BaseModel):
    minimal_input: str = Field(description="最小失败样例输入")
    expected_output: str = Field(description="期望输出")
    actual_output: str = Field(description="实际输出")
    error_category: str = Field(
        description="wrong_answer | runtime_error | time_limit_exceeded | memory_limit_exceeded"
    )
    suspected_reason: str = Field(default="", description="怀疑原因标签")


# ──────────────────────────────────────────────
# Top-level output
# ──────────────────────────────────────────────

class VerificationReport(BaseModel):
    """Verifier Agent 的最终输出。"""

    task_id: str
    verdict: str = Field(description="accept | reject | warning")
    strong_signals: StrongSignals = Field(default_factory=StrongSignals)
    medium_signals: MediumSignals = Field(default_factory=MediumSignals)
    weak_signals: WeakSignals = Field(default_factory=WeakSignals)
    counterexample: Counterexample | None = Field(default=None)
    summary: str = Field(default="")
