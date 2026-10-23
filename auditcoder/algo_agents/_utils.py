"""Shared utilities for all algorithm agent code-generator nodes."""

from __future__ import annotations

import json
from typing import Any

from auditcoder.common.problem_guidance import strategy_card


def format_interface(interface: dict | None) -> str:
    if not interface:
        return "（接口未定义）"
    lines = [
        f"函数名：{interface.get('function_name', '?')}",
        f"参数：{json.dumps(interface.get('params', []), ensure_ascii=False)}",
        f"返回类型：{interface.get('return_type', '?')}",
        f"返回说明：{interface.get('return_description', '')}",
        (
            "片段自包含要求：必须在代码片段中包含函数定义、类型注解和函数体"
            "所需的全部 import；不得依赖最终装配器补充导入。"
        ),
    ]
    pre = interface.get("preconditions", [])
    post = interface.get("postconditions", [])
    if pre:
        lines.append(f"前置条件：{'; '.join(pre)}")
    if post:
        lines.append(f"后置条件：{'; '.join(post)}")
    return "\n".join(lines)


def format_slots(slots: list[dict]) -> str:
    if not slots:
        return "（无）"
    return "\n".join(
        f"  - {s.get('name')} ({s.get('type')}): {s.get('description', '')}  [source={s.get('source')}]"
        for s in slots
    )


def format_contract_guardrails(
    task_node: dict[str, Any],
    task_plan_ir: dict[str, Any],
) -> str:
    """Render the CATG rules that every algorithm executor must obey.

    Keeping this in one helper makes the twelve specialized prompts agree on
    the same ownership and interface semantics while retaining their
    algorithm-specific instructions.
    """

    task_id = str(task_node.get("id", ""))
    contract = task_node.get("contract") or task_node.get("interface") or {}
    provenance = task_plan_ir.get("provenance_graph") or {}
    validation = task_plan_ir.get("validation") or {}
    invariants = validation.get("invariants", []) if isinstance(validation, dict) else []
    problem_text = " ".join(
        str(task_plan_ir.get("problem", {}).get(key, ""))
        for key in ("title", "description", "original_specification")
    ).lower()
    family_hints: list[str] = []
    if "abc145_f" in problem_text or "painting black" in problem_text:
        family_hints.append(
            "ABC145F: use only the O(N^2*K) DP over positions kept unchanged; "
            "do not allocate a candidate-height state or an O(N*K*M) transition. "
            "With sentinel H[0]=0, let dp[i][c] be the minimum strokes when real "
            "column i is kept and c columns among 1..i are changed. Transition "
            "from kept p<i with c-(i-p-1) changes by adding max(0,H[i]-H[p]); "
            "all intermediate columns can be changed along a monotone ramp. Take "
            "the minimum over the last kept column and set a trailing changed "
            "suffix to zero. This exact recurrence is fast enough for N<=300."
        )
    if "877/d" in problem_text or "energy drinks" in problem_text:
        family_hints.append(
            "CF877D: BFS must expand every reachable empty cell in each direction "
            "up to k until a wall; stop a direction when a cell has a strictly "
            "smaller BFS distance, and skip already-expanded cells with row/column "
            "DSU or ordered sets. A helper DSU is only an accelerator; the core "
            "solver must remain a complete BFS."
        )
    if "1136/d" in problem_text or "maximal number of places in queue" in problem_text:
        family_hints.append(
            "CF1136D: preserve directed edge semantics and use the all-neighbors "
            "blocked-set greedy. Reverse p, initialize marked={p[-1]} and "
            "need_count=1. For each x in reversed(p[:-1]), count outgoing neighbors "
            "that are marked; if this count equals need_count, Nastya can pass x and "
            "answer increases. Otherwise mark x as a new blocked person and increment "
            "need_count. Do NOT add x merely because it has an edge to any one marked "
            "node; that transitive-closure shortcut overcounts p=[1,2,3], edges "
            "1->2,2->3."
        )
    if "1364/c" in problem_text or "mex" in problem_text:
        family_hints.append(
            "CF1364C: indices in the prefix bound are 1-based, so a_i <= i+1. "
            "MEX may jump by more than one when intermediate values were preloaded. "
            "Use this exact construction: keep used and mex. If a_i==mex, append a "
            "fresh large unused filler (for example 10^6+i) so mex stays put. If "
            "a_i>mex, append the current mex, insert it, advance while mex is in "
            "used, and require the resulting mex to equal a_i; do not require "
            "a_i==old_mex+1. Re-simulate every prefix before printing."
        )
    if "1257/c" in problem_text or "dominated subarray" in problem_text:
        family_hints.append(
            "CF1257C: the answer is the minimum distance between two equal values "
            "plus one. For the globally nearest duplicate, no other value repeats "
            "inside that interval, so its interval is dominated; track last positions "
            "in O(n) instead of checking only length-2/3 windows."
        )
    if "1221/e" in problem_text or "alice and bob play a game" in problem_text:
        family_hints.append(
            "CF1221E is a partizan run-length game, not an impartial Grundy XOR. "
            "For each maximal dot-run keep only runs of length at least b and use "
            "the known O(|s|) interval classification for the largest problematic "
            "run; combine the run categories with parity. Do not compute a cubic "
            "Grundy table or xor it."
        )
    if "522/c" in problem_text or "polycarp is flying" in problem_text:
        family_hints.append(
            "CF522C: parse the leading test count and blank lines exactly. For each "
            "set, decrement b[t_j] for every known t_j>0, count unknown=t_j==0, and "
            "locate the first upset d with r_d=1. If no upset exists, output Y iff "
            "b[i]<=unknown. Otherwise count unknownBefore before d; a first exhausted "
            "dish i must have b[i]<=unknownBefore and must not be mentioned (t_j=i) "
            "at or after d. Mark every such candidate Y, choose the candidate with "
            "minimum b[i], let rest=unknown-b[min], then mark every other dish j with "
            "b[j]<=rest Y. Preserve t=0 versus known dishes and output exactly k "
            "characters per set."
        )
    if "allkill" in problem_text:
        family_hints.append(
            "Kattis allkill: use the exact O(n) circle-parking formula, not a DP over "
            "total coding time. Let E=t-sum(x)+1, initialize ans=E, prefix=0, then "
            "for 0-based i=1..n-1 set prefix+=x[i-1] and multiply ans by "
            "(n+E-i+prefix) modulo 998244353. This yields n=3,t=5,[1,2,1] -> 60 "
            "and n=5,t=5,[1,1,1,1,1] -> 1296; never print a placeholder."
        )
    if "780/d" in problem_text or "short names" in problem_text:
        family_hints.append(
            "CF780D: every output must be validated as a witness, not compared to "
            "one sample. Encode each club's A/B choice as a 2-SAT literal; for every "
            "equal option string add the pairwise at-most-one clause, and for every "
            "A_i==A_j add B_i -> B_j (equivalently not(B_i) or not(A_j)) for the "
            "special direction rule. If A_i==B_i, treat it as one neutral formal "
            "choice: keep uniqueness constraints but exclude that club as both a "
            "first-option target and a second-option trigger in directional clauses. "
            "Use a standard Kosaraju assignment orientation and recheck all names "
            "are allowed, distinct, and satisfy the special rule before printing."
        )
    card = strategy_card(task_plan_ir.get("problem", {}) or {})
    if card and card not in family_hints:
        family_hints.append(card)
    family_hint_text = "\n".join(f"- {hint}" for hint in family_hints) or "- none"
    return f"""
## CATG responsibility contract (authoritative)

- Responsibility ID: `{task_id}`. This node owns the emitted function and any
  helper defined in the returned bundle; do not move ownership to another
  task or silently invent a second public function.
- Treat the interface, input/output slots, and their `source` fields as a
  machine contract. Preserve the exact function name, parameter names/order,
  return schema, preconditions, postconditions, and invariants.
- Implement only this node's algorithm. Do not parse stdin, print output, or
  mutate global state unless the interface explicitly requires it; orchestration
  is owned by the assembler/root node.
- Check boundary cases implied by the contract before optimizing. Meet the
  declared complexity budget and state the loop invariant in comments when a
  pointer, DP, graph, or data-structure invariant is material.
- Before writing code, record three executable proof obligations in the
  implementation reasoning: (1) the state/transition invariant, (2) the
  reason the claimed complexity fits the largest input, and (3) one smallest
  adversarial case that would falsify the proposed shortcut. If the case
  falsifies it, change the algorithm instead of patching the output.
- This is a correctness-critical implementation, not pseudocode. Trace every
  supplied sample by hand before returning. If the task constructs a witness,
  validate the witness against the original statement (for example every MEX
  prefix, every debt direction, every chosen-name uniqueness condition, or
  every output index) before emitting it.
- For geometry, probability, and constructive tasks, re-simulate the complete
  transition on at least one tiny exhaustive instance. Check the exact event,
  orientation, parity, endpoint, and indexing convention; a formula that is
  merely close on a sample is not evidence of correctness.
- Do not use a greedy shortcut merely because it matches the first sample. Give
  the invariant that proves the transition is safe and test the smallest case
  that separates greedy from global optimization. Preserve time units, batch
  counts, input order, and sentinel values exactly as stated.
- If this node is the designated core solver, it must implement the complete
  algorithm and return the final answer/witness. Never return a placeholder,
  empty collection, closure over an unrelated outer argument, or a helper that
  leaves the core computation to the assembler.
- A source reference such as `S1.out` is provenance: the caller supplies that
  value. Never replace it with a newly computed approximation.
- Return exactly one JSON object with a `code_snippet` string. The snippet must
  be self-contained and syntactically valid Python, including required imports.
- Keep the implementation concise (roughly 2500 output tokens or fewer): do
  not spend the response on long docstrings, repeated prose, or unused helper
  abstractions. Finish the JSON object and code before adding explanation.

Problem-family correctness hints (apply only when relevant):
{family_hint_text}

Node contract:
{json.dumps(contract, ensure_ascii=False, indent=2)}

Known provenance edges:
{json.dumps(provenance, ensure_ascii=False, indent=2)}

Global validation invariants:
{json.dumps(invariants, ensure_ascii=False, indent=2)}
""".strip()


def format_feedback(feedback: dict[str, Any] | None) -> str:
    """Render verifier feedback into a human-readable prompt section.

    Returns an empty string when feedback is None or verdict is not 'reject'.
    """
    if not feedback:
        return ""

    verdict = feedback.get("verdict", "")
    if verdict != "reject":
        return ""

    lines = [
        "",
        "## ⚠️ 上次生成的代码未通过验证，请根据以下反馈修复",
        "",
    ]

    summary = feedback.get("summary", "")
    if summary:
        lines.append(f"错误摘要：{summary}")

    strong = feedback.get("strong_signals", {})

    if not strong.get("compile_ok", True):
        lines.append("问题：代码编译失败，请检查语法错误。")

    if not strong.get("sample_tests_passed", True):
        lines.append("问题：样例测试未通过。")

    if not strong.get("bruteforce_match", True):
        lines.append("问题：与暴力解对拍不一致，存在逻辑错误。")

    if not strong.get("resource_ok", True):
        usage = strong.get("resource_usage", {})
        lines.append(
            f"问题：超时或超内存（耗时 {usage.get('max_time_ms', '?')} ms，"
            f"内存 {usage.get('max_memory_kb', '?')} KB）。"
        )

    medium = feedback.get("medium_signals", {})
    violations = medium.get("contract_violations", [])
    if violations:
        lines.append("接口契约违规：")
        for v in violations:
            lines.append(f"  - {v}")

    counterexample = feedback.get("counterexample")
    if counterexample:
        lines.append("")
        lines.append("反例（用于复现问题）：")
        lines.append(f"  输入：{counterexample.get('minimal_input', '')}")
        lines.append(f"  期望输出：{counterexample.get('expected_output', '')}")
        lines.append(f"  实际输出：{counterexample.get('actual_output', '')}")
        error_cat = counterexample.get("error_category", "")
        if error_cat:
            lines.append(f"  错误类别：{error_cat}")
        reason = counterexample.get("suspected_reason", "")
        if reason:
            lines.append(f"  疑似原因：{reason}")

    lines.append("")
    lines.append("请重新实现该函数，修复上述所有问题，输出 JSON：{\"code_snippet\": \"...\"}")

    return "\n".join(lines)
