"""Deterministic localization and bounded-repair primitives.

The paper describes localization and repair as a small, auditable state
machine around the model calls.  This module contains that state-machine
logic.  It deliberately has no LangGraph or LLM dependency so that the same
rules can be exercised by unit tests and by the controlled-fault experiment.

The public helpers accept ordinary dictionaries because the runtime IR is
serialized JSON.  Every returned object is JSON serializable as well.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping, Sequence


# Obligation classes and global symptoms used by the method.
SEMANTIC = "semantic"
INTERFACE = "interface"
RESOURCE = "resource"
COMMITMENT = "commitment"
IMPLEMENTATION = "implementation"

TRACE = "Trace"
RESOURCE_RULE = "Resource"
LOCAL = "Local"
GLOBAL_ATTRIBUTION = "GlobalAttribution"

HIGH = "High"
MEDIUM = "Medium"
LOW = "Low"

PASS = "Pass"
WRONG_ANSWER = "WA"
RUNTIME_ERROR = "RE"
TIMEOUT = "TLE"
MEMORY_ERROR = "MLE"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _node_id(node: Mapping[str, Any]) -> str:
    return str(node.get("id") or "")


def node_map(ir: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a stable ID-to-node map, ignoring malformed entries."""

    result: dict[str, dict[str, Any]] = {}
    for raw in ir.get("task_nodes", []) or []:
        if not isinstance(raw, Mapping):
            continue
        identifier = _node_id(raw)
        if identifier:
            result[identifier] = dict(raw)
    return result


def root_id(ir: Mapping[str, Any], nodes: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    nodes = nodes or node_map(ir)
    candidate = str(ir.get("root_id") or "")
    if candidate in nodes:
        return candidate
    for identifier, node in nodes.items():
        if not node.get("parent"):
            return identifier
    return ""


def code_owner_ids(
    ir: Mapping[str, Any],
    code_snippets: Mapping[str, Any] | None = None,
) -> set[str]:
    """Compute ``U`` (code-owning responsibilities).

    Normal plans use leaf nodes.  The extra ``owns_code``/``owned_code``
    fields make internal orchestration owners representable without changing
    the existing TaskPlanIR schema.
    """

    snippets = code_snippets or {}
    owners: set[str] = set()
    for identifier, node in node_map(ir).items():
        if (
            node.get("owns_code")
            or node.get("owned_code")
            or node.get("is_code_owner")
            or (node.get("is_leaf") and node.get("agent_type"))
            or identifier in snippets
        ):
            owners.add(identifier)
    return owners


def hierarchy_children(ir: Mapping[str, Any]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for identifier, node in node_map(ir).items():
        parent = str(node.get("parent") or "")
        if parent:
            children[parent].append(identifier)
    for values in children.values():
        values.sort()
    return dict(children)


def hierarchy_subtree(ir: Mapping[str, Any], start: str) -> set[str]:
    """Return ``S_h(start)`` using only the parent relation."""

    nodes = node_map(ir)
    if start not in nodes:
        return set()
    children = hierarchy_children(ir)
    result: set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        stack.extend(reversed(children.get(current, [])))
    return result


def hierarchy_depths(ir: Mapping[str, Any]) -> dict[str, int]:
    nodes = node_map(ir)
    depths: dict[str, int] = {}
    for identifier in nodes:
        depth = 0
        seen: set[str] = set()
        current = identifier
        while current in nodes and nodes[current].get("parent"):
            parent = str(nodes[current].get("parent"))
            if parent in seen:
                break
            seen.add(parent)
            current = parent
            depth += 1
        depths[identifier] = depth
    return depths


def hierarchy_ancestors(ir: Mapping[str, Any], implicated: Iterable[str]) -> list[str]:
    """Return non-root common ancestors, deepest first."""

    nodes = node_map(ir)
    members = [identifier for identifier in implicated if identifier in nodes]
    if not members:
        return []
    root = root_id(ir, nodes)
    subtrees = {identifier: hierarchy_subtree(ir, identifier) for identifier in nodes}
    ancestors = [
        identifier
        for identifier, subtree in subtrees.items()
        if identifier != root and all(member in subtree for member in members)
    ]
    depths = hierarchy_depths(ir)
    return sorted(ancestors, key=lambda identifier: (-depths.get(identifier, 0), identifier))


def _normalise_class(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "sem": SEMANTIC,
        "semantic": SEMANTIC,
        "postcondition": SEMANTIC,
        "invariant": SEMANTIC,
        "value": SEMANTIC,
        "impl": IMPLEMENTATION,
        "implementation": IMPLEMENTATION,
        "interface": INTERFACE,
        "ifc": INTERFACE,
        "signature": INTERFACE,
        "provenance": INTERFACE,
        "resource": RESOURCE,
        "complexity": RESOURCE,
        "budget": RESOURCE,
        "commitment": COMMITMENT,
        "cmt": COMMITMENT,
        "strategy": COMMITMENT,
        "decomposition": COMMITMENT,
    }
    return aliases.get(text, text)


def obligation_class(obligation: Any) -> str | None:
    if obligation is None or obligation == "":
        return None
    if isinstance(obligation, Mapping):
        value = obligation.get("class", obligation.get("cls", obligation.get("kind")))
        if value is not None:
            return _normalise_class(value)
    return _normalise_class(obligation)


def _symptom(value: Any, execution_result: Mapping[str, Any] | None = None) -> str:
    text = str(value or "").strip().lower()
    if text in {"wa", "wrong_answer", "wrong answer", "wrong-answer"}:
        return WRONG_ANSWER
    if text in {"re", "runtime_error", "runtime", "compile_error", "contract_error"}:
        return RUNTIME_ERROR
    if text in {"tle", "timeout", "time_limit", "time_limit_exceeded"}:
        return TIMEOUT
    if text in {"mle", "memory_error", "memory_limit", "memory_limit_exceeded"}:
        return MEMORY_ERROR
    if text in {"pass", "none", "ok", ""}:
        if execution_result:
            if execution_result.get("timed_out"):
                return TIMEOUT
            if execution_result.get("error_type") == "timeout":
                return TIMEOUT
            if execution_result.get("error_type") in {"runtime_error", "contract_error"}:
                return RUNTIME_ERROR
            if execution_result.get("error_type") == "wrong_answer":
                return WRONG_ANSWER
        return PASS
    return str(value)


def symptom_classes(symptom: str) -> set[str]:
    symptom = _symptom(symptom)
    if symptom in {WRONG_ANSWER, RUNTIME_ERROR}:
        return {SEMANTIC, INTERFACE, COMMITMENT}
    if symptom in {TIMEOUT, MEMORY_ERROR}:
        return {RESOURCE, COMMITMENT}
    return set()


def _case_input(anchor: Mapping[str, Any]) -> Any:
    return anchor.get("input", anchor.get("stdin", anchor.get("case_input")))


def _first_failing_case(execution_result: Mapping[str, Any]) -> dict[str, Any] | None:
    results = execution_result.get("results") or execution_result.get("checks") or []
    if isinstance(results, Mapping):
        results = list(results.values())
    for index, raw in enumerate(results):
        if not isinstance(raw, Mapping):
            continue
        if not raw.get("passed", raw.get("status") in {PASS, "pass"}):
            item = dict(raw)
            item.setdefault("case_index", index)
            return item
    if execution_result.get("all_passed") is False:
        return {
            "case_index": 0,
            "input": execution_result.get("input", execution_result.get("stdin", "")),
            "expected": execution_result.get("expected", ""),
            "actual": execution_result.get("actual", ""),
            "stderr": execution_result.get("stderr", ""),
            "timed_out": bool(execution_result.get("timed_out")),
        }
    return None


def anchor_case(execution_result: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the first failed ordered development check (``chi*``)."""

    failing = _first_failing_case(execution_result)
    if failing is None:
        return None
    symptom = _symptom(
        failing.get("symptom", failing.get("error_type")),
        {**execution_result, **failing},
    )
    if symptom == PASS:
        symptom = _symptom(execution_result.get("error_type"), execution_result)
    failing["symptom"] = symptom
    return failing


def _function_defs(source: str) -> dict[str, tuple[int, int]]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError):
        return {}
    result: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            result[node.name] = (int(node.lineno), int(end))
    return result


def build_owner_map(
    ir: Mapping[str, Any],
    code_snippets: Mapping[str, str] | None = None,
    final_code: str = "",
    registered_units: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Build ``function/helper -> owner IDs`` (the executable ``mu`` map).

    Lists are retained instead of overwriting duplicate names.  Duplicate
    names are therefore ambiguous and cause traceback localization to abstain.
    """

    result: dict[str, list[str]] = defaultdict(list)
    nodes = node_map(ir)
    owners = code_owner_ids(ir, code_snippets)
    for identifier, node in nodes.items():
        iface = _as_dict(node.get("interface"))
        function_name = str(iface.get("function_name") or "")
        if function_name and identifier in owners:
            result[function_name].append(identifier)
        for unit in node.get("owned_units", []) or []:
            if isinstance(unit, Mapping):
                name = str(unit.get("name") or unit.get("function_name") or "")
            else:
                name = str(unit or "")
            if name:
                result[name].append(identifier)
        owned = node.get("owned_code")
        if isinstance(owned, Mapping):
            for name in owned:
                if name:
                    result[str(name)].append(identifier)
        elif isinstance(owned, (list, tuple, set)):
            for name in owned:
                if name:
                    result[str(name)].append(identifier)
    registry = registered_units or {}
    unit_entries = registry.get("units") if isinstance(registry, Mapping) else None
    if isinstance(unit_entries, list):
        for entry in unit_entries:
            if not isinstance(entry, Mapping):
                continue
            name = entry.get("name", entry.get("function_name"))
            owner = entry.get("owner", entry.get("task_id"))
            if name is None or owner is None:
                continue
            owners_for_unit = owner if isinstance(owner, (list, tuple, set)) else [owner]
            for identifier in owners_for_unit:
                if str(identifier):
                    result[str(name)].append(str(identifier))
    for name, owner in (registry.items() if isinstance(registry, Mapping) else ()):
        if name == "units":
            continue
        owners = owner if isinstance(owner, (list, tuple, set)) else [owner]
        for identifier in owners:
            if str(identifier):
                result[str(name)].append(str(identifier))
    # A snippet may contain helpers that the IR does not list.  It is still
    # useful to retain their names when a unit registry explicitly names the
    # snippet owner; no owner is inferred from source alone.
    for identifier, source in (code_snippets or {}).items():
        node = nodes.get(str(identifier), {})
        for name in _function_defs(str(source)):
            if name == str(_as_dict(node.get("interface")).get("function_name") or ""):
                result[name].append(str(identifier))
    return {name: sorted(set(owners)) for name, owners in result.items()}


def _build_interface_map(ir: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Backward-compatible interface map used by existing tests and callers."""

    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for identifier, node in node_map(ir).items():
        iface = _as_dict(node.get("interface"))
        function_name = str(iface.get("function_name") or "")
        if function_name:
            result[function_name].append(
                {"task_id": identifier, "agent_type": str(node.get("agent_type") or "")}
            )
    return dict(result)


def _traceback_function_names(stderr: str) -> list[str]:
    # Python traceback frames use ``in function``.  Keep source order and let
    # the caller inspect the innermost registered frame first.
    return re.findall(r"\bin\s+([A-Za-z_]\w*)", str(stderr or ""))


def traceback_has_generated_frame(stderr: str, interface_map: Mapping[str, Any]) -> bool:
    """Whether a traceback contains a frame that can be identified as generated code."""

    names = _traceback_function_names(stderr)
    if not names:
        return False
    if any(name in interface_map for name in names):
        return True
    # ``<string>``/``generated.py`` frames may not include ``in`` when a test
    # supplies a compact traceback.  Treat an explicit generated marker as a
    # triggered trace rule; owner resolution will still be required.
    lowered = str(stderr or "").lower()
    return "<string>" in lowered or "generated" in lowered or "auditcoder_injected" in lowered


def locate_traceback(
    stderr: str,
    owner_map: Mapping[str, Sequence[str]],
) -> tuple[dict[str, Any] | None, bool, str | None]:
    """Resolve the innermost generated frame through ``mu``.

    The second return value tells the caller that Trace was triggered.  A
    triggered but unresolved/ambiguous frame must abstain rather than fall
    through to weaker local evidence.
    """

    names = _traceback_function_names(stderr)
    triggered = traceback_has_generated_frame(stderr, owner_map)
    harness_names = {
        "wrapper",
        "exec",
        "run",
        "execute",
        "full_code_runner",
        "post_repair_code_runner",
        "_run",
        "_execute",
    }
    for function_name in reversed(names):
        owners = list(owner_map.get(function_name, ()))
        if not owners:
            # Once Trace is triggered, an unknown non-harness frame is a
            # generated unit with no registered ``mu`` entry.  Do not skip it
            # and attribute the failure to an outer frame.
            if function_name not in harness_names:
                return None, True, "unresolved_owner"
            continue
        if len(owners) != 1:
            return None, True, "ambiguous_owner"
        return (
            {
                "owners": [owners[0]],
                "task_id": owners[0],
                "function_name": function_name,
                "rule": TRACE,
                "method": "traceback",
                "confidence": HIGH,
            },
            True,
            None,
        )
    if triggered:
        return None, True, "unresolved_owner"
    return None, False, None


def _locate_by_traceback(
    stderr: str,
    interface_map: Mapping[str, Sequence[Mapping[str, str]]],
) -> dict[str, str] | None:
    """Compatibility wrapper for the pre-paper locator API."""

    owner_map: dict[str, list[str]] = defaultdict(list)
    for function_name, owners in interface_map.items():
        for owner in owners:
            identifier = str(owner.get("task_id") or "")
            if identifier:
                owner_map[function_name].append(identifier)
    location, triggered, _ = locate_traceback(stderr, owner_map)
    if not location or not triggered:
        return None
    return {
        "task_id": str(location["task_id"]),
        "agent_type": str(next(
            (
                owner.get("agent_type", "")
                for owner in interface_map.get(location["function_name"], [])
                if owner.get("task_id") == location["task_id"]
            ),
            "",
        )),
        "function_name": str(location["function_name"]),
        "method": "traceback",
        "confidence": HIGH.lower(),
    }


def _locate_by_verification_reports(
    verification_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, str] | None:
    """Legacy helper: only explicit unique rejections are considered.

    The paper locator itself additionally requires an obligation and witness;
    callers using this compatibility helper retain the old shape but no longer
    receive a location for ambiguous reports.
    """

    rejected = [
        (str(task_id), report)
        for task_id, report in verification_reports.items()
        if isinstance(report, Mapping) and str(report.get("verdict", "")).lower() == "reject"
    ]
    if len(rejected) != 1:
        return None
    task_id, report = rejected[0]
    return {
        "task_id": task_id,
        "agent_type": str(report.get("agent_type", "")),
        "method": "verification_report",
        "confidence": LOW.lower(),
    }


def _record_current(record: Mapping[str, Any], versions: Mapping[str, Any]) -> bool:
    if record.get("current") is False or record.get("stale") is True:
        return False
    version = record.get("version", record.get("ver"))
    if version is None:
        return True
    if isinstance(version, Mapping):
        for identifier, expected in version.items():
            if str(versions.get(str(identifier), 0)) != str(expected):
                return False
        return True
    owners = record.get("owners", record.get("own", [])) or []
    if not isinstance(owners, (list, tuple, set)):
        owners = [owners]
    return all(str(versions.get(str(identifier), 0)) == str(version) for identifier in owners)


def _owners(record: Mapping[str, Any]) -> set[str]:
    value = record.get("owners", record.get("own", record.get("task_ids", [])))
    if isinstance(value, str):
        return {value} if value else set()
    return {str(item) for item in (value or []) if str(item)}


def _obligation(record: Mapping[str, Any]) -> Any:
    return record.get("obligation", record.get("obl", record.get("contract_obligation")))


def _witness(record: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(record.get("witness", record.get("wit", {})))


def _provider_pairs(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    obligation = _as_dict(_obligation(record))
    pairs = record.get("provider_pairs", obligation.get("provider_pairs", obligation.get("providers", [])))
    if isinstance(pairs, Mapping):
        pairs = [pairs]
    result: list[tuple[str, str]] = []
    for pair in pairs or []:
        if isinstance(pair, Mapping):
            left = pair.get("provider", pair.get("source", pair.get("from")))
            right = pair.get("consumer", pair.get("target", pair.get("to")))
            if left is not None and right is not None:
                result.append((str(left), str(right)))
        elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
            result.append((str(pair[0]), str(pair[1])))
    direct = obligation.get("provider_pair")
    if isinstance(direct, (list, tuple)) and len(direct) >= 2:
        result.append((str(direct[0]), str(direct[1])))
    return result


def _input_matches(witness: Mapping[str, Any], anchor: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return (eligible, observed applicability)."""

    if witness.get("applicable") is False or witness.get("applies") is False:
        return False, False
    if witness.get("applicability") in {"false", False}:
        return False, False
    observed = bool(
        witness.get("applicability_observed")
        or witness.get("observed")
        or witness.get("executed_input")
        or witness.get("call_recorded")
        or str(witness.get("applicability", "")).lower() == "observed"
    )
    anchor_index = anchor.get("case_index", anchor.get("index"))
    witness_index = witness.get("case_index", witness.get("index"))
    if anchor_index is not None and witness_index is not None and str(anchor_index) != str(witness_index):
        return False, observed
    anchor_input = _case_input(anchor)
    witness_input = witness.get("input", witness.get("stdin", witness.get("case_input")))
    if anchor_input is not None and witness_input is not None and anchor_input != witness_input:
        return False, observed
    condition = witness.get("input_condition", witness.get("condition"))
    if callable(condition):  # useful for in-process tests; never serialized
        try:
            if not condition(anchor_input):
                return False, observed
        except Exception:
            return False, observed
    return True, observed


def _reachable(record: Mapping[str, Any], ir: Mapping[str, Any], owners: set[str]) -> bool:
    if record.get("reachable") is False or record.get("reachability") is False:
        return False
    if record.get("reachable") is True or record.get("reachability") is True:
        return True
    witness = _witness(record)
    if witness.get("reachable") is False or witness.get("path") is False:
        return False
    # A serialized provenance path is definitive when supplied.
    if witness.get("provenance_path") or witness.get("path"):
        return True
    provenance = ir.get("provenance_graph") or ir.get("provenance")
    if isinstance(provenance, Mapping) and provenance:
        # The graph can only disprove reachability when explicit slot names are
        # present.  Otherwise leave the record eligible for legacy IRs.
        constrained = record.get("constrained_slot") or _as_dict(_obligation(record)).get("slot")
        observable = record.get("observable_slot") or witness.get("observable_slot")
        if constrained and observable:
            return _graph_reachable(provenance, str(constrained), str(observable))
    # Older TaskPlanIRs did not serialize G_rho.  Missing metadata is unknown,
    # whereas an explicit negative value is rejected above.
    return True


def _graph_reachable(graph: Mapping[str, Any], start: str, target: str) -> bool:
    if start == target:
        return True
    adjacency: dict[str, list[str]] = defaultdict(list)
    if isinstance(graph, Mapping):
        for source, destinations in graph.items():
            if isinstance(destinations, str):
                destinations = [destinations]
            for destination in destinations or []:
                adjacency[str(source)].append(str(destination))
    queue = deque([start])
    seen = {start}
    while queue:
        current = queue.popleft()
        for destination in adjacency.get(current, []):
            if destination == target:
                return True
            if destination not in seen:
                seen.add(destination)
                queue.append(destination)
    return False


def _precondition_scope(record: Mapping[str, Any], ir: Mapping[str, Any]) -> dict[str, Any]:
    """Apply C5 and return a possibly re-addressed record."""

    witness = _witness(record)
    violated = bool(
        witness.get("precondition_violated")
        or witness.get("violates_precondition")
        or record.get("precondition_violated")
    )
    owners = _owners(record)
    if not violated or len(owners) != 1:
        return dict(record)
    consumer = next(iter(owners))
    provider = witness.get("provider", witness.get("source_owner"))
    if provider is None:
        pairs = _provider_pairs(record)
        if pairs:
            provider = pairs[0][0]
    if provider is None:
        node = node_map(ir).get(consumer, {})
        slots = node.get("inputs", []) or []
        if slots:
            source = _as_dict(slots[0]).get("source")
            if source:
                provider = str(source).split(".", 1)[0]
    if provider is None or str(provider) == consumer:
        # No provider can be identified, so the record cannot support a
        # single-owner implementation claim.
        return {**record, "_precondition_scope_unresolved": True}
    obligation = _obligation(record)
    interface_obligation = {
        "class": INTERFACE,
        "provider_pair": [str(provider), consumer],
        "name": f"provenance:{provider}->{consumer}",
    }
    return {
        **record,
        "owners": [str(provider), consumer],
        "own": [str(provider), consumer],
        "obligation": interface_obligation,
        "obl": interface_obligation,
        "readdressed_from": obligation,
    }


def _same_condition(record: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    left = _witness(record)
    right = _witness(candidate)
    for key in ("case_index", "index", "input", "stdin", "case_input", "condition"):
        if key in left and key in right and left[key] != right[key]:
            return False
    return True


def compatible_record(
    record: Mapping[str, Any],
    anchor: Mapping[str, Any],
    ir: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]] | None = None,
    versions: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Evaluate C1--C6 and return (record, observed_applicability)."""

    versions = versions or {}
    candidate = _precondition_scope(record, ir)
    if candidate.get("_precondition_scope_unresolved"):
        return None, False
    owners = _owners(candidate)
    obligation = _obligation(candidate)
    cls = obligation_class(obligation)
    if not owners or not _record_current(candidate, versions):  # C1
        return None, False
    verdict = str(candidate.get("verdict", candidate.get("vd", ""))).lower()
    if verdict not in {"reject", "rejected", "fail", "failed"}:  # C1
        return None, False
    if not cls or cls not in symptom_classes(str(anchor.get("symptom", ""))):  # C2
        return None, False
    if not _reachable(candidate, ir, owners):  # C3
        return None, False
    applicable, observed = _input_matches(_witness(candidate), anchor)  # C4
    if not applicable:
        return None, False
    # C6: a passing executed record under the same condition contradicts this
    # rejection.  Keep judged records out of the contradiction test because
    # they do not carry an executed witness.
    for other in records or ():
        if other is record:
            continue
        other_owners = _owners(other)
        other_kind = str(other.get("kind", other.get("evidence_kind", ""))).lower()
        other_verdict = str(other.get("verdict", other.get("vd", ""))).lower()
        if (
            other_owners & owners
            and other_kind in {"exec", "executed", "execution", "runtime"}
            and other_verdict in {"pass", "passed", "accept", "accepted"}
            and obligation_class(_obligation(other)) == cls
            and _same_condition(candidate, other)
        ):
            return None, False
    result = dict(candidate)
    result["owners"] = sorted(owners)
    result["own"] = sorted(owners)
    result["obligation_class"] = cls
    result["applicability_observed"] = observed
    result["current"] = True
    return result, observed


def coherent_records(
    records: Sequence[Mapping[str, Any]],
    implicated: set[str],
    ir: Mapping[str, Any],
) -> bool:
    if not records:
        return False
    commitment_records = [
        record
        for record in records
        if obligation_class(_obligation(record)) == COMMITMENT
    ]
    if commitment_records:
        commitments = {
            str(
                _as_dict(_obligation(record)).get(
                    "commitment",
                    _as_dict(_obligation(record)).get("name", record.get("commitment", "")),
                )
            )
            for record in commitment_records
        }
        commitments.discard("")
        if len(commitments) != 1:
            return False
        commitment = next(iter(commitments))
        nodes = node_map(ir)
        matching = [
            identifier
            for identifier, node in nodes.items()
            if str(node.get("commitment", node.get("goal", identifier))) == commitment
            or identifier == commitment
        ]
        if len(matching) != 1:
            return False
        return implicated <= hierarchy_subtree(ir, matching[0])
    if len(implicated) <= 1:
        return True
    edges: list[tuple[str, str]] = []
    for record in records:
        edges.extend(_provider_pairs(record))
    adjacency: dict[str, set[str]] = {identifier: set() for identifier in implicated}
    for left, right in edges:
        if left in implicated and right in implicated:
            adjacency[left].add(right)
            adjacency[right].add(left)
    if not adjacency or any(not values for values in adjacency.values()):
        return False
    seen: set[str] = set()
    queue = deque([next(iter(implicated))])
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        queue.extend(adjacency[current] - seen)
    return seen == implicated


def defect_class(records: Sequence[Mapping[str, Any]]) -> str:
    classes = {obligation_class(_obligation(record)) for record in records}
    if COMMITMENT in classes:
        return COMMITMENT
    if INTERFACE in classes:
        return INTERFACE
    return IMPLEMENTATION


def _resource_violations(
    execution_result: Mapping[str, Any],
    ir: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    measurements = execution_result.get("owner_measurements")
    if measurements is None:
        measurements = execution_result.get("resource_measurements")
    if not isinstance(measurements, Mapping):
        return {}
    nodes = node_map(ir)
    result: dict[str, list[dict[str, Any]]] = {}
    for identifier, raw_values in measurements.items():
        node = nodes.get(str(identifier), {})
        budget = node.get("complexity_budget", node.get("resource_budget", {}))
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        for raw in values:
            measurement = _as_dict(raw)
            violated = measurement.get("violated")
            if violated is None:
                metric = str(measurement.get("metric", "time")).lower()
                observed = measurement.get("observed", measurement.get("value"))
                limit = measurement.get("limit")
                if limit is None and isinstance(budget, Mapping):
                    limit = budget.get("time_ms" if metric == "time" else "memory_kb")
                try:
                    violated = observed is not None and limit is not None and float(observed) > float(limit)
                except (TypeError, ValueError):
                    violated = False
            if violated:
                result.setdefault(str(identifier), []).append(measurement)
    return result


def _global_attribution_candidates(
    ir: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    code_snippets: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Rank CATG code owners for a black-box whole-program failure.

    APPS gives us a trusted failing input/output pair, but no stack frame.  A
    global attribution is therefore weaker than Trace/Resource/Local evidence
    and must be explicit, deterministic, and auditable.  A root with an
    actual owned bundle is the strongest signal because it owns the assembled
    program; a root that only owns ``solve``/``__main__`` orchestration is not
    treated as the algorithm owner.  Otherwise a rejected local responsibility
    is preferred, followed by a generated code owner.  The scores are
    diagnostics, not an LLM guess and are only used when the stronger locator
    rules found nothing.
    """

    nodes = node_map(ir)
    code_owners = code_owner_ids(ir, code_snippets)
    if not code_owners:
        return []
    root = root_id(ir, nodes)
    signal_by_owner: dict[str, dict[str, int]] = {
        identifier: {"rejected": 0, "warning": 0, "records": 0}
        for identifier in code_owners
    }
    for raw in records:
        if not isinstance(raw, Mapping):
            continue
        owners = _owners(raw)
        for owner in owners & code_owners:
            signal = signal_by_owner[owner]
            signal["records"] += 1
            verdict = str(raw.get("verdict", raw.get("vd", ""))).lower()
            if verdict in {"reject", "rejected", "fail", "failed"}:
                signal["rejected"] += 1
            elif verdict in {"warning", "warn"}:
                signal["warning"] += 1

    candidates: list[dict[str, Any]] = []
    depths = hierarchy_depths(ir)
    snippets = code_snippets or {}
    for identifier in sorted(code_owners):
        node = nodes.get(identifier, {})
        signal = signal_by_owner[identifier]
        # ``register_assembly_units`` marks the root as a code owner when it
        # registers only ``solve``/``__main__``.  Those are orchestration
        # units, so prefer the leaf bundle that can actually be regenerated.
        orchestration_only = (
            identifier == root
            and not node.get("owned_code")
            and identifier not in snippets
            and bool(node.get("owned_units"))
        )
        explicit_root = identifier == root and not orchestration_only and bool(
            node.get("owns_code")
            or node.get("owned_code")
            or node.get("is_code_owner")
            or identifier in snippets
        )
        # Keep the score intentionally coarse.  It is only a tie-breaker for
        # a repair target, while the reason fields preserve why it won.
        score = 0
        reasons: list[str] = []
        if explicit_root:
            score += 1000
            reasons.append("explicit_root_program_owner")
        if signal["rejected"]:
            score += 80 * signal["rejected"]
            reasons.append("local_rejection_signal")
        if signal["warning"]:
            score += 20 * signal["warning"]
            reasons.append("local_warning_signal")
        if identifier in snippets:
            score += 10
            reasons.append("generated_code_bundle")
        if node.get("owns_code") or node.get("is_code_owner"):
            score += 5
            reasons.append("explicit_code_owner")
        score += min(int(depths.get(identifier, 0)), 10)
        candidates.append(
            {
                "task_id": identifier,
                "score": score,
                "reasons": reasons,
                "local_records": signal["records"],
                "local_rejections": signal["rejected"],
                "local_warnings": signal["warning"],
            }
        )
    return sorted(
        candidates,
        key=lambda item: (-int(item["score"]), str(item["task_id"])),
    )


def _locate_by_global_attribution(
    execution_result: Mapping[str, Any],
    anchor: Mapping[str, Any],
    ir: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    code_snippets: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Attribute a trusted black-box failure to one CATG owner.

    This is the deliberate fallback after Trace, Resource, and Local.  It is
    limited to an observed failed run and carries ``Low`` confidence so the
    report and repair trace distinguish it from direct evidence.
    """

    if execution_result.get("all_passed", True):
        return None
    trusted_failures = execution_result.get("trusted_failures")
    if trusted_failures is not None and not trusted_failures:
        # Advisory planner mismatches are not a benchmark oracle.
        return None
    candidates = _global_attribution_candidates(ir, records, code_snippets)
    if not candidates:
        return None
    selected = candidates[0]
    owner = str(selected["task_id"])
    evidence = {
        "id": f"global-attribution:{anchor.get('case_index', 0)}:{owner}",
        "kind": "exec",
        "source": "global_execution_attribution",
        "verdict": "Reject",
        "owners": [owner],
        "obligation": {
            "class": IMPLEMENTATION,
            "name": "full_program_output",
        },
        "witness": {
            "case_index": anchor.get("case_index"),
            "input": _case_input(anchor),
            "expected": anchor.get("expected", ""),
            "actual": anchor.get("actual", ""),
            "symptom": anchor.get("symptom", ""),
            "applicability": "observed",
            "attribution_basis": selected["reasons"],
        },
        "current": True,
    }
    return {
        "owners": [owner],
        "task_id": owner,
        "implicated_nodes": [owner],
        "rule": GLOBAL_ATTRIBUTION,
        "method": "global_output_attribution",
        "confidence": LOW,
        "evidence": [evidence],
        "anchor": anchor,
        "attribution_candidates": candidates,
        "attribution_basis": selected["reasons"],
        "defect_class": IMPLEMENTATION,
        "error_type": execution_result.get("error_type", "unknown"),
    }


def locate(
    execution_result: Mapping[str, Any],
    ir: Mapping[str, Any],
    local_records: Sequence[Mapping[str, Any]] | None = None,
    owner_map: Mapping[str, Sequence[str]] | None = None,
    versions: Mapping[str, Any] | None = None,
    final_code: str = "",
    code_snippets: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Trace -> Resource -> Local -> GlobalAttribution -> abstention."""

    anchor = anchor_case(execution_result)
    if anchor is None:
        return None
    owner_map = owner_map or build_owner_map(ir, code_snippets, final_code)
    stderr = str(anchor.get("stderr") or execution_result.get("stderr") or "")
    trace, trace_triggered, trace_reason = locate_traceback(stderr, owner_map)
    if trace_triggered:
        if trace is None:
            return {
                "status": "abstained",
                "reason": trace_reason or "unresolved_owner",
                "rule": TRACE,
                "anchor": anchor,
                "error_type": execution_result.get("error_type", "unknown"),
            }
        trace["anchor"] = anchor
        trace["evidence"] = [{
            "kind": "exec",
            "verdict": "reject",
            "owners": trace["owners"],
            "witness": {"case_index": anchor.get("case_index"), "observed": True},
        }]
        trace["implicated_nodes"] = list(trace["owners"])
        trace["defect_class"] = IMPLEMENTATION
        return trace

    symptom = _symptom(anchor.get("symptom"), execution_result)
    if symptom in {TIMEOUT, MEMORY_ERROR}:
        violations = _resource_violations(execution_result, ir)
        if len(violations) == 1:
            owners = next(iter(violations))
            evidence = [
                {
                    "kind": "exec",
                    "verdict": "reject",
                    "owners": [owners],
                    "obligation": {"class": RESOURCE},
                    "witness": {"case_index": anchor.get("case_index"), **measurement},
                }
                for measurement in violations[owners]
            ]
            return {
                "owners": [owners],
                "task_id": owners,
                "implicated_nodes": [owners],
                "rule": RESOURCE_RULE,
                "method": "resource_measurement",
                "confidence": MEDIUM,
                "evidence": evidence,
                "anchor": anchor,
                "defect_class": IMPLEMENTATION,
            }
        if violations:
            return {
                "status": "abstained",
                "reason": "multiple_resource_owners",
                "rule": RESOURCE_RULE,
                "anchor": anchor,
            }

    records = [dict(record) for record in (local_records or []) if isinstance(record, Mapping)]
    compatible: list[dict[str, Any]] = []
    observed: list[dict[str, Any]] = []
    for record in records:
        candidate, is_observed = compatible_record(record, anchor, ir, records, versions)
        if candidate is None:
            continue
        compatible.append(candidate)
        if is_observed:
            observed.append(candidate)
    if not compatible:
        global_location = _locate_by_global_attribution(
            execution_result,
            anchor,
            ir,
            records,
            code_snippets,
        )
        if global_location is not None:
            return global_location
        return {
            "status": "abstained",
            "reason": "no_compatible_evidence",
            "rule": LOCAL,
            "anchor": anchor,
            "error_type": execution_result.get("error_type", "unknown"),
        }
    supporting = observed or compatible
    implicated: set[str] = set()
    for record in supporting:
        implicated |= _owners(record)
    if not coherent_records(supporting, implicated, ir):
        return {
            "status": "abstained",
            "reason": "competing_explanations",
            "rule": LOCAL,
            "anchor": anchor,
            "evidence": supporting,
        }
    grade = MEDIUM if observed else LOW
    location: dict[str, Any] = {
        "owners": sorted(implicated),
        "implicated_nodes": sorted(implicated),
        "task_id": next(iter(sorted(implicated))) if len(implicated) == 1 else None,
        "rule": LOCAL,
        "method": "executed_evidence" if observed else "verification_report",
        "confidence": grade,
        "evidence": supporting,
        "anchor": anchor,
        "defect_class": defect_class(supporting),
        "error_type": execution_result.get("error_type", "unknown"),
    }
    return location


def local_records_from_state(
    state: Mapping[str, Any],
    ir: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Collect serialized evidence records without inventing obligations."""

    explicit = state.get("evidence_records", state.get("local_evidence"))
    if isinstance(explicit, list):
        return [dict(item) for item in explicit if isinstance(item, Mapping)]
    reports = state.get("verification_reports") or {}
    records: list[dict[str, Any]] = []
    for identifier, report in reports.items():
        if not isinstance(report, Mapping):
            continue
        if str(report.get("verdict", "")).lower() not in {"reject", "rejected", "fail", "failed"}:
            continue
        # Prefer explicit paper-shaped evidence.  Legacy verifier reports can
        # still be lifted into a judged obligation when they contain a
        # concrete counterexample, contract violation, or resource signal;
        # the resulting record remains Low because applicability is asserted.
        obligation = report.get("obligation", report.get("obl"))
        witness = report.get("witness", report.get("wit"))
        if obligation is None:
            counterexample = report.get("counterexample")
            medium = _as_dict(report.get("medium_signals"))
            strong = _as_dict(report.get("strong_signals"))
            violations = medium.get("contract_violations") or report.get("contract_violations")
            if violations:
                obligation = {
                    "class": INTERFACE,
                    "name": "contract_signature",
                    "description": violations,
                }
            elif not strong.get("resource_ok", True) or report.get("resource_violation"):
                obligation = {"class": RESOURCE, "name": "resource_budget"}
            elif counterexample or report.get("summary"):
                obligation = {
                    "class": SEMANTIC,
                    "name": "reported_counterexample",
                }
                if witness is None and isinstance(counterexample, Mapping):
                    witness = {
                        "input": counterexample.get("minimal_input"),
                        "expected": counterexample.get("expected_output"),
                        "actual": counterexample.get("actual_output"),
                        "applicability": "asserted",
                    }
        record = {
            **report,
            "owners": report.get("owners", report.get("own", [str(identifier)])),
            "obligation": obligation,
            "witness": witness or {},
            "kind": report.get("kind", report.get("evidence_kind", "judged")),
            "version": report.get("version", report.get("ver")),
        }
        records.append(record)
    return records


def normalize_location(location: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not location or location.get("status") == "abstained":
        return None
    result = dict(location)
    owners = result.get("owners", result.get("implicated_nodes"))
    if owners is None and result.get("task_id"):
        owners = [result["task_id"]]
    if isinstance(owners, str):
        owners = [owners]
    result["owners"] = sorted({str(item) for item in (owners or []) if str(item)})
    result["implicated_nodes"] = list(result["owners"])
    if len(result["owners"]) == 1:
        result.setdefault("task_id", result["owners"][0])
    return result


def _commitment_name(record: Mapping[str, Any]) -> str:
    obligation = _as_dict(_obligation(record))
    return str(obligation.get("commitment", obligation.get("name", record.get("commitment", ""))) or "")


def _scope_commitment_matches(node: Mapping[str, Any], commitment: str, identifier: str) -> bool:
    return bool(commitment) and (
        str(node.get("commitment", "")) == commitment
        or str(node.get("goal", "")) == commitment
        or identifier == commitment
    )


def select_scope(
    location: Mapping[str, Any] | None,
    ir: Mapping[str, Any],
    code_snippets: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Paper Algorithm 3: choose a supported nominal and final region."""

    location = normalize_location(location)
    if location is None:
        return None
    nodes = node_map(ir)
    owners = set(location.get("owners", []))
    if not owners or not owners <= set(nodes):
        return None
    code_owners = code_owner_ids(ir, code_snippets)
    delta = str(location.get("defect_class", IMPLEMENTATION)).lower()
    evidence = [record for record in location.get("evidence", []) if isinstance(record, Mapping)]
    nominal: set[str] | None = None
    nominal_reason = ""
    if len(owners) == 1:
        owner = next(iter(owners))
        if owner in code_owners and delta in {IMPLEMENTATION, "impl", ""}:
            nominal = {owner}
            nominal_reason = "singleton"
    if nominal is None:
        ancestors = hierarchy_ancestors(ir, owners)
        if delta in {COMMITMENT, "cmt"}:
            commitments = {_commitment_name(record) for record in evidence if _commitment_name(record)}
            for ancestor in ancestors:
                node = nodes.get(ancestor, {})
                if len(commitments) == 1 and _scope_commitment_matches(node, next(iter(commitments)), ancestor):
                    nominal = hierarchy_subtree(ir, ancestor)
                    nominal_reason = "commitment_branch"
                    break
        elif delta in {INTERFACE, "ifc"}:
            pairs = [pair for record in evidence for pair in _provider_pairs(record)]
            for ancestor in ancestors:
                subtree = hierarchy_subtree(ir, ancestor)
                if pairs and all(left in subtree and right in subtree for left, right in pairs):
                    nominal = subtree
                    nominal_reason = "interface_branch"
                    break
    if nominal is None:
        return None

    region = set(nominal)
    exceptions: list[dict[str, Any]] = []
    # Approved exceptions are declared by evidence before regeneration.  No
    # post-hoc expansion is performed when a candidate happens to touch a
    # frozen unit.
    for exception in location.get("integration_exceptions", []) or []:
        if not isinstance(exception, Mapping):
            return None
        support = exception.get("evidence", exception.get("supporting_evidence"))
        affected = exception.get("owners", exception.get("affected_owners", exception.get("nodes", [])))
        if not support or not affected:
            return None
        affected_ids = {str(item) for item in affected}
        if exception.get("structural_replanning"):
            expanded: set[str] = set()
            for identifier in affected_ids:
                expanded |= hierarchy_subtree(ir, identifier)
            affected_ids = expanded
        if not affected_ids <= set(nodes):
            return None
        region |= affected_ids
        exceptions.append(dict(exception))

    all_nodes = set(nodes)
    frozen = all_nodes - region
    if region == all_nodes or not (frozen & code_owners):
        return None
    boundary_pairs = []
    for source, consumer in dependency_edges(ir):
        if (source in region) != (consumer in region):
            boundary_pairs.append([source, consumer])
    return {
        "nominal_region": sorted(nominal),
        "region": sorted(region),
        "frozen_region": sorted(frozen),
        "integration_exceptions": exceptions,
        "boundary_dependencies": boundary_pairs,
        "reason": nominal_reason,
        "defect_class": delta,
    }


def dependency_edges(ir: Mapping[str, Any]) -> list[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for identifier, node in node_map(ir).items():
        for dependency in node.get("dependencies", []) or []:
            if isinstance(dependency, Mapping):
                source = dependency.get("source", dependency.get("provider", dependency.get("from")))
            else:
                source = dependency
            if source:
                edges.add((str(source), identifier))
        for slot in node.get("inputs", []) or []:
            source = _as_dict(slot).get("source")
            if source:
                edges.add((str(source).split(".", 1)[0], identifier))
    return sorted(edges)


def rank_execution(
    execution_result: Mapping[str, Any] | None,
    verification_reports: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[int, int, int, int, int]:
    """Compute the paper's lexicographic candidate rank."""

    result = execution_result or {}
    checks = result.get("results") or result.get("checks") or []
    if isinstance(checks, Mapping):
        checks = list(checks.values())
    checks = [item for item in checks if isinstance(item, Mapping)]
    external_pass = sum(bool(item.get("passed", item.get("status") in {PASS, "pass"})) for item in checks)
    external_clean = sum(
        not bool(item.get("timed_out"))
        and int(item.get("exit_code", 0) or 0) == 0
        and not bool(item.get("memory_error"))
        for item in checks
    )
    reports = verification_reports or {}
    internal = [report for report in reports.values() if isinstance(report, Mapping)]
    internal_pass = sum(str(report.get("verdict", "")).lower() in {"accept", "accepted", "pass", "passed", "warning"} for report in internal)
    internal_clean = sum(
        str(report.get("verdict", "")).lower() not in {"runtime_error", "timeout", "memory_error"}
        and not bool(report.get("runtime_error"))
        for report in internal
    )
    all_passed = bool(result.get("all_passed"))
    # ``all_passed`` is the first coordinate exactly as in the paper.  Counts
    # remain useful when both candidates fail.
    return (int(all_passed), external_pass, internal_pass, external_clean, internal_clean)


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def changed_code_units(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[str]:
    return sorted(
        str(identifier)
        for identifier in set(before) | set(after)
        if str(before.get(identifier, "")) != str(after.get(identifier, ""))
    )


def frozen_complement_ok(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    region: Iterable[str],
) -> tuple[bool, list[str]]:
    region = {str(identifier) for identifier in region}
    violations = [
        identifier
        for identifier in sorted(set(before) | set(after))
        if identifier not in region and str(before.get(identifier, "")) != str(after.get(identifier, ""))
    ]
    return not violations, violations


def code_line_delta(before: str, after: str) -> dict[str, Any]:
    before_lines = str(before or "").splitlines()
    after_lines = str(after or "").splitlines()
    matcher = SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed += max(i2 - i1, j2 - j1)
    return {
        "before_lines": len(before_lines),
        "after_lines": len(after_lines),
        "changed_lines": changed,
        "changed_ratio": changed / len(after_lines) if after_lines else 0.0,
    }


def make_snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    """Copy active state needed to restore a rejected transaction."""

    return {
        "final_code": str(state.get("final_code", "")),
        "code_snippets": copy.deepcopy(state.get("code_snippets") or {}),
        "task_plan_ir": copy.deepcopy(state.get("task_plan_ir") or {}),
        "verification_reports": copy.deepcopy(state.get("verification_reports") or {}),
        "execution_result": copy.deepcopy(state.get("execution_result") or {}),
        "audit_versions": copy.deepcopy(state.get("audit_versions") or {}),
        "contract_graph": copy.deepcopy(state.get("contract_graph") or state.get("contract_annotated_task_graph") or {}),
        "contract_annotated_task_graph": copy.deepcopy(state.get("contract_annotated_task_graph") or state.get("contract_graph") or {}),
        "responsibility_records": copy.deepcopy(state.get("responsibility_records") or []),
        "development_checks": copy.deepcopy(state.get("development_checks") or []),
        "registered_units": copy.deepcopy(state.get("registered_units") or {"units": []}),
        "ownership_map": copy.deepcopy(state.get("ownership_map") or {}),
        "evidence_records": copy.deepcopy(state.get("evidence_records") or []),
        "audit_trace": copy.deepcopy(state.get("audit_trace") or {}),
    }


def repair_event(
    *,
    location: Mapping[str, Any] | None,
    scope: Mapping[str, Any] | None,
    before_state: Mapping[str, Any],
    candidate_state: Mapping[str, Any],
    candidate_execution: Mapping[str, Any] | None,
    decision: str,
    rollback_status: str,
    attempt: int,
    reason: str | None = None,
) -> dict[str, Any]:
    before_snippets = before_state.get("code_snippets") or {}
    candidate_snippets = candidate_state.get("code_snippets") or {}
    region = set((scope or {}).get("region") or [])
    frozen = set((scope or {}).get("frozen_region") or [])
    return {
        "attempt": attempt,
        "localization": copy.deepcopy(dict(location or {})),
        "nominal_region": list((scope or {}).get("nominal_region") or []),
        "repair_region": {
            "nodes": sorted(region),
            "region_type": "node" if len(region) == 1 else "branch",
        },
        "frozen_region": sorted(frozen),
        "integration_exceptions": copy.deepcopy((scope or {}).get("integration_exceptions") or []),
        "boundary_dependencies": copy.deepcopy((scope or {}).get("boundary_dependencies") or []),
        "graph_delta": {
            "before_node_ids": sorted(node_map(before_state.get("task_plan_ir") or {})),
            "after_node_ids": sorted(node_map(candidate_state.get("task_plan_ir") or {})),
        },
        "code_delta": {
            "changed_units": changed_code_units(before_snippets, candidate_snippets),
            "frozen_violations": [
                identifier
                for identifier in changed_code_units(before_snippets, candidate_snippets)
                if identifier in frozen
            ],
            "final_code": code_line_delta(
                str(before_state.get("final_code", "")),
                str(candidate_state.get("final_code", "")),
            ),
        },
        "candidate_execution": copy.deepcopy(candidate_execution or {}),
        "rank_before": list(rank_execution(before_state.get("execution_result"), before_state.get("verification_reports"))),
        "rank_after": list(rank_execution(candidate_execution, candidate_state.get("verification_reports"))),
        "decision": decision,
        "rollback_status": rollback_status,
        "reason": reason or "",
    }


__all__ = [
    "COMMITMENT",
    "GLOBAL_ATTRIBUTION",
    "IMPLEMENTATION",
    "INTERFACE",
    "LOCAL",
    "LOW",
    "MEDIUM",
    "HIGH",
    "RESOURCE_RULE",
    "TRACE",
    "anchor_case",
    "build_owner_map",
    "code_owner_ids",
    "coherent_records",
    "compatible_record",
    "defect_class",
    "dependency_edges",
    "frozen_complement_ok",
    "hierarchy_ancestors",
    "hierarchy_subtree",
    "locate",
    "local_records_from_state",
    "_locate_by_global_attribution",
    "make_snapshot",
    "node_map",
    "normalize_location",
    "rank_execution",
    "repair_event",
    "select_scope",
    "sha256_text",
    "traceback_has_generated_frame",
    "_build_interface_map",
    "_locate_by_traceback",
    "_locate_by_verification_reports",
]
