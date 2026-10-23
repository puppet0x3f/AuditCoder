"""Contract-Annotated Task Graph (CATG) support.

The original implementation represented the plan as a flat ``TaskPlanIR`` and
left most of the lifecycle bookkeeping to the final verifier.  The paper's
construction invariant is stronger: responsibility identifiers, contracts,
provenance, development checks, and executor routing have to exist *before*
code is sampled, and the same identifiers must own the resulting code and
evidence.  This module is the small, JSON-friendly adapter that makes that
invariant executable without replacing the public ``TaskPlanIR`` API.

The helpers intentionally accept ordinary dictionaries.  This keeps the
module usable by LangGraph states, experiment scripts, and callers that load a
serialized plan, while the Pydantic models provide a convenient schema for
new code and tests.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections import defaultdict, deque
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, Field, ConfigDict


SCHEMA_VERSION = "catg-1.0"

SEMANTIC = "semantic"
INTERFACE = "interface"
RESOURCE = "resource"
COMMITMENT = "commitment"
EXECUTED = "exec"
JUDGED = "judged"
PASS = "Pass"
REJECT = "Reject"

# The registry is deliberately explicit.  Routing is a table lookup, not a
# model decision made after a failure.  Keep aliases for the historical
# ``orchestration`` and ``assembler`` labels.
DEFAULT_EXECUTOR_REGISTRY: dict[str, str] = {
    "dp": "agent_dp",
    "greedy": "agent_greedy",
    "binary_search": "agent_binary_search",
    "string": "agent_string",
    "segment_tree": "agent_segment_tree",
    "monotonic_queue": "agent_monotonic_queue",
    "graph_bfs": "agent_graph_bfs",
    "graph_dfs": "agent_graph_dfs",
    "union_find": "agent_union_find",
    "math": "agent_math",
    "sorting": "agent_sorting",
    "two_pointer": "agent_two_pointer",
    "orchestration": "assembler",
    "assembler": "assembler",
}


class CATGNode(BaseModel):
    """Serialized responsibility node used by the CATG adapter."""

    model_config = ConfigDict(extra="allow")

    id: str
    goal: str = ""
    commitment: str = ""
    executor_category: str = ""
    executor: str = ""
    parent: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    inputs: list[dict[str, Any]] = Field(default_factory=list)
    outputs: list[dict[str, Any]] = Field(default_factory=list)
    contract: dict[str, Any] = Field(default_factory=dict)
    complexity_budget: Any = ""
    local_checks: list[str] = Field(default_factory=list)
    code_owner: bool = False
    version: int = 0


class EvidenceRecord(BaseModel):
    """Versioned local/global evidence record (``e`` in the paper)."""

    model_config = ConfigDict(extra="allow")

    id: str = ""
    owners: list[str] = Field(default_factory=list)
    version: dict[str, int] = Field(default_factory=dict)
    verdict: str = ""
    obligation: Any = None
    witness: dict[str, Any] = Field(default_factory=dict)
    kind: str = JUDGED
    current: bool = True


class ResponsibilityRecord(BaseModel):
    """Persistent lifecycle record for one responsibility identity."""

    model_config = ConfigDict(extra="allow")

    id: str
    goal: str = ""
    commitment: str = ""
    executor_category: str = ""
    executor: str = ""
    interface: dict[str, Any] = Field(default_factory=dict)
    complexity_budget: Any = ""
    provenance: dict[str, Any] = Field(default_factory=dict)
    owned_code: str = ""
    owned_units: list[str] = Field(default_factory=list)
    validation: list[dict[str, Any]] = Field(default_factory=list)
    integration_exceptions: list[dict[str, Any]] = Field(default_factory=list)
    repair_history: list[dict[str, Any]] = Field(default_factory=list)
    version: int = 0


class RepairEvent(BaseModel):
    """Append-only repair event shape shared with the final verifier."""

    model_config = ConfigDict(extra="allow")

    attempt: int = 0
    localization: dict[str, Any] = Field(default_factory=dict)
    nominal_region: list[str] = Field(default_factory=list)
    region: list[str] = Field(default_factory=list)
    frozen_region: list[str] = Field(default_factory=list)
    integration_exceptions: list[dict[str, Any]] = Field(default_factory=list)
    graph_delta: dict[str, Any] = Field(default_factory=dict)
    code_delta: dict[str, Any] = Field(default_factory=dict)
    candidate_execution: dict[str, Any] = Field(default_factory=dict)
    decision: str = "Reject"
    rollback_status: str = "Restored"


class ContractAnnotatedTaskGraph(BaseModel):
    """Complete serialized CATG.

    ``nodes`` stores the graph itself; ``responsibility_records`` stores the
    lifecycle address attached to every node.  The duplicated index is useful
    for JSON consumers and avoids repeated linear scans in the repair loop.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: str = SCHEMA_VERSION
    root_id: str = ""
    nodes: list[CATGNode] = Field(default_factory=list)
    decomposition_edges: list[dict[str, str]] = Field(default_factory=list)
    dependency_edges: list[dict[str, str]] = Field(default_factory=list)
    topological_order: list[str] = Field(default_factory=list)
    provenance_graph: dict[str, list[str]] = Field(default_factory=dict)
    development_checks: list[dict[str, Any]] = Field(default_factory=list)
    executor_registry: dict[str, str] = Field(default_factory=dict)
    code_owner_ids: list[str] = Field(default_factory=list)
    responsibility_records: list[ResponsibilityRecord] = Field(default_factory=list)
    ownership_map: dict[str, Any] = Field(default_factory=lambda: {"units": []})


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _node_list(ir: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in (ir.get("task_nodes") or []) if isinstance(item, Mapping)]


def _id(node: Mapping[str, Any]) -> str:
    return str(node.get("id") or "")


def _interface(node: Mapping[str, Any]) -> dict[str, Any]:
    raw = node.get("interface") or node.get("contract") or {}
    if not isinstance(raw, Mapping):
        return {}
    result = dict(raw)
    # These names make the serialized contract self-describing while keeping
    # compatibility with the existing FunctionInterface fields.
    result.setdefault("params", [])
    result.setdefault("preconditions", [])
    result.setdefault("postconditions", [])
    result.setdefault("invariants", [])
    result.setdefault("function_name", "")
    result.setdefault("return_type", "")
    return result


def _slots(node: Mapping[str, Any], field: str) -> list[dict[str, Any]]:
    raw = node.get(field) or []
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _executor_category(node: Mapping[str, Any]) -> str:
    if node.get("executor_category"):
        return str(node["executor_category"])
    if node.get("agent_type"):
        return str(node["agent_type"])
    return "orchestration"


def _commitment(node: Mapping[str, Any]) -> str:
    return str(node.get("commitment") or node.get("goal") or _id(node))


def _normalise_checks(ir: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Materialize ordered development checks ``Q`` from validation fields."""

    validation = _mapping(ir.get("validation"))
    checks: list[dict[str, Any]] = []
    for kind, field in (("sample", "sample_test_cases"), ("edge", "edge_cases")):
        values = validation.get(field) or []
        if not isinstance(values, list):
            continue
        for index, raw in enumerate(values):
            if not isinstance(raw, Mapping):
                continue
            expected = raw.get("expected_output", raw.get("output", ""))
            checks.append(
                {
                    "id": f"{kind}:{index}",
                    "order": len(checks),
                    "kind": kind,
                    "input": str(raw.get("input", "")),
                    "expected_output": str(expected),
                    "description": str(raw.get("description", "")),
                    "held_out": False,
                    "source": "task_plan_ir.validation." + field,
                }
            )
    # Do not silently create a fake oracle.  A plan can be checked structurally
    # even when its development Q is empty.
    return checks


def _dependency_edges(nodes: list[dict[str, Any]]) -> list[tuple[str, str]]:
    known = {_id(node) for node in nodes}
    edges: set[tuple[str, str]] = set()
    for node in nodes:
        target = _id(node)
        if not target:
            continue
        for dep in node.get("dependencies") or []:
            if isinstance(dep, Mapping):
                source = dep.get("source", dep.get("from", dep.get("provider")))
            else:
                source = dep
            source = str(source or "")
            if source in known and source != target:
                edges.add((source, target))
        for slot in _slots(node, "inputs"):
            source = str(slot.get("source") or "")
            if "." in source:
                source_id = source.split(".", 1)[0]
                if source_id in known and source_id != target:
                    edges.add((source_id, target))
    return sorted(edges)


def _topological_order(node_ids: Iterable[str], edges: Iterable[tuple[str, str]]) -> tuple[list[str], list[str]]:
    ids = sorted({str(item) for item in node_ids if str(item)})
    adjacency: dict[str, set[str]] = {item: set() for item in ids}
    indegree = {item: 0 for item in ids}
    for source, target in edges:
        if source not in adjacency or target not in adjacency or target in adjacency[source]:
            continue
        adjacency[source].add(target)
        indegree[target] += 1
    queue = deque(sorted(item for item, degree in indegree.items() if degree == 0))
    order: list[str] = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for target in sorted(adjacency[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    cycle_nodes = sorted(set(ids) - set(order))
    # Preserve all IDs in the serialized graph even for a malformed model
    # output; validation reports the cycle and callers can abstain.
    return order + cycle_nodes, cycle_nodes


def _provenance_graph(nodes: list[dict[str, Any]]) -> dict[str, list[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for node in nodes:
        identifier = _id(node)
        inputs = _slots(node, "inputs")
        outputs = _slots(node, "outputs")
        output_keys = [f"{identifier}.{slot.get('name')}" for slot in outputs if slot.get("name")]
        for slot in inputs:
            input_name = slot.get("name")
            if not input_name:
                continue
            input_key = f"{identifier}.{input_name}"
            source = str(slot.get("source") or "")
            if "." in source:
                graph[source].add(input_key)
            for output_key in output_keys:
                graph[input_key].add(output_key)
        for output_key in output_keys:
            graph.setdefault(output_key, set())
    return {key: sorted(values) for key, values in sorted(graph.items())}


def _obligations(node: Mapping[str, Any], interface: Mapping[str, Any]) -> list[dict[str, Any]]:
    identifier = _id(node)
    result: list[dict[str, Any]] = []
    for index, text in enumerate(interface.get("postconditions") or []):
        result.append({"id": f"{identifier}:post:{index}", "class": SEMANTIC, "description": str(text), "owner": identifier})
    for index, text in enumerate(interface.get("invariants") or []):
        result.append({"id": f"{identifier}:inv:{index}", "class": SEMANTIC, "description": str(text), "owner": identifier})
    for index, text in enumerate(interface.get("preconditions") or []):
        result.append({"id": f"{identifier}:pre:{index}", "class": INTERFACE, "description": str(text), "owner": identifier})
    if interface.get("function_name"):
        result.append({"id": f"{identifier}:signature", "class": INTERFACE, "description": "function name and signature", "owner": identifier})
    for slot in _slots(node, "inputs"):
        source = str(slot.get("source") or "")
        if "." in source:
            provider = source.split(".", 1)[0]
            result.append({
                "id": f"{identifier}:provenance:{slot.get('name')}",
                "class": INTERFACE,
                "description": f"{provider} supplies {identifier}.{slot.get('name')}",
                "provider_pair": [provider, identifier],
                "owner": identifier,
            })
    complexity = node.get("complexity_budget")
    if complexity:
        result.append({"id": f"{identifier}:resource", "class": RESOURCE, "description": complexity, "owner": identifier})
    result.append({"id": f"{identifier}:commitment", "class": COMMITMENT, "description": _commitment(node), "commitment": _commitment(node), "owner": identifier})
    return result


def _record_for_node(
    node: Mapping[str, Any],
    *,
    version: int = 0,
    executor_registry: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    identifier = _id(node)
    interface = _interface(node)
    source_map = {
        str(slot.get("name")): slot.get("source")
        for slot in _slots(node, "inputs")
        if slot.get("name")
    }
    contract = {
        "name": interface.get("function_name", ""),
        "args": interface.get("params", []),
        "ret": interface.get("return_type", ""),
        "pre": interface.get("preconditions", []),
        "post": interface.get("postconditions", []),
        "inv": interface.get("invariants", []),
        "obligations": _obligations(node, interface),
    }
    owner = bool(node.get("owns_code") or node.get("owned_code") or node.get("is_code_owner") or (node.get("is_leaf") and node.get("agent_type")))
    registry = executor_registry or DEFAULT_EXECUTOR_REGISTRY
    return {
        "id": identifier,
        "goal": str(node.get("goal") or ""),
        "commitment": _commitment(node),
        "executor_category": _executor_category(node),
        "executor": registry.get(_executor_category(node), ""),
        "interface": interface,
        "contract": contract,
        "complexity_budget": copy.deepcopy(node.get("complexity_budget", "")),
        "provenance": {"inputs": source_map, "graph_keys": [f"{identifier}.{name}" for name in source_map]},
        "owned_code": str(node.get("owned_code") or "") if isinstance(node.get("owned_code"), str) else "",
        "owned_units": list(node.get("owned_units") or []) if isinstance(node.get("owned_units"), list) else [],
        "validation": [],
        "integration_exceptions": list(node.get("integration_exceptions") or []) if isinstance(node.get("integration_exceptions"), list) else [],
        "repair_history": list(node.get("repair_history") or []) if isinstance(node.get("repair_history"), list) else [],
        "version": int(node.get("version", version) or version),
        "code_owner": owner,
    }


def _prior_responsibility_records(ir: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Read lifecycle records already attached to an IR without losing history."""

    raw: Any = ir.get("responsibility_records")
    if not raw:
        raw = _mapping(ir.get("contract_annotated_task_graph")).get("responsibility_records")
    if isinstance(raw, Mapping):
        raw = list(raw.values())
    result: dict[str, dict[str, Any]] = {}
    for item in raw or []:
        if isinstance(item, Mapping) and item.get("id"):
            result[str(item["id"])] = copy.deepcopy(dict(item))
    return result


def _merge_responsibility_record(
    fresh: Mapping[str, Any],
    prior: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Combine current CATG metadata with append-only lifecycle state."""

    merged = copy.deepcopy(dict(fresh))
    if not prior:
        return merged
    old = copy.deepcopy(dict(prior))
    # These fields are state accumulated after planning.  Keep them when the
    # CATG is rebuilt at another pipeline boundary; otherwise every new leaf
    # registration would erase evidence for the leaves registered earlier.
    if old.get("owned_code"):
        merged["owned_code"] = old["owned_code"]
    if old.get("owned_units"):
        merged["owned_units"] = sorted(set(merged.get("owned_units") or []) | set(old["owned_units"] or []))
    if old.get("validation"):
        merged["validation"] = old["validation"]
    for field in ("integration_exceptions", "repair_history"):
        if old.get(field):
            merged[field] = old[field]
    if "version" in old:
        merged["version"] = int(old.get("version", 0) or 0)
    if old.get("code_owner"):
        merged["code_owner"] = True
    # Preserve extension fields used by downstream audit consumers while
    # allowing the current CATG to refresh its structural metadata.
    for key, value in old.items():
        merged.setdefault(key, value)
    return merged


def build_contract_annotated_task_graph(
    task_plan_ir: Mapping[str, Any],
    *,
    ownership_map: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a ``TaskPlanIR`` into the paper's CATG representation.

    The function is deterministic and side-effect free.  ``ensure_catg`` is
    the mutating convenience wrapper used at pipeline boundaries.
    """

    nodes = _node_list(task_plan_ir)
    nodes_with_ids = [node for node in nodes if _id(node)]
    node_ids = [_id(node) for node in nodes_with_ids]
    node_by_id = {_id(node): node for node in nodes_with_ids}
    root = str(task_plan_ir.get("root_id") or "")
    if root not in node_by_id:
        root = next((identifier for identifier, node in node_by_id.items() if not node.get("parent")), "")
    dependencies = _dependency_edges(nodes)
    topological, cycle_nodes = _topological_order(node_ids, dependencies)
    decomposition = [
        {"parent": str(node.get("parent")), "child": identifier}
        for identifier, node in node_by_id.items()
        if node.get("parent") and str(node.get("parent")) in node_by_id
    ]
    registry = dict(DEFAULT_EXECUTOR_REGISTRY)
    custom_registry = task_plan_ir.get("executor_registry")
    if isinstance(custom_registry, Mapping):
        registry.update({str(k): str(v) for k, v in custom_registry.items()})
    prior_records = _prior_responsibility_records(task_plan_ir)
    records = [
        _merge_responsibility_record(
            _record_for_node(node, executor_registry=registry),
            prior_records.get(_id(node)),
        )
        for node in nodes
        if _id(node)
    ]
    code_owner_ids = sorted(
        {
            identifier
            for identifier, node in node_by_id.items()
            if node.get("owns_code")
            or node.get("owned_code")
            or node.get("is_code_owner")
            or (node.get("is_leaf") and node.get("agent_type"))
        }
        | {
            str(record["id"])
            for record in records
            if record.get("code_owner") or record.get("owned_code")
        }
    )
    units = []
    if isinstance(ownership_map, Mapping):
        raw_units = ownership_map.get("units")
        if isinstance(raw_units, list):
            units = copy.deepcopy(raw_units)
    checks = _normalise_checks(task_plan_ir)
    # Associate the fixed Q with every record as a validation reference.  Keep
    # local verifier reports and prior repair evidence already stored on the
    # record; rebuilding CATG must never erase that lifecycle history.
    q_refs = [{"check_id": check["id"], "scope": "global"} for check in checks]
    for record in records:
        validation = list(record.get("validation") or [])
        known_checks = {
            str(item.get("check_id"))
            for item in validation
            if isinstance(item, Mapping) and item.get("check_id")
        }
        record["validation"] = validation + [
            item for item in q_refs if item["check_id"] not in known_checks
        ]
    graph = {
        "schema_version": SCHEMA_VERSION,
        "root_id": root,
        "nodes": [
            {
                **node,
                "commitment": _commitment(node),
                "executor_category": _executor_category(node),
                "executor": registry.get(_executor_category(node), ""),
                "contract": {
                    **_interface(node),
                    "obligations": _obligations(node, _interface(node)),
                },
                "code_owner": _id(node) in code_owner_ids,
                "version": int(node.get("version", 0) or 0),
            }
            for node in nodes_with_ids
        ],
        "decomposition_edges": decomposition,
        "dependency_edges": [{"source": source, "target": target} for source, target in dependencies],
        "topological_order": topological,
        "dependency_cycle_nodes": cycle_nodes,
        "provenance_graph": _provenance_graph(nodes),
        "development_checks": checks,
        "executor_registry": registry,
        "code_owner_ids": code_owner_ids,
        "responsibility_records": records,
        "responsibility_index": {record["id"]: copy.deepcopy(record) for record in records},
        "ownership_map": {"units": units},
    }
    graph["validation_errors"] = validate_contract_graph(graph)
    return graph


def ensure_catg(
    task_plan_ir: Mapping[str, Any],
    *,
    ownership_map: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach a fresh CATG to a plan while preserving legacy IR fields."""

    result = dict(task_plan_ir)
    graph = build_contract_annotated_task_graph(
        result,
        ownership_map=(ownership_map if ownership_map is not None else result.get("ownership_map")),
    )
    result["contract_annotated_task_graph"] = graph
    result["contract_graph"] = graph
    result["provenance_graph"] = graph["provenance_graph"]
    result["development_checks"] = graph["development_checks"]
    result["executor_registry"] = graph["executor_registry"]
    result["responsibility_records"] = copy.deepcopy(graph["responsibility_records"])
    result["ownership_map"] = copy.deepcopy(graph["ownership_map"])
    result["catg_validation_errors"] = list(graph.get("validation_errors", []))
    # Make the pre-code commitment and routing visible on TaskNode dictionaries
    # as well.  Existing consumers ignore the additional fields.
    node_index = {str(node.get("id")): node for node in result.get("task_nodes", []) if isinstance(node, dict)}
    for catg_node in graph["nodes"]:
        target = node_index.get(str(catg_node.get("id")))
        if target is None:
            continue
        if not target.get("commitment"):
            target["commitment"] = catg_node.get("commitment", "")
        if not target.get("executor_category"):
            target["executor_category"] = catg_node.get("executor_category", "")
        if not target.get("contract"):
            target["contract"] = catg_node.get("contract", {})
        target.setdefault("version", catg_node.get("version", 0))
        if target.get("owns_code") is None:
            target["owns_code"] = catg_node.get("code_owner", False)
    return result


def normalize_task_plan_ir(task_plan_ir: Mapping[str, Any]) -> dict[str, Any]:
    """Repair only mechanical IR defects before CATG allocation.

    LLM planning occasionally omits a parent, interface array, or execution
    order even when the intended decomposition is clear.  This pass never
    changes a non-empty goal, algorithm choice, contract field, or data-flow
    source.  It supplies deterministic defaults so downstream agents receive
    a usable contract and the CATG validator can report the remaining
    semantic issues instead of failing with a ``KeyError``.
    """

    result = copy.deepcopy(dict(task_plan_ir))
    raw_nodes = result.get("task_nodes") or []
    nodes = [dict(item) for item in raw_nodes if isinstance(item, Mapping)]
    seen: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        identifier = str(node.get("id") or f"S{index}")
        if identifier in seen:
            suffix = 2
            original = identifier
            while f"{original}_{suffix}" in seen:
                suffix += 1
            identifier = f"{original}_{suffix}"
        seen.add(identifier)
        node["id"] = identifier
        node.setdefault("goal", "")
        node.setdefault("type", "orchestration" if not node.get("agent_type") else node.get("agent_type"))
        node.setdefault("is_leaf", bool(node.get("agent_type")))
        node.setdefault("dependencies", [])
        node.setdefault("inputs", [])
        node.setdefault("outputs", [])
        node.setdefault("hints", [])
        node.setdefault("context", "")
        node.setdefault("complexity_budget", "")
        cleaned.append(node)
    result["task_nodes"] = cleaned
    node_ids = {str(node["id"]) for node in cleaned}
    root = str(result.get("root_id") or "")
    if root not in node_ids:
        root = next((str(node["id"]) for node in cleaned if not node.get("parent")), "")
    if not root and cleaned:
        root = str(cleaned[0]["id"])
    result["root_id"] = root
    for node in cleaned:
        identifier = str(node["id"])
        if identifier == root:
            node["parent"] = None
        else:
            parent = str(node.get("parent") or "")
            node["parent"] = parent if parent in node_ids and parent != identifier else root
        deps: list[str] = []
        for raw_dep in node.get("dependencies") or []:
            dep = raw_dep.get("source", raw_dep.get("from", raw_dep.get("provider"))) if isinstance(raw_dep, Mapping) else raw_dep
            dep = str(dep or "")
            if dep in node_ids and dep != identifier and dep not in deps:
                deps.append(dep)
        node["dependencies"] = deps
        if not isinstance(node.get("inputs"), list):
            node["inputs"] = []
        if not isinstance(node.get("outputs"), list):
            node["outputs"] = []
        # A missing leaf interface is a mechanical omission.  The generated
        # name is stable and the real semantic contract remains visible in
        # goal/hints for the algorithm agent.
        if node.get("is_leaf") and not isinstance(node.get("interface"), Mapping):
            safe = re.sub(r"[^A-Za-z0-9_]", "_", identifier).strip("_") or "task"
            params = [
                {
                    "name": str(slot.get("name") or f"arg_{i}"),
                    "type": str(slot.get("type") or "Any"),
                    "description": str(slot.get("description") or ""),
                }
                for i, slot in enumerate(node.get("inputs") or [])
                if isinstance(slot, Mapping)
            ]
            outputs = [slot for slot in (node.get("outputs") or []) if isinstance(slot, Mapping)]
            node["interface"] = {
                "function_name": f"solve_{safe}",
                "params": params,
                "return_type": str(outputs[0].get("type") or "Any") if outputs else "Any",
                "return_description": str(outputs[0].get("description") or "result") if outputs else "result",
                "preconditions": [],
                "postconditions": [],
                "invariants": [],
                "generated_default": True,
            }
    graph = build_contract_annotated_task_graph(result)
    leaves = {str(node.get("id")) for node in cleaned if node.get("is_leaf")}
    supplied_order = [str(item) for item in result.get("execution_order") or []]
    result["execution_order"] = [item for item in supplied_order if item in leaves]
    result["execution_order"] += [item for item in graph["topological_order"] if item in leaves and item not in result["execution_order"]]
    return result


def _record_index(ir: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = ir.get("responsibility_records") or _mapping(ir.get("contract_annotated_task_graph")).get("responsibility_records") or []
    result: dict[str, dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        raw = list(raw.values())
    for item in raw:
        if isinstance(item, Mapping) and item.get("id"):
            result[str(item["id"])] = dict(item)
    return result


def register_code_owner(
    task_plan_ir: dict[str, Any],
    task_id: str,
    source_code: str,
    *,
    unit_names: Iterable[str] | None = None,
    verification_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind one generated bundle and its local validation to its owner ID."""

    result = ensure_catg(task_plan_ir)
    identifier = str(task_id)
    code = str(source_code or "")
    names = [str(name) for name in (unit_names or _function_names(code)) if str(name)]
    if not names:
        node = next((n for n in result.get("task_nodes", []) if str(n.get("id")) == identifier), {})
        name = _interface(node).get("function_name")
        if name:
            names = [str(name)]
    records = _record_index(result)
    record = records.setdefault(identifier, {"id": identifier})
    record.update({"owned_code": code, "owned_units": sorted(set(names)), "code_owner": True})
    if verification_report is not None:
        report = dict(verification_report)
        record.setdefault("validation", [])
        record["validation"].append({"kind": "local_verification", "report": report})
    record["version"] = int(record.get("version", 0) or 0)
    result["responsibility_records"] = [records[key] for key in sorted(records)]
    result["ownership_map"] = build_ownership_map(result, {identifier: code}, base=result.get("ownership_map"))
    # Store selected code on the node for serialized DCTC and audit consumers.
    for node in result.get("task_nodes", []):
        if isinstance(node, dict) and str(node.get("id")) == identifier:
            node["owned_code"] = code
            node["owned_units"] = sorted(set(names))
            node["owns_code"] = True
            node["version"] = record["version"]
    graph = build_contract_annotated_task_graph(result, ownership_map=result.get("ownership_map"))
    graph["responsibility_records"] = copy.deepcopy(result["responsibility_records"])
    graph["responsibility_index"] = {r["id"]: copy.deepcopy(r) for r in result["responsibility_records"]}
    result["contract_annotated_task_graph"] = graph
    result["contract_graph"] = graph
    return result


def register_assembly_units(
    task_plan_ir: dict[str, Any],
    final_code: str,
    *,
    owner_id: str | None = None,
) -> dict[str, Any]:
    """Register orchestration units emitted by the assembler.

    Leaf bundles stay attached to their leaf IDs.  Only the top-level
    orchestration entry point is assigned to the root (or an explicitly
    declared orchestration owner), preventing a helper from acquiring two
    owners merely because it appears in the assembled source.
    """

    result = ensure_catg(task_plan_ir)
    identifier = str(owner_id or result.get("root_id") or "")
    if not identifier:
        return result
    names: list[str] = []
    for candidate in ("solve", "main", "__main__"):
        if re.search(rf"^\s*def\s+{re.escape(candidate)}\s*\(", final_code or "", re.MULTILINE):
            names.append(candidate)
    if "if __name__" in (final_code or ""):
        names.append("__main__")
    records = _record_index(result)
    record = records.setdefault(identifier, {"id": identifier})
    record.update({"owned_units": sorted(set(record.get("owned_units") or []) | set(names)), "orchestration_code_hash": hashlib.sha256(str(final_code or "").encode("utf-8")).hexdigest()})
    record["code_owner"] = bool(names)
    result["responsibility_records"] = [records[key] for key in sorted(records)]
    ownership = build_ownership_map(result, {}, base=result.get("ownership_map"))
    units = {str(item.get("name")): dict(item) for item in ownership.get("units", []) if isinstance(item, Mapping) and item.get("name")}
    for name in names:
        units.setdefault(name, {"name": name, "owner": identifier, "task_id": identifier})
    ownership["units"] = [units[name] for name in sorted(units)]
    result["ownership_map"] = ownership
    for node in result.get("task_nodes", []):
        if isinstance(node, dict) and str(node.get("id")) == identifier:
            node["owned_units"] = sorted(set(node.get("owned_units") or []) | set(names))
            node["owns_code"] = bool(names) or bool(node.get("owns_code"))
    graph = build_contract_annotated_task_graph(result, ownership_map=result.get("ownership_map"))
    graph["responsibility_records"] = copy.deepcopy(result["responsibility_records"])
    graph["responsibility_index"] = {r["id"]: copy.deepcopy(r) for r in result["responsibility_records"]}
    result["contract_annotated_task_graph"] = graph
    result["contract_graph"] = graph
    return result


def _function_names(source: str) -> list[str]:
    # Avoid importing the AST on every generated unit; this conservative regex
    # covers Python function and async-function declarations.
    return sorted(set(re.findall(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", source or "", re.MULTILINE)))


def build_ownership_map(
    ir: Mapping[str, Any],
    code_snippets: Mapping[str, str] | None = None,
    *,
    base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a total unit registry with one owner per registered unit."""

    units: list[dict[str, Any]] = []
    raw_units = _mapping(base).get("units") if isinstance(base, Mapping) else None
    if isinstance(raw_units, list):
        units.extend(copy.deepcopy([u for u in raw_units if isinstance(u, Mapping)]))
    by_name: dict[str, dict[str, Any]] = {}
    for unit in units:
        name = str(unit.get("name") or unit.get("function_name") or "")
        if name:
            by_name[name] = unit
    records = _record_index(ir)
    snippets = code_snippets or {}
    for identifier, source in snippets.items():
        names = records.get(str(identifier), {}).get("owned_units") or _function_names(str(source))
        for name in names:
            rendered = str(name)
            if not rendered:
                continue
            # Re-registering the same name for the same owner is idempotent;
            # conflicting owners remain explicit and are rejected by validation.
            entry = by_name.get(rendered)
            if entry is None:
                entry = {"name": rendered, "owner": str(identifier), "task_id": str(identifier)}
                by_name[rendered] = entry
            elif str(entry.get("owner", entry.get("task_id", ""))) != str(identifier):
                owners = entry.setdefault("owners", [entry.get("owner", entry.get("task_id"))])
                if str(identifier) not in owners:
                    owners.append(str(identifier))
    return {"units": [by_name[name] for name in sorted(by_name)]}


def record_local_evidence(
    task_plan_ir: dict[str, Any],
    task_id: str,
    report: Mapping[str, Any],
    *,
    version: int | None = None,
) -> list[dict[str, Any]]:
    """Lift a verifier report into paper-shaped evidence records.

    Verifier reports do not always include a runnable witness.  Such records
    are explicitly marked ``judged`` and therefore receive Low evidence by
    the final locator; this is safer than pretending an LLM assertion was an
    execution trace.
    """

    identifier = str(task_id)
    report = dict(report or {})
    verdict_text = str(report.get("verdict", "")).lower()
    verdict = PASS if verdict_text in {"accept", "accepted", "pass", "passed", "warning"} else REJECT
    medium = _mapping(report.get("medium_signals"))
    strong = _mapping(report.get("strong_signals"))
    obligation = report.get("obligation", report.get("obl"))
    if obligation is None:
        violations = medium.get("contract_violations") or report.get("contract_violations")
        if violations:
            obligation = {"class": INTERFACE, "name": "contract_signature", "description": violations}
        elif medium.get("resource_violation") or strong.get("resource_ok") is False or report.get("resource_violation"):
            obligation = {"class": RESOURCE, "name": "resource_budget"}
        else:
            obligation = {"class": SEMANTIC, "name": "reported_counterexample"}
    counterexample = report.get("counterexample")
    witness = report.get("witness", report.get("wit"))
    if not isinstance(witness, Mapping):
        witness = {}
    witness = dict(witness)
    if isinstance(counterexample, Mapping):
        witness.setdefault("input", counterexample.get("minimal_input"))
        witness.setdefault("expected", counterexample.get("expected_output"))
        witness.setdefault("actual", counterexample.get("actual_output"))
    witness.setdefault("applicability", "asserted")
    report_version = int(version if version is not None else 0)
    record = {
        "id": f"local:{identifier}:{len(_mapping(task_plan_ir).get('evidence_records') or [])}",
        "owners": [identifier],
        "own": [identifier],
        "version": {identifier: report_version},
        "verdict": verdict,
        "vd": verdict,
        "obligation": obligation,
        "obl": obligation,
        "witness": witness,
        "wit": witness,
        "kind": str(report.get("evidence_kind", report.get("kind", JUDGED))).lower(),
        "source": "verifier_agent",
        "current": True,
        "summary": str(report.get("summary", "")),
    }
    records = [dict(item) for item in (task_plan_ir.get("evidence_records") or []) if isinstance(item, Mapping)]
    # Keep the append-only evidence trace required by the paper.  A retry may
    # supersede a previous local report, but deleting that report would make
    # the audit history and provenance graph incomplete.  Mark old records as
    # historical while leaving the newest record current for lookup code.
    for item in records:
        if item.get("source") == "verifier_agent" and item.get("owners") == [identifier] and item.get("current", True):
            item["current"] = False
            item["superseded_by"] = record["id"]
    records.append(record)
    task_plan_ir["evidence_records"] = records
    idx = _record_index(task_plan_ir)
    owner_record = idx.setdefault(identifier, {"id": identifier})
    validation = list(owner_record.get("validation") or [])
    validation.append(record)
    owner_record["validation"] = validation
    task_plan_ir["responsibility_records"] = [idx[key] for key in sorted(idx)]
    task_plan_ir["contract_annotated_task_graph"] = build_contract_annotated_task_graph(task_plan_ir, ownership_map=task_plan_ir.get("ownership_map"))
    task_plan_ir["contract_annotated_task_graph"]["responsibility_records"] = copy.deepcopy(task_plan_ir["responsibility_records"])
    task_plan_ir["contract_annotated_task_graph"]["responsibility_index"] = {r["id"]: copy.deepcopy(r) for r in task_plan_ir["responsibility_records"]}
    task_plan_ir["contract_graph"] = task_plan_ir["contract_annotated_task_graph"]
    return records


def validate_contract_graph(graph_or_ir: Mapping[str, Any]) -> list[str]:
    """Return deterministic CATG integrity errors; no model judgment involved."""

    raw = graph_or_ir.get("contract_annotated_task_graph") if isinstance(graph_or_ir, Mapping) else None
    graph = _mapping(raw) if isinstance(raw, Mapping) else dict(graph_or_ir)
    nodes_raw = graph.get("nodes") or []
    nodes = [dict(node) for node in nodes_raw if isinstance(node, Mapping)]
    errors: list[str] = []
    ids = [str(node.get("id") or "") for node in nodes]
    if any(not identifier for identifier in ids):
        errors.append("CATG contains a node without an id")
    if len(ids) != len(set(ids)):
        errors.append("CATG responsibility IDs are not unique")
    known = set(ids)
    node_by_id = {identifier: node for identifier, node in zip(ids, nodes) if identifier}
    root = str(graph.get("root_id") or "")
    if root not in known:
        errors.append(f"CATG root_id {root!r} is missing")
    roots = [identifier for identifier in ids if identifier and not node_by_id.get(identifier, {}).get("parent")]
    if len(roots) != 1:
        errors.append(f"CATG decomposition relation has {len(roots)} roots")
    parent_of = {
        identifier: str(node.get("parent"))
        for identifier, node in node_by_id.items()
        if node.get("parent")
    }
    for identifier, parent in parent_of.items():
        if parent not in known:
            errors.append(f"CATG node {identifier} references unknown parent {parent}")
    for identifier in ids:
        seen: set[str] = set()
        current = identifier
        while current in parent_of:
            if current in seen:
                errors.append(f"CATG decomposition relation contains a cycle at {identifier}")
                break
            seen.add(current)
            current = parent_of[current]
    dep_edges: list[tuple[str, str]] = []
    for raw_edge in graph.get("dependency_edges") or []:
        edge = _mapping(raw_edge)
        source, target = str(edge.get("source") or edge.get("from") or ""), str(edge.get("target") or edge.get("to") or "")
        if source not in known or target not in known:
            errors.append(f"CATG dependency edge references unknown node: {source}->{target}")
        elif source == target:
            errors.append(f"CATG dependency self-loop: {source}")
        else:
            dep_edges.append((source, target))
    _, cycle_nodes = _topological_order(known, dep_edges)
    if cycle_nodes:
        errors.append("CATG dependency relation contains a cycle: " + ", ".join(cycle_nodes))
    order = [str(item) for item in graph.get("topological_order") or []]
    if set(order) != known:
        errors.append("CATG topological_order does not cover exactly all nodes")
    position = {identifier: index for index, identifier in enumerate(order)}
    for source, target in dep_edges:
        if source in position and target in position and position[source] > position[target]:
            errors.append(f"CATG topological_order violates {source}->{target}")
    registry = _mapping(graph.get("executor_registry"))
    for node in nodes:
        category = str(node.get("executor_category") or "")
        if category and category not in registry:
            errors.append(f"CATG node {node.get('id')} uses unregistered executor {category}")
    for raw_unit in (_mapping(graph.get("ownership_map")).get("units") or []):
        unit = _mapping(raw_unit)
        owners = unit.get("owners")
        if owners is None:
            owners = [unit.get("owner", unit.get("task_id"))]
        owners = [str(item) for item in owners if str(item)]
        if len(set(owners)) != 1:
            errors.append(f"CATG unit {unit.get('name', '')} does not have exactly one owner")
        elif owners[0] not in known:
            errors.append(f"CATG unit {unit.get('name', '')} has unknown owner {owners[0]}")
    return list(dict.fromkeys(errors))


def route_executor(task_node: Mapping[str, Any], registry: Mapping[str, str] | None = None) -> str | None:
    """Resolve an executor category before generation; ``None`` means reject."""

    table = dict(DEFAULT_EXECUTOR_REGISTRY)
    if isinstance(registry, Mapping):
        table.update({str(k): str(v) for k, v in registry.items()})
    return table.get(_executor_category(task_node))


def catg_summary(task_plan_ir: Mapping[str, Any]) -> dict[str, Any]:
    """Compact diagnostic used by smoke tests and final reports."""

    graph = _mapping(task_plan_ir.get("contract_annotated_task_graph"))
    records = _record_index(task_plan_ir)
    return {
        "schema_version": graph.get("schema_version", SCHEMA_VERSION),
        "node_count": len(graph.get("nodes") or []),
        "code_owner_count": len(graph.get("code_owner_ids") or []),
        "registered_unit_count": len((_mapping(graph.get("ownership_map")).get("units") or [])),
        "development_check_count": len(graph.get("development_checks") or []),
        "responsibility_record_count": len(records),
        "evidence_record_count": len(task_plan_ir.get("evidence_records") or []),
        "validation_errors": list(graph.get("validation_errors") or task_plan_ir.get("catg_validation_errors") or []),
    }


__all__ = [
    "CATGNode",
    "ContractAnnotatedTaskGraph",
    "EvidenceRecord",
    "RepairEvent",
    "ResponsibilityRecord",
    "DEFAULT_EXECUTOR_REGISTRY",
    "build_contract_annotated_task_graph",
    "build_ownership_map",
    "catg_summary",
    "ensure_catg",
    "normalize_task_plan_ir",
    "record_local_evidence",
    "register_code_owner",
    "register_assembly_units",
    "route_executor",
    "validate_contract_graph",
]
