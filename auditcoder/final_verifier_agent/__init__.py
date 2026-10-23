def build_final_verifier_graph(*args, **kwargs):
    """Lazy import so deterministic audit primitives need no LangGraph."""

    from auditcoder.final_verifier_agent.graph import build_final_verifier_graph as _build

    return _build(*args, **kwargs)


__all__ = ["build_final_verifier_graph"]
