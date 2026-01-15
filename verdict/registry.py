"""Matcher registry.

The six matchers from the paper are registered at import. Anyone comparing a
new approach can add theirs without editing this package:

    import verdict

    @verdict.register("my-matcher", description="my approach")
    def my_matcher(pair_id: str):
        ...
        return Decision(pair_id, "my-matcher", "eligible", "because ...", [])

    verdict.match(pair_id, system="my-matcher")

A matcher is any callable taking a pair id and returning a
:class:`matchers.schema.Decision`.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from matchers.schema import Decision

#: name -> (callable, description). Built-ins are filled in by _register_builtins.
_REGISTRY: Dict[str, tuple] = {}

#: the names shipped with the paper, in presentation order
BUILTINS = (
    ("verdict",  "smt_lm_evidence_arbiter",
     "SMT + LLM auditor with LM-judge evidence (recommended, auditable)"),
    ("smt-only", "smt_raw",
     "atom miner + Z3 only, no review"),
    ("atoms",    "smt_atoms_arbiter",
     "SMT + LLM auditor on rejects (atoms only)"),
    ("lm-only",  "lm_only",
     "single LLM call over chart + criteria"),
    ("hybrid",   "hybrid_strict",
     "accept-if-either + LM-evidence arbiter (max F1)"),
)


def register(name: str, fn: Optional[Callable] = None, *,
             description: str = "", override: bool = False):
    """Register a matcher under `name`. Usable directly or as a decorator.

    Raises:
        ValueError: `name` is already registered and `override` is not set.
                    Shadowing a built-in silently would make two people's
                    "verdict" results incomparable.
    """
    def _add(f: Callable) -> Callable:
        if name in _REGISTRY and not override:
            raise ValueError(
                f"matcher {name!r} is already registered; pass override=True "
                f"if you mean to replace it")
        _REGISTRY[name] = (f, description or (f.__doc__ or "").strip().split("\n")[0])
        return f
    return _add if fn is None else _add(fn)


def unregister(name: str) -> None:
    """Remove a matcher. Mainly for tests."""
    _REGISTRY.pop(name, None)


def get(name: str) -> Callable:
    """The callable registered under `name`."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown system {name!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name][0]


def systems() -> Dict[str, str]:
    """Registered matchers as {name: description}."""
    return {k: v[1] for k, v in _REGISTRY.items()}


def _register_builtins() -> None:
    """Bind the paper's matchers lazily.

    matchers.variants is imported here rather than at module import so that
    `import verdict` stays cheap and does not need the solver.
    """
    from matchers import variants
    for name, attr, desc in BUILTINS:
        fn = getattr(variants, attr, None)
        if fn is not None and name not in _REGISTRY:
            _REGISTRY[name] = (fn, desc)


def ensure_builtins() -> None:
    """Register the built-ins if they are not present yet."""
    if not all(n in _REGISTRY for n, _, _ in BUILTINS):
        _register_builtins()
