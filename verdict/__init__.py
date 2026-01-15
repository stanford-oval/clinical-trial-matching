"""VERDICT — auditable patient--trial eligibility matching.

Decide a pair and see exactly why:

    import verdict

    d = verdict.match("sigir-20141__NCT00337116")
    print(d.decision, d.reasoning)
    for step in d.audit_trail:
        print(step.stage, step.decision)

Pick a different matcher variant:

    verdict.match(pair_id, system="hybrid")
    verdict.systems()                     # {name: description}

Missing pair data is NOT a verdict. By default the underlying variants return
decision="ineligible", reasoning="no data" (the behaviour the paper's numbers
were produced under), which downstream is indistinguishable from a real
INELIGIBLE. This API defaults to strict=True and raises instead:

    verdict.match("nope__NCT0")           # raises MissingPairData
    verdict.match(pair_id, strict=False)  # paper-compatible sentinel

Pair data is read from $VERDICT_PAIR_DATA (default <repo>/experiments/53_v2_full).
"""
from __future__ import annotations

from matchers.schema import Decision, AuditStep, MissingPairData, NO_DATA
from verdict.data import iter_pairs, pair_root
from verdict.registry import BUILTINS, ensure_builtins, get, register, unregister

__all__ = ["match", "explain", "systems", "pairs", "register", "unregister",
           "Decision", "AuditStep", "MissingPairData", "NO_DATA"]

#: Back-compat: the built-in name -> (variants attribute, description) table.
#: `systems()` and `match()` read the registry, which may hold more than this.
SYSTEMS = {name: (attr, desc) for name, attr, desc in BUILTINS}


def systems() -> dict[str, str]:
    """Registered matchers, as {name: description}.

    Includes anything added with `verdict.register`, not just the six from
    the paper.
    """
    ensure_builtins()
    from verdict.registry import systems as _systems
    return _systems()


def pairs() -> list[str]:
    """Pair ids available in the configured pair-data directory."""
    return list(iter_pairs())


def match(pair_id: str, system: str = "verdict", *, strict: bool = True) -> Decision:
    """Decide one patient--trial pair.

    Args:
        pair_id: "<patient>__<NCT>", e.g. "sigir-20141__NCT00337116".
        system:  a key of SYSTEMS (default "verdict").
        strict:  raise MissingPairData when the pair cannot be loaded.
                 Set False for the paper's sentinel behaviour.

    Returns:
        Decision(pair_id, variant, decision, reasoning, audit_trail).

    Raises:
        KeyError:          unknown `system`.
        MissingPairData:   pair not loadable and strict=True.
    """
    ensure_builtins()
    fn = get(system)                       # raises KeyError listing valid names
    from matchers import variants
    was = variants.is_strict()
    variants.strict(strict)
    try:
        return fn(pair_id)
    finally:
        variants.strict(was)


def explain(pair_id: str, system: str = "verdict", *, strict: bool = True,
            artifacts: bool = True) -> str:
    """The decision from match(), rendered as a readable audit trail.

    When `artifacts` is set and the pair has a stored SMT program, the
    MaxSMT accountability artifacts are appended: what the solver had to
    assume because the chart was silent, and what would change the answer.
    Silently omitted when no program is stored or no solver is installed --
    the audit trail above is unaffected either way.
    """
    d = match(pair_id, system, strict=strict)
    out = [f"pair    : {pair_id}",
           f"system  : {system}",
           f"decision: {d.decision.upper()}",
           f"why     : {d.reasoning}",
           "",
           f"audit trail ({len(d.audit_trail)} steps)"]
    for i, step in enumerate(d.audit_trail, 1):
        fields = step if isinstance(step, dict) else getattr(step, "__dict__", {})
        name = fields.get("stage") or fields.get("step") or f"step {i}"
        out.append(f"{i}. {name}")
        for k, v in fields.items():
            if k in ("stage", "step") or v in (None, "", {}, []):
                continue
            out.append(f"     {k}: {str(v)[:200]}")

    if artifacts:
        try:
            from verdict.artifacts import summarize
            extra = summarize(pair_id)
        except Exception:
            extra = ""                     # never let this break the trail
        if extra:
            out += ["", extra]
    return "\n".join(out)
