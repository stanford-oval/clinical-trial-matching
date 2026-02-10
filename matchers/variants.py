"""The seven systems compared in the paper.

Each variant is a function with signature:
    decide(pair_id: str, engine=None) -> Decision

For variants that don't need an LLM (smt_raw, lm_only, smt_atoms_arbiter,
smt_lm_evidence_arbiter when caches are available), engine can be None.

For variants that need fresh LLM calls and have no cache, an engine is required.

Each variant returns a Decision with a structured audit_trail. Reading the
audit_trail of any two variants side-by-side shows exactly what each does.
"""
from __future__ import annotations
import os
import json, pathlib, hashlib
from typing import Any, Dict, List, Optional

from matchers.schema import Decision, AuditStep, MissingPairData, NO_DATA
from matchers.data import load_pair_data, load_judge_verdict
from matchers.prompts import (
    ATOMS_ONLY_ARBITER_PROMPT,
    LM_EVIDENCE_ARBITER_PROMPT,
    LM_JUDGE_PROMPT,
    LM_JUDGE_PRESCREEN_PROMPT,
    EXTRACTOR_PROMPT,
    CRITIC_PROMPT,
    MULTIAGENT_ARBITER_PROMPT,
)


# --------------------------------------------------------------------------
# Missing-pair handling
#
# By default a variant that cannot load its pair returns
# Decision(decision="ineligible", reasoning="no data"). That is what the
# paper's numbers were computed under, so it stays the default. It is unsafe
# downstream: an absent file then looks exactly like a real INELIGIBLE, and for
# a trial matcher that is the harmful direction. Enable strict mode to raise:
#
#     from matchers import variants
#     variants.strict(True)                  # or export VERDICT_STRICT=1
#
_STRICT = os.environ.get("VERDICT_STRICT", "").lower() in ("1", "true", "yes")


def strict(enabled: bool = True) -> None:
    """Raise MissingPairData instead of returning a 'no data' Decision."""
    global _STRICT
    _STRICT = bool(enabled)


def is_strict() -> bool:
    return _STRICT


def _no_data(pair_id: str, variant: str) -> Decision:
    if _STRICT:
        raise MissingPairData(
            f"no pair data for {pair_id!r} (variant {variant!r}). "
            f"Check $VERDICT_PAIR_DATA; see docs/DATA.md.")
    return Decision(pair_id, variant, "ineligible", NO_DATA, [])



ROOT = pathlib.Path(os.environ.get('VERDICT_ROOT',
    pathlib.Path(__file__).resolve().parents[1]))


# ============================================================================
# CACHE LOADING — re-use cached arbiter calls where available
# ============================================================================

def _load_jsonl_cache(path: pathlib.Path) -> Dict[str, Any]:
    out = {}
    if not path.exists(): return out
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        try:
            o = json.loads(line); out[o["key"]] = o["result"]
        except Exception: pass
    return out


def _fp(*parts):
    """Match the fingerprinting scheme used by experiments/97_smt_based/run_smt_based.py."""
    return hashlib.sha256("||".join(p[:s] for p, s in parts).encode()).hexdigest()


class _LazyJsonlCache(dict):
    """A cache that reads its files on first use, not at import.

    Importing this module used to read four jsonl caches from disk, so merely
    `import matchers.variants` did I/O and failed noisily where the corpus was
    absent. Paths are merged in the order given, so later files override
    earlier ones exactly as the previous eager `.update()` chain did.
    """

    def __init__(self, *rel_paths: str):
        super().__init__()
        self._rel_paths = rel_paths
        self._loaded = False

    def _ensure(self) -> None:
        if self._loaded:
            return
        self._loaded = True                      # set first: a failed read
        for rel in self._rel_paths:              # must not retry on every call
            super().update(_load_jsonl_cache(ROOT / rel))

    # every read path materialises the cache first
    def get(self, key, default=None):
        self._ensure(); return super().get(key, default)

    def __getitem__(self, key):
        self._ensure(); return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure(); return super().__contains__(key)

    def __len__(self):
        self._ensure(); return super().__len__()

    def __iter__(self):
        self._ensure(); return super().__iter__()

    def items(self):
        self._ensure(); return super().items()

    def keys(self):
        self._ensure(); return super().keys()

    def values(self):
        self._ensure(); return super().values()


_ATOMS_CACHE = _LazyJsonlCache("experiments/97_smt_based/cache_atoms_only.jsonl")
_LM_CACHE = _LazyJsonlCache("experiments/97_smt_based/cache_with_lm.jsonl")
# full cache first, then the disagreement-only cache (122 pairs) which overlaps
# it -- same merge order as before, so lookups resolve identically
_PRESCREEN_CACHE = _LazyJsonlCache(
    "experiments/99_counterfactual_lm/lm_prescreen_full_cache.jsonl",
    "experiments/99_counterfactual_lm/lm_prescreen_cache.jsonl")
_MULTIAGENT_CACHE = _LazyJsonlCache("experiments/100_multiagent_nl/multiagent_cache.jsonl")


# ============================================================================
# Helper: format blocking atoms for arbiter prompts
# ============================================================================

def _fmt_atoms(blockers, side_per_atom=None):
    """Format blocking atoms exactly the way experiments/96/97 scripts do.
    Used to ensure cache fingerprints match.
    """
    if not blockers: return ""
    lines = []
    for a in blockers:
        side = a.get("side", "inclusion")  # default; data loader fills this in
        val = a.get("value")
        v_str = "null" if val is None else val
        lines.append(
            f"  side={side}\n  atom: {a['atom']}\n  mined_value: {v_str}\n"
            f"  miner_rationale: {(a['rationale'] or '')[:280]}\n"
            f"  evidence_cited: {(a['evidence'] or '')[:180]}"
        )
    return "\n\n".join(lines)


# ============================================================================
# 1. TrialGPT (external baseline)
# ============================================================================

def trialgpt(pair_id: str, engine=None) -> Decision:
    """External baseline (Yang et al. 2023). Sentence-level scoring; we use
    cached predictions from the per_system field of the judge files.

    Audit trail: opaque; only a final decision is recovered, no per-criterion
    scoring is exposed by the cached output.
    """
    EXP53 = ROOT / "experiments/53_v2_full"
    pid, nct = pair_id.split("__", 1)
    tg_dec = None
    for jf in (EXP53 / "judges_clinician_v2").glob(f"{pid}__{nct}*.json"):
        try:
            j = json.loads(jf.read_text())
            tg_dec = (j.get("per_system") or {}).get("tg", {}).get("decision")
            break
        except Exception: pass
    if tg_dec not in ("eligible", "ineligible"): tg_dec = "ineligible"
    return Decision(
        pair_id=pair_id, variant="trialgpt", decision=tg_dec,
        reasoning="TrialGPT sentence-level scorer (cached prediction).",
        audit_trail=[AuditStep(stage="trialgpt_score", decision=tg_dec,
                               rationale="(opaque; no per-criterion scores in cache)",
                               evidence={})],
    )


# ============================================================================
# 2. LM-only (single-component ablation)
# ============================================================================

def lm_only(pair_id: str, engine=None) -> Decision:
    """A single LM call reads chart + criteria together; outputs eligibility +
    free-text rationale. The LM-judge component of our hybrid pipeline running
    in isolation.

    Audit trail: just the LM-judge's free-text rationale. No structured
    artifacts.
    """
    d = load_pair_data(pair_id)
    if d is None:
        return _no_data(pair_id, "lm_only")
    return Decision(
        pair_id=pair_id, variant="lm_only", decision=d["lm_decision"] or "ineligible",
        reasoning=(d["lm_explanation"] or d["lm_reasoning"])[:200],
        audit_trail=[AuditStep(
            stage="lm_judge", decision=d["lm_decision"],
            rationale=d["lm_reasoning"][:400],
            evidence={"explanation": d["lm_explanation"][:300]})],
    )


# ============================================================================
# 3. SMT-raw (single-component ablation)
# ============================================================================

def smt_raw(pair_id: str, engine=None) -> Decision:
    """The atom miner + Z3 solver running alone. No arbiter, no LM-judge.
    Corresponds to the prior-symbolic systems of SatIR/FABLE.

    Audit trail: every mined atom with its value and citation; the SMT solver's
    decision; on rejection, the named blocking atoms.
    """
    d = load_pair_data(pair_id)
    if d is None:
        return _no_data(pair_id, "smt_raw")

    audit = [AuditStep(
        stage="atom_mining", decision=None,
        rationale=f"Mined {d['n_total_atoms']} atoms across inclusion+exclusion.",
        evidence={"n_atoms": d["n_total_atoms"]})]

    audit.append(AuditStep(
        stage="smt_solve",
        decision=d["smt_decision"],
        rationale=(f"Inclusion: {d['smt_inclusion_status']}; "
                   f"Exclusion: {d['smt_exclusion_status']}."),
        evidence={"blocking_atoms": d["smt_blocking_atoms"]} if d["smt_blocking_atoms"] else {}
    ))

    return Decision(
        pair_id=pair_id, variant="smt_raw", decision=d["smt_decision"] or "ineligible",
        reasoning=(f"SMT solver: {d['smt_decision']} "
                   f"(inc={d['smt_inclusion_status']}, exc={d['smt_exclusion_status']})"),
        audit_trail=audit,
    )


# ============================================================================
# 4. SMT + atoms-only arbiter (auditable)
# ============================================================================

def smt_atoms_arbiter(pair_id: str, engine=None) -> Decision:
    """SMT-raw plus an LLM auditor on solver-rejects. The auditor sees ONLY
    the blocking atoms (no LM-judge rationale). It decides whether each is
    anchored to chart text. If all are heuristic, it deletes them and the
    solver re-runs; the SMT solver still owns the decision.
    """
    d = load_pair_data(pair_id)
    if d is None:
        return _no_data(pair_id, "smt_atoms_arbiter")

    audit = [AuditStep(stage="atom_mining", evidence={"n_atoms": d["n_total_atoms"]})]
    audit.append(AuditStep(
        stage="smt_solve", decision=d["smt_decision"],
        evidence={"blocking_atoms": d["smt_blocking_atoms"]} if d["smt_blocking_atoms"] else {}))

    # If solver accepted, no arbitration needed
    if d["smt_decision"] == "eligible":
        return Decision(pair_id, "smt_atoms_arbiter", "eligible",
                        "SMT solver accepted; no arbitration needed.", audit)

    # Solver rejected — consult arbiter cache
    atoms_block = _fmt_atoms(d["smt_blocking_atoms"])
    key = _fp(("atoms",10),(d["patient_note"],1500),
              (d["inclusion_criteria"],600),(d["exclusion_criteria"],600),
              (atoms_block,2200))
    arb = _ATOMS_CACHE.get(key)

    # If not in cache, fall back to cached canonical key from run_smt_based.py
    if arb is None and engine is None:
        # Try to find any matching arbiter call by exact note match
        for k, v in _ATOMS_CACHE.items():
            # Cache keys are SHA-based; can't reverse. Punt.
            pass

    if arb is None:
        # No cache hit; cannot run new LLM call without engine
        audit.append(AuditStep(stage="atoms_only_arbiter", decision="(uncached)",
                               rationale="No cached arbiter call; skipped."))
        return Decision(pair_id, "smt_atoms_arbiter", d["smt_decision"],
                        "No cached arbiter; using SMT verdict.", audit)

    arb_forward = arb["forward"]
    audit.append(AuditStep(
        stage="atoms_only_arbiter",
        decision="eligible" if arb_forward else "ineligible",
        rationale=arb.get("rationale", "")[:400],
        evidence={"action": "delete_blocking_atoms" if arb_forward else "keep_rejection"},
    ))
    final = "eligible" if arb_forward else "ineligible"
    return Decision(
        pair_id=pair_id, variant="smt_atoms_arbiter", decision=final,
        reasoning=("SMT rejected; atoms-only arbiter "
                   + ("deleted blocking atoms (forward)." if arb_forward
                      else "kept rejection.")),
        audit_trail=audit,
    )


# ============================================================================
# 5. SMT + LM-evidence arbiter (auditable; recommended)
# ============================================================================

def smt_lm_evidence_arbiter(pair_id: str, engine=None) -> Decision:
    """The auditable variant we recommend. SMT-raw plus an LLM auditor on
    rejects, where the auditor also sees a separate LM-judge's rationale as
    auxiliary chart-evidence. The LM-judge has NO decision authority; its
    rationale is just another input the auditor uses when judging atom
    grounding. The SMT solver still makes every decision."""
    d = load_pair_data(pair_id)
    if d is None:
        return _no_data(pair_id, "smt_lm_evidence_arbiter")

    audit = [AuditStep(stage="atom_mining", evidence={"n_atoms": d["n_total_atoms"]})]
    audit.append(AuditStep(stage="smt_solve", decision=d["smt_decision"],
        evidence={"blocking_atoms": d["smt_blocking_atoms"]} if d["smt_blocking_atoms"] else {}))

    if d["smt_decision"] == "eligible":
        return Decision(pair_id, "smt_lm_evidence_arbiter", "eligible",
                        "SMT solver accepted; no arbitration needed.", audit)

    atoms_block = _fmt_atoms(d["smt_blocking_atoms"])
    lm_rsn = d["lm_rsn_for_cache"]
    key = _fp(("lm",10),(d["patient_note"],1500),
              (d["inclusion_criteria"],600),(d["exclusion_criteria"],600),
              (atoms_block,2200),(lm_rsn,800))
    arb = _LM_CACHE.get(key)

    if arb is None:
        audit.append(AuditStep(stage="lm_evidence_arbiter", decision="(uncached)",
                               rationale="No cached arbiter call; skipped."))
        return Decision(pair_id, "smt_lm_evidence_arbiter", d["smt_decision"],
                        "No cached arbiter; using SMT verdict.", audit)

    arb_forward = arb["forward"]
    audit.append(AuditStep(
        stage="lm_evidence_arbiter",
        decision="eligible" if arb_forward else "ineligible",
        rationale=arb.get("rationale", "")[:400],
        evidence={
            "action": "delete_blocking_atoms" if arb_forward else "keep_rejection",
            "lm_judge_rationale_seen_by_arbiter": lm_rsn[:300],
        },
    ))
    final = "eligible" if arb_forward else "ineligible"
    return Decision(
        pair_id=pair_id, variant="smt_lm_evidence_arbiter", decision=final,
        reasoning=("SMT rejected; LM-evidence arbiter "
                   + ("deleted blocking atoms (forward)." if arb_forward
                      else "kept rejection.")),
        audit_trail=audit,
    )


# ============================================================================
# 6. Hybrid-loose (LM-judge has direct accept-authority)
# ============================================================================

def hybrid_loose(pair_id: str, engine=None) -> Decision:
    """Adds a parallel LM-judge to SMT-raw; accept-if-either rule. The
    atoms-only arbiter fires only when both reject.

    Audit caveat: when SMT rejects but LM-judge accepts, the patient is
    forwarded based on the LM-judge's free-text rationale alone. Not anchored
    to symbolic atoms. Decision.is_auditable() returns False for these cases.
    """
    d = load_pair_data(pair_id)
    if d is None:
        return _no_data(pair_id, "hybrid_loose")

    audit = [AuditStep(stage="atom_mining", evidence={"n_atoms": d["n_total_atoms"]})]
    audit.append(AuditStep(stage="smt_solve", decision=d["smt_decision"],
        evidence={"blocking_atoms": d["smt_blocking_atoms"]} if d["smt_blocking_atoms"] else {}))
    audit.append(AuditStep(stage="lm_judge", decision=d["lm_decision"],
        rationale=d["lm_reasoning"][:300],
        evidence={"explanation": d["lm_explanation"][:200]}))

    # Accept-if-either
    if d["smt_decision"] == "eligible":
        # SMT alone forwards (whether or not LM agrees)
        if d["lm_decision"] == "eligible":
            audit.append(AuditStep(stage="accept_if_either", decision="eligible",
                                   rationale="Both components accept."))
        else:
            audit.append(AuditStep(stage="accept_if_either", decision="eligible",
                                   rationale="SMT accepts; LM-judge rejects but accept-if-either forwards.",
                                   evidence={"authority": "smt_solver"}))
        return Decision(pair_id, "hybrid_loose", "eligible",
                        audit[-1].rationale, audit)

    if d["lm_decision"] == "eligible":
        # SMT rejects; LM accepts → forward on LM rationale alone (audit caveat)
        audit.append(AuditStep(stage="accept_if_either", decision="eligible",
                               rationale="SMT rejects; LM-judge accepts; accept-if-either forwards on LM-judge rationale alone (LM-rescues band).",
                               evidence={"authority": "lm_judge_alone"}))
        return Decision(pair_id, "hybrid_loose", "eligible", audit[-1].rationale, audit)

    # Both reject → atoms-only arbiter
    atoms_block = _fmt_atoms(d["smt_blocking_atoms"])
    key = _fp(("atoms",10),(d["patient_note"],1500),
              (d["inclusion_criteria"],600),(d["exclusion_criteria"],600),
              (atoms_block,2200))
    arb = _ATOMS_CACHE.get(key)
    if arb is None:
        audit.append(AuditStep(stage="atoms_only_arbiter", decision="(uncached)"))
        return Decision(pair_id, "hybrid_loose", "ineligible",
                        "Both reject; arbiter uncached; final reject.", audit)
    audit.append(AuditStep(
        stage="atoms_only_arbiter", decision="eligible" if arb["forward"] else "ineligible",
        rationale=arb.get("rationale","")[:400]))
    final = "eligible" if arb["forward"] else "ineligible"
    return Decision(pair_id, "hybrid_loose", final,
                    "Both reject; atoms-only arbiter " + ("forwarded." if arb["forward"] else "kept rejection."),
                    audit)


# ============================================================================
# 7. Hybrid-strict (LM-judge has direct accept-authority; recommended for max F1)
# ============================================================================

def hybrid_strict(pair_id: str, engine=None) -> Decision:
    """The hybrid variant we recommend for max F1. Same as Hybrid-loose, but
    the arbiter on both-reject cases is the LM-evidence arbiter (sees the
    LM-judge's rationale as auxiliary chart-evidence). Same audit caveat as
    Hybrid-loose: LM-judge has direct accept-authority on the LM-rescues band.
    """
    d = load_pair_data(pair_id)
    if d is None:
        return _no_data(pair_id, "hybrid_strict")

    audit = [AuditStep(stage="atom_mining", evidence={"n_atoms": d["n_total_atoms"]})]
    audit.append(AuditStep(stage="smt_solve", decision=d["smt_decision"],
        evidence={"blocking_atoms": d["smt_blocking_atoms"]} if d["smt_blocking_atoms"] else {}))
    audit.append(AuditStep(stage="lm_judge", decision=d["lm_decision"],
        rationale=d["lm_reasoning"][:300]))

    if d["smt_decision"] == "eligible":
        audit.append(AuditStep(stage="accept_if_either", decision="eligible",
                               rationale="Both accept." if d["lm_decision"] == "eligible"
                               else "SMT accepts; LM rejects; SMT forwards.",
                               evidence={"authority": "smt_solver"}))
        return Decision(pair_id, "hybrid_strict", "eligible", audit[-1].rationale, audit)

    if d["lm_decision"] == "eligible":
        audit.append(AuditStep(stage="accept_if_either", decision="eligible",
                               rationale="LM-rescues band: SMT rejects, LM accepts, forwarded on LM rationale alone.",
                               evidence={"authority": "lm_judge_alone"}))
        return Decision(pair_id, "hybrid_strict", "eligible", audit[-1].rationale, audit)

    # Both reject → LM-evidence arbiter
    atoms_block = _fmt_atoms(d["smt_blocking_atoms"])
    lm_rsn = d["lm_rsn_for_cache"]
    key = _fp(("lm",10),(d["patient_note"],1500),
              (d["inclusion_criteria"],600),(d["exclusion_criteria"],600),
              (atoms_block,2200),(lm_rsn,800))
    arb = _LM_CACHE.get(key)
    if arb is None:
        audit.append(AuditStep(stage="lm_evidence_arbiter", decision="(uncached)"))
        return Decision(pair_id, "hybrid_strict", "ineligible",
                        "Both reject; arbiter uncached; final reject.", audit)
    audit.append(AuditStep(
        stage="lm_evidence_arbiter", decision="eligible" if arb["forward"] else "ineligible",
        rationale=arb.get("rationale","")[:400],
        evidence={"lm_judge_rationale_seen": lm_rsn[:200]}))
    final = "eligible" if arb["forward"] else "ineligible"
    return Decision(pair_id, "hybrid_strict", final,
                    "Both reject; LM-evidence arbiter " + ("forwarded." if arb["forward"] else "kept rejection."),
                    audit)


# ============================================================================
# 8. LM-only with prescreen-doctrine prompt (stronger NL-only baseline)
# ============================================================================

def lm_only_prescreen(pair_id: str, engine=None) -> Decision:
    """A single LLM call with a strengthened prescreen-doctrine prompt that
    explicitly tells the LM to forward on chart silence. This is a stronger
    LM-only baseline than `lm_only`. Cached on the full 538-pair set.

    Key point: this beats Hybrid-strict on F2 (0.886 vs 0.869) by aggressive
    forwarding (96% recall at 69% precision), but loses on five of seven
    metrics. See paper Section 6.1 and the headline table.
    """
    d = load_pair_data(pair_id)
    if d is None:
        return _no_data(pair_id, "lm_only_prescreen")

    # Cache key matches experiments/99_counterfactual_lm/run_better_nl_full.py
    td = f"INCLUSION CRITERIA:\n{d['inclusion_criteria']}\n\nEXCLUSION CRITERIA:\n{d['exclusion_criteria']}"
    key = hashlib.sha256(json.dumps(
        {"trial": td[:1500], "note": d["patient_note"][:1500], "v": "v2"},
        sort_keys=True).encode()).hexdigest()
    r = _PRESCREEN_CACHE.get(key)
    if r is None:
        return Decision(pair_id, "lm_only_prescreen", "ineligible",
                        "uncached prescreen-prompt LM call.",
                        [AuditStep(stage="lm_judge_prescreen", decision="(uncached)")])
    dec = "eligible" if r["eligibility"] == "eligible" else "ineligible"
    return Decision(
        pair_id=pair_id, variant="lm_only_prescreen", decision=dec,
        reasoning=r.get("explanation", "")[:200],
        audit_trail=[AuditStep(
            stage="lm_judge_prescreen", decision=dec,
            rationale=r.get("explanation", "")[:400],
            evidence={"prompt_variant": "prescreen_doctrine"})],
    )


# ============================================================================
# 9. Multi-agent NL pipeline (4 stages, all NL — strongest NL baseline)
# ============================================================================

def multiagent_nl(pair_id: str, engine=None) -> Decision:
    """Four-stage NL pipeline mirroring our SMT pipeline's structural complexity:
        (1) extractor: identifies chart facts with quoted spans
        (2) critic: evaluates each criterion as satisfied/violated/silent
        (3) arbiter: classifies violations as contradiction/screening-deferred/speculative
        (4) decider: deterministic — reject only if any explicit-contradiction

    All steps are LM calls; no SMT solver.

    Key result: WORSE than the single-call prescreen baseline on every metric
    (F1=0.754 vs 0.799) and -0.142 kappa vs Hybrid-strict. Without a structural
    constraint, errors compound across NL stages.
    """
    d = load_pair_data(pair_id)
    if d is None:
        return _no_data(pair_id, "multiagent_nl")

    # Cache key matches experiments/100_multiagent_nl/run_multiagent.py
    note = d["patient_note"]; inc = d["inclusion_criteria"]; exc = d["exclusion_criteria"]

    def cache_key(stage, **fields):
        return hashlib.sha256(json.dumps(
            {"stage": stage, **fields}, sort_keys=True).encode()).hexdigest()

    # Stage 1: extractor
    k1 = cache_key("extractor", note=note[:1500])
    r1 = _MULTIAGENT_CACHE.get(k1)
    if r1 is None:
        return Decision(pair_id, "multiagent_nl", "ineligible",
                        "uncached multi-agent stage 1 (extractor).",
                        [AuditStep(stage="extractor", decision="(uncached)")])
    facts = r1.get("facts", [])

    audit = [AuditStep(
        stage="extractor", decision=None,
        rationale=f"Extracted {len(facts)} clinical facts.",
        evidence={"n_facts": len(facts), "sample": [f.get("fact","") for f in facts[:3]]},
    )]

    # Stage 2: critic
    facts_str = "\n".join(f"- {f.get('fact','?')}: \"{f.get('chart_quote','?')[:120]}\"" for f in facts[:25])
    k2 = cache_key("critic", facts=facts_str[:1500], inc=inc[:600], exc=exc[:600])
    r2 = _MULTIAGENT_CACHE.get(k2)
    if r2 is None:
        return Decision(pair_id, "multiagent_nl", "ineligible",
                        "uncached multi-agent stage 2 (critic).", audit + [
                        AuditStep(stage="critic", decision="(uncached)")])
    incs = r2.get("inclusion", [])
    excs = r2.get("exclusion", [])

    audit.append(AuditStep(
        stage="critic", decision=None,
        rationale=(f"Inclusion: {sum(1 for c in incs if c.get('status')=='satisfied')}/{len(incs)} satisfied; "
                   f"Exclusion: {sum(1 for c in excs if c.get('status')=='violated')}/{len(excs)} violated."),
        evidence={"n_inc": len(incs), "n_exc": len(excs)},
    ))

    # Stage 3: arbiter
    evals = "INCLUSION:\n"
    for c in incs[:15]:
        evals += f"  [{c.get('status','?')}] {c.get('criterion','?')[:80]} | evidence: {c.get('evidence','?')[:80]}\n"
    evals += "EXCLUSION:\n"
    for c in excs[:15]:
        evals += f"  [{c.get('status','?')}] {c.get('criterion','?')[:80]} | evidence: {c.get('evidence','?')[:80]}\n"
    k3 = cache_key("arbiter", evals=evals[:1800])
    r3 = _MULTIAGENT_CACHE.get(k3)
    if r3 is None:
        return Decision(pair_id, "multiagent_nl", "ineligible",
                        "uncached multi-agent stage 3 (arbiter).", audit + [
                        AuditStep(stage="multiagent_arbiter", decision="(uncached)")])
    audit_items = r3.get("audit", [])

    audit.append(AuditStep(
        stage="multiagent_arbiter", decision=None,
        rationale=(f"Audit: {sum(1 for a in audit_items if a.get('type')=='explicit-contradiction')} "
                   f"explicit-contradiction, "
                   f"{sum(1 for a in audit_items if a.get('type')=='screening-deferred')} screening-deferred, "
                   f"{sum(1 for a in audit_items if a.get('type')=='speculative-rejection')} speculative."),
        evidence={"n_items": len(audit_items)},
    ))

    # Stage 4: decider (deterministic)
    has_contradiction = any(a.get("type") == "explicit-contradiction" for a in audit_items)
    decision = "ineligible" if has_contradiction else "eligible"
    audit.append(AuditStep(
        stage="multiagent_decider", decision=decision,
        rationale=("Reject: at least one explicit-contradiction." if has_contradiction
                   else "Forward: no explicit-contradiction in audit."),
        evidence={"deterministic": True},
    ))

    return Decision(
        pair_id=pair_id, variant="multiagent_nl", decision=decision,
        reasoning=audit[-1].rationale, audit_trail=audit,
    )


# ============================================================================
# Registry
# ============================================================================

VARIANTS = {
    "trialgpt": trialgpt,
    "lm_only": lm_only,
    "lm_only_prescreen": lm_only_prescreen,
    "multiagent_nl": multiagent_nl,
    "smt_raw": smt_raw,
    "smt_atoms_arbiter": smt_atoms_arbiter,
    "smt_lm_evidence_arbiter": smt_lm_evidence_arbiter,
    "hybrid_loose": hybrid_loose,
    "hybrid_strict": hybrid_strict,
}

# Variants grouped by the paper's categories
VARIANT_GROUPS = {
    "external_baseline": ["trialgpt"],
    "single_component_ablations": ["lm_only", "smt_raw"],
    "stronger_nl_only_baselines": ["lm_only_prescreen", "multiagent_nl"],
    "smt_based_auditable": ["smt_atoms_arbiter", "smt_lm_evidence_arbiter"],
    "hybrid_partial_audit": ["hybrid_loose", "hybrid_strict"],
}
