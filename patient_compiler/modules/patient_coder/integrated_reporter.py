# ─────────────────────────────────────────────────────────────────
# Integrated, minimal per-requirement report + per-attempt SMT slices
# ─────────────────────────────────────────────────────────────────
import json as _json
from dataclasses import dataclass, field
import re
from pathlib import Path

_TAG_FINDER = re.compile(r":named\s+([Rr]\d+_A\d+_[A-Z0-9_]+)")

def _first_n(xs, n=50):
    xs = list(xs or [])
    if len(xs) <= n:
        return xs
    return xs[:n] + [f"... (+{len(xs)-n} more)"]

def _req_text(ctx: dict, i: int) -> str:
    try:
        r = ctx["requirements"][i]
        return (r.get("requirement") if isinstance(r, dict) else str(r or "")).strip()
    except Exception:
        return ""

def _component_texts(ctx: dict, i: int) -> list[str]:
    try:
        r = ctx["requirements"][i]
        comps = r.get("components") or [] if isinstance(r, dict) else []
        out = []
        for c in comps:
            out.append(c.get("text", "") if isinstance(c, dict) else str(c))
        return [t for t in out if t]
    except Exception:
        return []

def _collect_timeframes(decls: list[dict]) -> list[str]:
    vals = {d.get("timeframe", "") for d in decls if isinstance(d, dict)}
    return sorted([v for v in vals if v])

def _decl_names(decls: list[dict]) -> list[str]:
    return [d.get("entity_variable_name", "") for d in decls if isinstance(d, dict) and d.get("entity_variable_name")]

def _named_tags_from_lines(lines: list[str]) -> list[str]:
    tags = []
    for ln in lines or []:
        for m in _TAG_FINDER.finditer(ln):
            tags.append(m.group(1))
    return sorted(set(tags))

@dataclass
class _ReqReport:
    trial_id: str
    side: str
    req_idx: int
    requirement: str = ""
    component_texts: list[str] = field(default_factory=list)

    # namer
    namer_mode: str = ""
    decl_counts: dict = field(default_factory=dict)
    decl_names: list[str] = field(default_factory=list)
    timeframes: list[str] = field(default_factory=list)
    reusable_count: int = 0
    namer_errors: list[str] = field(default_factory=list)

    # translator (aggregate)
    tag_count: int = 0
    tags: list[str] = field(default_factory=list)
    fragment_line_count: int = 0

    # solver (aggregate)
    solver_status: str = ""
    solver_elapsed_sec: float | None = None
    unsat_core: list[str] = field(default_factory=list)
    solver_message: str = ""

    # verifier (aggregate)
    verifier_ok: bool | None = None
    verifier_checks: dict | None = None

    # meta (aggregate)
    attempt: int | None = None
    success: bool | None = None

    # NEW: detailed per-attempt timeline
    attempts: list[dict] = field(default_factory=list)  # each: {attempt,label,fragment_path,line_count,tag_count,tags,solver_status,unsat_core}

    def to_min_dict(self) -> dict:
        """Compact report; clip long lists."""
        return {
            "trial_id": self.trial_id,
            "side": self.side,
            "req_idx": self.req_idx,
            "requirement": self.requirement,
            "components": _first_n(self.component_texts, 5),

            "namer": {
                "mode": self.namer_mode,
                "decl_counts": self.decl_counts,
                "decl_names": _first_n(self.decl_names, 50),
                "timeframes": self.timeframes,
                "reusable_count": self.reusable_count,
                "errors": _first_n(self.namer_errors, 20),
            },

            "translator": {
                "tag_count": self.tag_count,
                "tags": _first_n(self.tags, 100),
                "fragment_line_count": self.fragment_line_count,
            },

            "solver": {
                "status": self.solver_status,
                "elapsed_sec": self.solver_elapsed_sec,
                "unsat_core": _first_n(self.unsat_core, 50),
                "message": self.solver_message[:240],
            },

            "verifier": {
                "ok": self.verifier_ok,
                "checks": self.verifier_checks,
            },

            "attempt": self.attempt,
            "success": self.success,

            # NEW: per-attempt artifacts (tags are clipped here)
            "attempts": [
                {
                    "attempt": a.get("attempt"),
                    "label": a.get("label"),
                    "fragment_path": a.get("fragment_path"),
                    "line_count": a.get("line_count"),
                    "tag_count": a.get("tag_count"),
                    "tags": _first_n(a.get("tags", []), 100),
                    "solver_status": a.get("solver_status"),
                    "unsat_core": _first_n(a.get("unsat_core", []), 50),
                }
                for a in self.attempts
            ],
        }

class IntegratedRequirementReporter:
    """Small aggregator that reads from context and writes compact JSON + per-attempt .smt2 files."""
    def __init__(self, context: dict, req_idx: int, report_dir: str | Path | None):
        self.context = context
        self.req_idx = req_idx
        self.report_dir = Path(report_dir or "./req_reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)

        trial = context.get("trial_id", "unknown_trial")
        side  = context.get("inc_exc", "unknown")

        # Precompute base/slices dirs so we can write per-attempt files immediately
        self.base_dir = self.report_dir / f"{trial}_{side}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.slices_dir = self.base_dir / "slices"
        self.slices_dir.mkdir(parents=True, exist_ok=True)

        self.rep = _ReqReport(
            trial_id=str(trial),
            side=str(side),
            req_idx=req_idx,
            requirement=_req_text(context, req_idx),
            component_texts=_component_texts(context, req_idx),
        )

    # ——— stages ————————————————————————————————————————————————
    def capture_namer(self):
        ctx = self.context
        self.rep.namer_mode = ctx.get("namer_mode", "")

        asps = ctx.get("new_age_sex_pregnancystatus_declarations", []) or []
        canon = ctx.get("new_canonical_variable_declarations", []) or []
        other = ctx.get("all_other_variable_declarations", []) or ctx.get("new_noncanonical_variable_declarations", []) or []

        self.rep.decl_counts = {
            "age_sex_preg": len(asps),
            "canonical": len(canon),
            "other": len(other),
            "total": len(asps) + len(canon) + len(other),
        }
        self.rep.decl_names = _decl_names(asps) + _decl_names(canon) + _decl_names(other)
        self.rep.timeframes = sorted(set(_collect_timeframes(asps + canon + other)))
        self.rep.reusable_count = len(self.context.get("reusable_variables", []) or [])
        self.rep.namer_errors = list(self.context.get("errors", []) or [])

    # NEW: write the current translator fragment to a per-attempt .smt2 file
    def capture_fragment(self, attempt: int, *, label: str = "draft"):
        lines = self.context.get("new_smt_lines", []) or []
        tags = _named_tags_from_lines(lines)
        path = self.slices_dir / f"req_{self.rep.req_idx:03d}_attempt_{attempt:02d}_{label}.smt2"
        path.write_text("\n".join(lines), encoding="utf-8")

        # aggregate (top-level)
        self.rep.tags = sorted(set((self.rep.tags or []) + tags))
        self.rep.tag_count = len(self.rep.tags)
        self.rep.fragment_line_count = max(self.rep.fragment_line_count, len(lines))

        # per-attempt
        self._upsert_attempt_record(
            attempt,
            {
                "attempt": attempt,
                "label": label,
                "fragment_path": str(path),
                "line_count": len(lines),
                "tag_count": len(tags),
                "tags": tags,
            },
        )

    # NEW: update the same attempt with solver status/core
    def update_attempt_solver(self, attempt: int):
        sc = self.context.get("solver_check") or {}
        self._upsert_attempt_record(
            attempt,
            {
                "solver_status": sc.get("status", ""),
                "unsat_core": list(sc.get("unsat_core") or []),
            },
        )

        # keep aggregate solver fields too
        self.rep.solver_status = sc.get("status", "")
        self.rep.solver_message = (sc.get("message") or "")[:1000]
        stats = sc.get("stats") or {}
        self.rep.solver_elapsed_sec = stats.get("elapsed_sec")
        self.rep.unsat_core = list(sc.get("unsat_core") or [])

    # Optional: mark the accepted slice explicitly
    def mark_accepted(self, attempt: int):
        # write a convenience copy tagged as accepted (same current lines)
        lines = self.context.get("new_smt_lines", []) or []
        acc = self.slices_dir / f"req_{self.rep.req_idx:03d}_accepted_attempt_{attempt:02d}.smt2"
        acc.write_text("\n".join(lines), encoding="utf-8")
        self._upsert_attempt_record(
            attempt,
            {"label": "accepted", "fragment_path": str(acc), "line_count": len(lines)},
        )

    # helper to insert/update attempt record
    def _upsert_attempt_record(self, attempt: int, patch: dict):
        for rec in self.rep.attempts:
            if rec.get("attempt") == attempt:
                rec.update({k: v for k, v in patch.items() if v is not None})
                break
        else:
            self.rep.attempts.append({**patch})

    # ——— finalize/write ————————————————————————————————————————
    def finalize(self, *, success: bool, attempt: int | None):
        self.rep.success = success
        self.rep.attempt = attempt

    def write(self):
        per_req = self.base_dir / f"req_{self.rep.req_idx:03d}.json"
        per_req.write_text(_json.dumps(self.rep.to_min_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

        # also append to a trial-scoped JSONL for easy grepping
        jsonl = self.base_dir / "req_reports.jsonl"
        with open(jsonl, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(self.rep.to_min_dict(), ensure_ascii=False) + "\n")

        # keep in context for programmatic access
        self.context.setdefault("req_reports", {})[str(self.rep.req_idx)] = self.rep.to_min_dict()
