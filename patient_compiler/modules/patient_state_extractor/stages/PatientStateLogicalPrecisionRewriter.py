from pathlib import Path
import json
import dspy
from .PatientStateLogicalPrecisionRewriterVerifier import PatientStateLogicalPrecisionRewriterVerifier
from smt_core.parse_functions import parse_rewrite_output
from typing import Dict, Any, List


class PatientStateLogicalPrecisionRewriter(dspy.Module):
    def __init__(self, engine, log_dir, max_attempts: int = 3):
        super().__init__()
        self.engine        = engine
        self.max_attempts  = max_attempts
        self.log_dir       = log_dir

    # ---------------------------------------------------------------
    def forward(self, context: dict, use_full_context: bool = True) -> dict:  # noqa: C901 (long fn)
        has_facts = bool(context.get("patient_facts"))
        slot_key  = "patient_facts" if has_facts else "requirements"
        text_key  = "fact" if has_facts else "requirement"

        items      = context.get(slot_key, [])
        req_lines  = [(it.get(text_key) if isinstance(it, dict) else str(it)) for it in items]
        original_req_lines = list(req_lines)

        # ---------- build prompt ----------
        ctx_text    = context.get("contextual_text", "")
        base_prompt = context["PatientStateLogicalPrecisionRewriter_prompt"].replace("#CONTEXTUAL_TEXT#", ctx_text)
        base_prompt = base_prompt.replace("#PATIENT_NOTE#", "\n".join(req_lines))

        # ---------- helpers ----------
        def _pairs_text(objs):
            lines = []
            for i, r in enumerate(objs):
                orig = str(r.get("source", r.get(text_key, "")).strip()) if isinstance(r, dict) else str(r).strip()
                rew  = str(r.get(text_key, "")).strip() if isinstance(r, dict) else str(r).strip()
                lines += [f"[{i:02d}] ORIGINAL: {orig}", f"[{i:02d}] REWRITTEN: {rew}"]
            return "\n".join(lines)

        # ---------- state ----------
        best_good: Dict[int, str] = {}
        verifier_tries: List[Dict[str, Any]] = []
        final_verification_history: List[Dict[str, Any]] = []
        success = False

        def _update_best_good(ctx):
            parsed = ctx.get("verification", {}).get("parsed", {}) or {}
            by_idx = parsed.get("by_index", {}) or {}
            reqs   = ctx.get(slot_key, [])
            for k, v in by_idx.items():
                if v.get("ALL_GOOD") == "YES":
                    try:
                        idx = int(k)
                        if 0 <= idx < len(reqs):
                            best_good[idx] = reqs[idx][text_key]
                    except ValueError:
                        continue

        # ---------- regen loop ----------
        max_regen_attempts = int(context.get("precision_max_regen_attempts", 2))
        for regen_round in range(1, max_regen_attempts + 1):
            # ------- obtain rewrite (with parse retries) -------
            attempts, rewritten = 0, None
            while attempts < self.max_attempts:
                out = self.engine(base_prompt)[0]
                rewritten = parse_rewrite_output(out, expect_n=len(req_lines))
                if rewritten is not False:
                    break
                attempts += 1
            if rewritten is False:
                rewritten = req_lines  # fallback

            # stitch
            new_items, mapping = [], {}
            for old, new, obj in zip(req_lines, rewritten, items):
                d = dict(obj) if isinstance(obj, dict) else {}
                d.update({text_key: new, "source": old})
                new_items.append(d)
                mapping[old] = new
            context[slot_key] = new_items
            context["precision_mapping"] = mapping

            # verifier
            verifier = PatientStateLogicalPrecisionRewriterVerifier(self.engine)
            context   = verifier(context)
            _update_best_good(context)

            verifier_tries.append({
                "regen": regen_round,
                "pass": 0,
                "raw": context["verification"].get("raw"),
                "parsed": context["verification"].get("parsed"),
            })

            # ------- iterative corrections -------
            max_verify_passes = int(context.get("precision_max_verify_passes", 2))
            for verify_pass in range(1, max_verify_passes + 1):
                parsed = context["verification"].get("parsed", {}) or {}
                by_idx = parsed.get("by_index", {}) or {}
                if by_idx and all(v.get("ALL_GOOD") == "YES" for v in by_idx.values()):
                    success = True; break

                reqs = context[slot_key]
                changed = False
                for k, v in by_idx.items():
                    if v.get("ALL_GOOD") == "NO":
                        idx = int(k)
                        corr = v.get("corrected_fact", "").strip()
                        if corr and corr != reqs[idx][text_key]:
                            reqs[idx][text_key] = corr
                            changed = True
                if not changed:
                    break

                context = verifier(context)
                _update_best_good(context)
                verifier_tries.append({
                    "regen": regen_round,
                    "pass": verify_pass,
                    "raw": context["verification"].get("raw"),
                    "parsed": context["verification"].get("parsed"),
                })

                parsed2 = context["verification"].get("parsed", {}) or {}
                if parsed2.get("by_index") and all(v.get("ALL_GOOD") == "YES" for v in parsed2["by_index"].values()):
                    success = True; break

            if success:
                break

        # ---------- final fallback ----------
        reqs = context.get(slot_key, [])
        mapping = context.get("precision_mapping", {})
        for idx, item in enumerate(reqs):
            if idx in best_good:
                item[text_key] = best_good[idx]
            else:
                original = original_req_lines[idx]
                item[text_key] = original
                mapping[original] = original
        context[slot_key] = reqs
        context["precision_mapping"] = mapping




        # -------- logging --------
        precision_log_out = self.log_dir
        if precision_log_out:
            base = Path(precision_log_out)
            out_dir = base.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = base.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{stem}.precision_verification_tries.json").write_text(json.dumps(verifier_tries, ensure_ascii=False, indent=2), encoding="utf-8")
        return context
