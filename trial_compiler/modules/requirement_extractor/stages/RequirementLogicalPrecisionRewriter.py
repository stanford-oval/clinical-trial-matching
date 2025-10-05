from pathlib import Path
import json
import dspy
from .RequirementLogicalPrecisionRewriterVerifier import RequirementLogicalPrecisionRewriterVerifier
from smt_core.parse_functions import parse_rewrite_output
import re

class RequirementLogicalPrecisionRewriter(dspy.Module):
    def __init__(self, engine, log_dir, max_attempts: int = 3):
        super().__init__()
        self.engine = engine
        self.max_attempts = max_attempts
        self.log_dir = log_dir

    def _sanitize_contextual_text(self, ctx_text: str) -> str:
        try:
            obj = json.loads(ctx_text)
        except Exception:
            # Not JSON? At least remove inclusion/exclusion blocks from the raw text.
            return re.sub(r"Inclusion criteria:.*?(?=\nExclusion criteria:|$)", "", ctx_text, flags=re.S|re.I)

        # Drop enumerations that duplicate your work
        for k in ["inclusion_criteria", "exclusion_criteria"]:
            if k in obj:
                del obj[k]

        # Remove inclusion enumeration from the long "text"
        if isinstance(obj.get("text"), str):
            obj["text"] = re.sub(r"Inclusion criteria:.*?(?=\nExclusion criteria:|$)", "", obj["text"], flags=re.S|re.I)

        # (Optional) keep only a whitelist
        keep = ["_id", "title", "metadata", "brief_summary", "text"]
        obj = {k: v for k, v in obj.items() if k in keep}
        return json.dumps(obj, ensure_ascii=False, indent=2)


    def forward(self, context: dict, use_full_context: bool = True) -> dict:
        # ---------- build rewrite prompt ----------
        raw_ctx = context.get("contextual_text", "")
        ctx_text = self._sanitize_contextual_text(raw_ctx)

        req_lines = [
            r["requirement"] if isinstance(r, dict) else str(r)
            for r in context.get("requirements", [])
        ]

        inc_exc = context.get("inc_exc")
        assert inc_exc in {"inclusion", "exclusion"}, "context['inc_exc'] must be 'inclusion' or 'exclusion'"

        if inc_exc == "inclusion":
            base_prompt = context["RequirementLogicalPrecisionRewriterInclusion_prompt"]
        else:
            base_prompt = context["RequirementLogicalPrecisionRewriterExclusion_prompt"]

        base_prompt = base_prompt.replace("#CONTEXTUAL_TEXT#", ctx_text)
        # === BATCHING CHANGE: do NOT inject full #REQUIREMENT_TEXT# here; we’ll inject per-batch below
        # base_prompt = base_prompt.replace("#REQUIREMENT_TEXT#", "\n".join(req_lines))

        # ---------- helper: render pairs text ----------
        def _pairs_text(requirements, index_offset=0):  # === BATCHING CHANGE: added index_offset
            lines = []
            for i, r in enumerate(requirements):
                if isinstance(r, dict):
                    orig = str(r.get("source", "")).strip()
                    rew = str(r.get("requirement", "")).strip()
                else:
                    orig = str(r).strip()
                    rew = str(r).strip()
                gi = i + index_offset  # === BATCHING CHANGE
                lines.append(f"[{gi:02d}] ORIGINAL: {orig}")
                lines.append(f"[{gi:02d}] REWRITTEN: {rew}")
            return "\n".join(lines)

        # ---------- PER-ITEM FALLBACK: tracking originals and which indices ever pass ----------
        original_req_lines = list(req_lines)  # index-aligned copy of originals
        ever_good = set()  # indices that have EVER been judged ALL_GOOD=YES

        def _mark_ever_good_from_ctx(ctx, index_offset=0):  # === BATCHING CHANGE: offset
            parsed = ctx.get("verification", {}).get("parsed", {}) or {}
            by_index = parsed.get("by_index", {}) or {}
            for k, v in by_index.items():
                if v.get("ALL_GOOD") == "YES":
                    try:
                        # add global offset so indices are global, not per-batch
                        ever_good.add(int(k) + index_offset)  # === BATCHING CHANGE
                    except (TypeError, ValueError):
                        pass

        # === BATCHING CHANGE: prepare global accumulators
        combined_requirements = []            # list of dicts in global order
        combined_precision_mapping = {}       # original -> rewritten (global)
        all_verifier_tries = []               # keep your existing detailed tries across batches
        final_verification_history = []       # aggregate verification histories
        success_any = False                   # not used for flow control across batches, but kept for parity

        # === BATCHING CHANGE: split into fixed-size batches of 10
        batch_size = 10
        batches = [req_lines[i:i + batch_size] for i in range(0, len(req_lines), batch_size)]

        # convenience for per-batch file naming
        #precision_log_out = self.log_dir
        precision_log_out = "mbench/req_mbench/precision_maps"

        if precision_log_out:
            base_path = Path(precision_log_out)
            out_dir = base_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = base_path.stem

        # === BATCHING CHANGE: factor the original “single-pass” core into a per-batch routine
        def _process_one_batch(req_lines_batch, batch_offset, batch_id):
            """
            Runs the original rewrite/verify loop on a single batch of requirements.
            `batch_offset` is the global index where this batch starts.
            Returns: (batch_requirements_as_dicts, batch_context) preserving order.
            """
            # Build the batch-specific prompt by injecting only this batch’s requirements
            prompt = base_prompt.replace("#REQUIREMENT_TEXT#", "\n".join(req_lines_batch))  # === BATCHING CHANGE
            # print(f"prompt is {prompt}")
            # ---------- OUTER LOOP: regeneration attempts ----------
            max_regen_attempts = int(context.get("precision_max_regen_attempts", 2))  # outer loop
            verifier_tries = []
            success = False
            verification_history = []

            # ---------- call LLM to rewrite with retries (inner LLM retry) ----------
            attempts = 0
            rewritten = None
            while attempts < self.max_attempts:
                llm_out = self.engine(prompt)[0]
                # print(f"llm_out is {llm_out}")
                rewritten = parse_rewrite_output(llm_out, expect_n=len(req_lines_batch))
                if rewritten is not False:
                    break
                attempts += 1
                print(f"[precision-rewriter][batch {batch_id}] rewrite retry {attempts}/{self.max_attempts}")

            if rewritten is False:
                rewritten = req_lines_batch  # graceful fallback

            # ---------- stitch back into same data shape (batch-local) ----------
            new_reqs = []
            mapping = {}
            for old, new in zip(req_lines_batch, rewritten):
                entry = {"requirement": new, "source": old}
                new_reqs.append(entry)
                mapping[old] = new

            # create a working copy of context for this batch
            work_ctx = dict(context)
            work_ctx["requirements"] = new_reqs
            work_ctx["precision_mapping"] = {**mapping}

            # ---------- run verifier (log initial try) ----------
            try:
                verifier = RequirementLogicalPrecisionRewriterVerifier(self.engine)
                work_ctx = verifier(work_ctx)
            except Exception as e:
                work_ctx["verification"] = {
                    "mode": inc_exc,
                    "raw": None,
                    "parsed": {"by_index": {}},
                    "all_good_counts": {"YES": 0, "NO": 0},
                    "error": f"{type(e).__name__}: {e}",
                }

            # PER-ITEM FALLBACK: record any indices that are already good after initial verify
            _mark_ever_good_from_ctx(work_ctx, index_offset=batch_offset)  # === BATCHING CHANGE

            # record the initial try of this regen round (with global indices)
            verifier_tries.append({
                "try_num": len(verifier_tries) + 1,
                "phase": f"batch{batch_id}_initial",
                "pairs_text": _pairs_text(work_ctx.get("requirements", []), index_offset=batch_offset),  # === BATCHING CHANGE
                "raw": work_ctx.get("verification", {}).get("raw"),
                "parsed": work_ctx.get("verification", {}).get("parsed"),
            })

            # ---------- INNER LOOP: auto-correct using verifier suggestions & re-verify ----------
            max_verify_passes = int(context.get("precision_max_verify_passes", 2))
            for reverify_pass in range(1, max_verify_passes + 1):
                parsed = work_ctx.get("verification", {}).get("parsed", {}) or {}
                by_index = parsed.get("by_index", {}) or {}

                # counts + 'all good' only if judgments exist
                yes = sum(1 for v in by_index.values() if v.get("ALL_GOOD") == "YES")
                no = sum(1 for v in by_index.values() if v.get("ALL_GOOD") == "NO")
                print(f"[precision-rewriter][batch {batch_id}] reverify pass #{reverify_pass}: YES={yes} NO={no} (empty? {not bool(by_index)})")

                all_good = bool(by_index) and all(v.get("ALL_GOOD") == "YES" for v in by_index.values())
                if all_good:
                    success = True
                    break

                verification_history.append(work_ctx.get("verification", {}))

                # apply corrections
                reqs = work_ctx.get("requirements", [])
                changed = False
                for k, v in by_index.items():
                    if v.get("ALL_GOOD") == "NO":
                        corr = (v.get("corrected_requirement") or "").strip()
                        if not corr:
                            continue
                        try:
                            idx = int(k)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= idx < len(reqs) and isinstance(reqs[idx], dict):
                            orig_source = reqs[idx].get("source", "")
                            old_text = (reqs[idx].get("requirement", "") or "").strip()
                            if corr and corr != old_text:
                                reqs[idx]["requirement"] = corr
                                if orig_source:
                                    work_ctx.setdefault("precision_mapping", {})[orig_source] = corr
                                changed = True

                print(f"[precision-rewriter][batch {batch_id}] reverify pass #{reverify_pass} corrections applied? {changed}")
                if not changed:
                    break

                # re-verify after applying corrections
                work_ctx = verifier(work_ctx)

                # PER-ITEM FALLBACK: mark items that became good after reverify
                _mark_ever_good_from_ctx(work_ctx, index_offset=batch_offset)  # === BATCHING CHANGE

                # record this reverify try
                verifier_tries.append({
                    "try_num": len(verifier_tries) + 1,
                    "phase": f"batch{batch_id}_reverify_pass_{reverify_pass}",
                    "pairs_text": _pairs_text(work_ctx.get("requirements", []), index_offset=batch_offset),  # === BATCHING CHANGE
                    "raw": work_ctx.get("verification", {}).get("raw"),
                    "parsed": work_ctx.get("verification", {}).get("parsed"),
                })

                # check again if healed
                parsed2 = work_ctx.get("verification", {}).get("parsed", {}) or {}
                by_index2 = parsed2.get("by_index", {}) or {}
                if bool(by_index2) and all(v.get("ALL_GOOD") == "YES" for v in by_index2.values()):
                    success = True
                    break

            # ---------- per-batch logging ----------
            if self.log_dir:
                base = Path(self.log_dir)
                out_dir = base.parent
                stem = base.stem

                (out_dir / f"{stem}.verifier_input_batch{batch_id}.txt").write_text(
                    _pairs_text(work_ctx.get("requirements", []), index_offset=batch_offset),
                    encoding="utf-8"
                )
                raw_text = work_ctx.get("verification", {}).get("raw") or ""
                (out_dir / f"{stem}.verifier_raw_batch{batch_id}.txt").write_text(raw_text, encoding="utf-8")
                parsed_json = work_ctx.get("verification", {}).get("parsed", {}) or {}
                (out_dir / f"{stem}.verifier_parsed_batch{batch_id}.json").write_text(
                    json.dumps(parsed_json, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            return work_ctx.get("requirements", []), work_ctx, verifier_tries, verification_history, success

        # === BATCHING CHANGE: run all batches and accumulate results
        global_index = 0
        for batch_id, req_lines_batch in enumerate(batches, start=1):
            batch_requirements, batch_ctx, vtries, vhist, healed = _process_one_batch(
                req_lines_batch, batch_offset=global_index, batch_id=batch_id
            )
            combined_requirements.extend(batch_requirements)
            all_verifier_tries.extend(vtries)
            final_verification_history.extend(vhist)
            success_any = success_any or healed
            # merge precision mappings from this batch
            for k, v in batch_ctx.get("precision_mapping", {}).items():
                combined_precision_mapping[k] = v
            global_index += len(req_lines_batch)

        # ---------- PER-ITEM FALLBACK: revert any index that never passed verification ----------
        reqs = combined_requirements  # === BATCHING CHANGE: now combined
        mapping = combined_precision_mapping
        for idx, item in enumerate(reqs):
            if isinstance(item, dict) and idx not in ever_good:
                # revert to the original input text for this index
                original_text = item.get("source", original_req_lines[idx] if idx < len(original_req_lines) else "")
                item["requirement"] = original_text
                if original_text:
                    # keep mapping truthful to the final state (original -> original)
                    mapping[original_text] = original_text

        # ---------- logging (aggregated) ----------
        trial_id = context.get("trial_id", "unknown")
        precision_log_out = f"mbench/req_mbench/precision_maps/{trial_id}_{inc_exc}_precision"  #self.log_dir
        print("[debug] precision_log_out is", precision_log_out)
        if precision_log_out:
            base = Path(precision_log_out)
            out_dir = base.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = base.stem

            # latest pairs (aggregated)
            verifier_input_text = _pairs_text(reqs, index_offset=0)  # already global indices
            (out_dir / f"{stem}.verifier_input.txt").write_text(verifier_input_text, encoding="utf-8")

            # tries
            if all_verifier_tries:
                (out_dir / f"{stem}.verifier_tries.json").write_text(
                    json.dumps(all_verifier_tries, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                for t in all_verifier_tries:
                    num = t.get("try_num", 0)
                    phase = t.get("phase", "unknown")
                    (out_dir / f"{stem}.verifier_input_try{num}_{phase}.txt").write_text(
                        t.get("pairs_text", ""), encoding="utf-8"
                    )
                    (out_dir / f"{stem}.verifier_raw_try{num}_{phase}.txt").write_text(
                        t.get("raw", "") or "", encoding="utf-8"
                    )
                    if t.get("parsed") is not None:
                        (out_dir / f"{stem}.verifier_parsed_try{num}_{phase}.json").write_text(
                            json.dumps(t["parsed"], ensure_ascii=False, indent=2),
                            encoding="utf-8"
                        )

            if final_verification_history:
                (out_dir / f"{stem}.verifier_history.json").write_text(
                    json.dumps(final_verification_history, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

        # ---------- finalize context ----------
        context["requirements"] = reqs
        context["precision_mapping"] = mapping
        if final_verification_history:
            context["verification_history"] = final_verification_history

        return context
