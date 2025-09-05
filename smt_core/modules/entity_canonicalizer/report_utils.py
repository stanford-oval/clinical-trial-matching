from __future__ import annotations
import os, datetime as dt, re
from typing import Any, Dict, List

# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _md_escape(txt: str) -> str:
    """Escape vertical bars / newlines so Markdown tables stay intact."""
    return txt.replace("|", "\\|").replace("\n", " ")


_tag_re = re.compile(r"\(([^()]+)\)$")


def _semantic_tag(fsn: str | None) -> str:
    """Return the semantic-tag (text in parentheses) from an FSN."""
    if not fsn:
        return "—"
    m = _tag_re.search(fsn.rstrip())
    return m.group(1) if m else "—"


def _candidate_lookup(ctx: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build surface-string → first-candidate map (case-insensitive).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for c in ctx.get("candidates", []):
        if "concept_id" not in c:
            continue
        key = c["extracted_span"].lower() if "extracted_span" in c else c["text"].lower()
        out.setdefault(key, c)
    return out


# ────────────────────────────────────────────────────────────────────────────
#  Main report writer
# ────────────────────────────────────────────────────────────────────────────
def write_entity_report(context: Dict[str, Any], out_dir: str = "reports") -> str:
    """
    Produce a Markdown file showing:
      • LLM-extracted entities vs. recall target
      • Filtered / canonicalised entities (extracted_span → PT)
    Returns the file path.
    """
    _ensure_dir(out_dir)

    trial_id = re.sub(r"[^\w\-]", "_", str(context.get("trial_id", "unknown")))
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"{trial_id}_entity_report_{ts}.md")

    reqs: List[Any] = context.get("requirements", [])
    refs_all = context.get("requirement_entities", [])
    valid_by_req = context.get("valid_entities_by_req", {})  # after filter

    # Which raw-entity mapping is available?
    for key in (
        "llm_surface_entities_by_req",
        "entities_by_req",
        "surface_entities_by_req",
    ):
        if key in context:
            by_ent = context[key]
            break
    else:
        by_ent = {}

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Entity extraction report (LLM vs. Recall)\n\n")
        fh.write(f"*Generated: {dt.datetime.now().isoformat(timespec='seconds')}*\n\n")

        # ───────── iterate over requirements ──────────────────────────
        for idx, raw in enumerate(reqs):
            # Requirement header
            fh.write(f"## Requirement {idx}\n")
            raw_txt = (
                raw
                if isinstance(raw, str)
                else raw.get("requirement")
                or raw.get("text")
                or raw.get("sentence")
                or str(raw)
            ).strip().replace("\n", " ")
            fh.write(f"> {raw_txt}\n\n")

            # ----- LLM vs. recall table -----
            llm_list = [
                _md_escape(
                    ent.get("extracted_span")
                    or ent.get("text")
                    or ent.get("surface")
                    or str(ent)
                )
                for ent in by_ent.get(idx, [])
            ]
            recall_list: List[str] = []
            if idx < len(refs_all):
                recall_list = [
                    _md_escape(s) for s in refs_all[idx].get("entities_recall", [])
                ]

            if not llm_list and not recall_list:
                fh.write("_No entities on either side._\n\n")
            else:
                fh.write("| LLM-extracted | Recall (target) |\n")
                fh.write("|---|---|\n")
                for i in range(max(len(llm_list), len(recall_list))):
                    llm_cell = llm_list[i] if i < len(llm_list) else ""
                    rec_cell = recall_list[i] if i < len(recall_list) else ""
                    fh.write(f"| {llm_cell} | {rec_cell} |\n")
                fh.write("\n")

            # ----- Filtered / canonical table -----
            kept = valid_by_req.get(idx, {})
            if kept:
                fh.write("| Entity after filter | Preferred term | Tag |\n")
                fh.write("|---|---|---|\n")
                for ent in kept.values():
                    surface = _md_escape(ent.get("extracted_span", ""))
                    pt = _md_escape(ent.get("preferred_term", ""))
                    tag = _semantic_tag(ent.get("fully_specified_name"))
                    fh.write(f"| {surface} | {pt} | {tag} |\n")
                fh.write("\n")

    return path