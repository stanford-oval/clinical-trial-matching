
import json

def load_entity_annotations(jsonl_path: str):
    """
    Build a lookup table from the JSONL produced by excel_multi2jsonl_keep_all.py

    Key  = (trial_id, criterion_type, criterion_index)
    Val  = {
        "entities_recall":    [...],   # list[dict{text: str}]
        "entities_precision": [...],
        "entities_old":       [...],
    }
    """
    table = {}
    with open(jsonl_path, encoding="utf-8") as fh:
        for ln in fh:
            rec = json.loads(ln)
            key = (rec["trial_id"], rec["criterion_type"], rec["criterion_index"])
            table[key] = {
                k: rec.get(k, [])            # may be absent → empty list
                for k in ("entities_recall", "entities_precision", "entities_old")
            }
    return table