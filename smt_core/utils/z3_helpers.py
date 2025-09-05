import datetime as _dt
import json
import pprint
from typing import List, Dict

import z3
from z3.z3util import get_vars  # tiny util shipped with z3-solver



def _log(stage: str, idx: int, extra: str = "") -> None:
    """Uniform console logger with timestamp."""
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] req#{idx:<3} – {stage:<18} {extra}", flush=True)


def _slice_as_text(ctx: dict, idx: int) -> str:
    """Return the SMT lines belonging only to requirement *idx*."""
    start, end = ctx["req_blocks"][idx]
    return "\n".join(ctx["smt_program_lines"][start:end])


def _whole_program(ctx: dict) -> str:
    """Return the entire SMT program currently accumulated."""
    return "\n".join(ctx.get("smt_program_lines", []))

def _print_variable_index(
    index: Dict[str, Dict[str, str]],
    style: str = "json"
) -> None:
    """
    Nicely print the SMT variable‑index mapping.

    Parameters
    ----------
    index : dict
        Mapping var_name → { "type": ..., "description": ... }
    style : {"json", "pprint"}
        • "json"   – deterministic, save‑friendly  
        • "pprint" – quick glance for debugging
    """
    if style == "json":
        # Sort keys so output is deterministic
        print(json.dumps(
            {k: index[k] for k in sorted(index)},
            indent=2,
            ensure_ascii=False
        ))
    else:
        pprint.pprint(index, sort_dicts=True)

        

def _collect_leaf_vars(smt_source: str) -> List[str]:
    """
    Return the set of *leaf* variables that actually appear in assertions.

    A leaf is defined as an uninterpreted constant with no arguments.
    """
    s       = z3.Solver()
    s.from_string(smt_source)        # load the whole program
    leafset = set()

    for a in s.assertions():
        for v in get_vars(a):        # z3util flattens duplicates for us
            if v.num_args() == 0:    # extra guard (must be a constant)
                leafset.add(v.decl().name())

    return sorted(leafset)



# ----------------------------------------------------------------------
def _print_leaf_variables(ctx: dict, w: int = 28) -> None:
    """
    Pretty-print every collected leaf variable.

    Shows:
        name | type | description | enum_values (if any)
    """
    leaf = ctx.get("leaf_detail", {})
    if not leaf:
        print("<< no leaf variables captured >>\n")
        return

    print("\n==== leaf variables (needed at runtime) ====")
    for v in sorted(leaf):
        meta  = leaf[v]
        typ   = meta.get("type", "<?>")
        desc  = meta.get("description", "")
        extra = ""
        if "enum_values" in meta:
            extra = "   {" + ", ".join(meta["enum_values"]) + "}"
        print(f"{v.ljust(w)} {typ.ljust(14)} {desc}{extra}")
    print()


def _print_patient_var_values(ctx: dict, w: int = 28) -> None:
    """
    Pretty-print the JSON mapping produced by SMTVariableValueMiner.

    Supports both
      • single-patient  – {var → value}
      • multi-patient   – {pid → {var → value}}
    """
    pv = ctx.get("patient_var_values", {})
    if not pv:
        print("<< no patient_var_values captured >>\n")
        return

    # decide batch vs single
    multi = any(isinstance(v, dict) for v in pv.values())
    items = pv.items() if multi else [("<patient>", pv)]

    for pid, mapping in items:
        if not isinstance(mapping, dict):
            print(f"\n!! WARNING: mapping for {pid} is not a dict – got {type(mapping).__name__}")
            continue

        print(f"\n==== extracted values for {pid} ====")
        for var in sorted(mapping):
            print(f"{var.ljust(w)} → {mapping[var]}")
    print()
