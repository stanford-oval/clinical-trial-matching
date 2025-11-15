#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
patient_coded_fact_entailment_pipeline_diagnose_classifed.py

Layout assumptions:
  - This script is in: <repo>/irsrc/patient_side/
  - run_multi_pass_pipeline_diagnose_classified.py is in the same folder
  - patch_uncertain_can_exclusion_from_certain.py is in the same folder
  - ONLY run_patient_diagnose_exclusion.py is in: <repo>/psrc/run_patient_diagnose_exclusion.py

Steps (NOW 6 steps):
1) Build <build-root>/patient_coded_results/<patient>/not_certain_canonical.jsonl
   from one or more SOURCE jsonl(s) (user-specified, repeatable).
   - Dedupe by 5-key:
       (entity_variable_name, start_time_in_hours, end_time_in_hours,
        start_time_inclusive, end_time_inclusive)
   - For diagnosis rows, can_be_used_for_exclusion OR-merge:
       True > False > None/missing

2) (NEW) Compare not_certain_canonical.jsonl with the resolved certain canonical (--certain-file-path).
   If a variable (entity_variable_name) appears in certain, then in not_certain:
     - if it's a diagnosis variable, force can_be_used_for_exclusion=True.

3) (NEW) Run run_patient_diagnose_exclusion.py on not_certain_canonical.jsonl with --inplace
   to annotate missing can_be_used_for_exclusion, overwriting not_certain_canonical.jsonl.

4) Run run_multi_pass_pipeline_diagnose_classified.py
   (it must read not_certain_canonical.jsonl as its input now)

5) Patch:
   patch_uncertain_can_exclusion_from_certain.py <note_id_base>
     --uncertain-jsonl <build-root>/patient_facts_export/<patient>/<side>/canonical.jsonl
     --certain-jsonl   <certain-file-path resolved for this patient>
   -> produces canonical.patched.jsonl next to uncertain canonical.jsonl

6) If canonical.patched.jsonl still has diagnosis rows missing/unparseable can_be_used_for_exclusion:
   run_patient_diagnose_exclusion.py <note_id_base>
     --diagnosis-jsonl canonical.patched.jsonl
     --out            canonical.patched.notdealtwith_complete.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import requests
from requests.adapters import HTTPAdapter
from functools import lru_cache


CAN_EXC_KEY = "can_be_used_for_exclusion"


def _find_repo_root(script_dir: Path) -> Path:
    for p in [script_dir] + list(script_dir.parents):
        if (p / "psrc").is_dir():
            return p
    return script_dir.resolve().parents[1]


def _boolish(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        try:
            return bool(int(v))
        except Exception:
            return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
    return None


def _is_diag(name: str) -> bool:
    return (name or "").strip().lower().startswith("patient_has_diagnosis_of_")


def _note_id_base(patient: str) -> str:
    p = patient
    for suf in ("_inclusion", "_exclusion"):
        if p.endswith(suf):
            return p[: -len(suf)]
    return p


def _dedupe_key(obj: Dict[str, Any]) -> Tuple:
    return (
        obj.get("entity_variable_name"),
        obj.get("start_time_in_hours"),
        obj.get("end_time_in_hours"),
        bool(obj.get("start_time_inclusive")),
        bool(obj.get("end_time_inclusive")),
    )


def _merge_can_exc(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """OR merge: True > False > None/missing."""
    if not _is_diag(str(dst.get("entity_variable_name") or "")):
        return
    dv = _boolish(dst.get(CAN_EXC_KEY))
    sv = _boolish(src.get(CAN_EXC_KEY))
    if dv is True or sv is True:
        dst[CAN_EXC_KEY] = True
        return
    if dv is False or sv is False:
        dst[CAN_EXC_KEY] = False
        return
    # else keep missing/None


def _run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _read_jsonl_objects(p: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _write_jsonl_objects(p: Path, rows: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------------- SNOMED (Snowstorm) preferred_term auto-fill ----------------

class SnowstormClient:
    """
    Same usage as VectorEmbeddingConceptSearch._fetch_pt_fsn:
      GET {snowstorm_url}/browser/{branch}/concepts/{cid}?expand=fsn(),pt()
      then read: (data.get("pt") or {}).get("term", "")
    """
    def __init__(self, *, snowstorm_url: str = "http://localhost:8080", branch: str = "MAIN", timeout: float = 8.0, verbose: bool = True):
        self.snowstorm_url = snowstorm_url.rstrip("/") if snowstorm_url else ""
        self.branch = branch
        self.timeout = float(timeout)
        self.verbose = bool(verbose)

        self.http = requests.Session()
        self.http.headers.update({"Accept": "application/json"})
        try:
            adapter = HTTPAdapter(pool_maxsize=32)
            self.http.mount("http://", adapter)
            self.http.mount("https://", adapter)
        except Exception:
            pass

    @lru_cache(maxsize=8192)
    def fetch_pt(self, cid: str) -> str:
        if not self.snowstorm_url:
            return ""
        cid = str(cid).strip()
        if not cid:
            return ""
        url = f"{self.snowstorm_url}/browser/{self.branch}/concepts/{cid}?expand=fsn(),pt()"
        try:
            r = self.http.get(url, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            pt = (data.get("pt") or {}).get("term", "")
            return str(pt or "").strip()
        except Exception as exc:
            if self.verbose:
                print(f"[snomed][warn] Snowstorm PT lookup failed for conceptId={cid}: {exc}")
            return ""


def fill_preferred_term_in_jsonl_inplace(p: Path, snomed: SnowstormClient) -> int:
    """
    In-place: for each JSON line, if top-level preferred_term missing/empty,
    lookup by top-level conceptId via Snowstorm and fill preferred_term.
    Returns number of patched rows.
    """
    if not p.is_file():
        return 0

    tmp = p.with_suffix(p.suffix + ".tmp_prefterm")
    patched = 0
    total = 0
    missing = 0

    with p.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.rstrip("\n")
            s = raw.strip()
            if not s:
                fout.write(raw + "\n")
                continue
            try:
                obj = json.loads(s)
            except Exception:
                # keep original if unparsable
                fout.write(raw + "\n")
                continue
            if not isinstance(obj, dict):
                fout.write(raw + "\n")
                continue

            total += 1
            pt = obj.get("preferred_term")
            if pt is None or (isinstance(pt, str) and not pt.strip()):
                missing += 1
                cid = obj.get("conceptId")
                # 你说 conceptId 一定有；这里仍做防御
                if cid is not None:
                    new_pt = snomed.fetch_pt(str(cid))
                    if new_pt:
                        obj["preferred_term"] = new_pt
                        patched += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if patched > 0:
        tmp.replace(p)
    else:
        try:
            tmp.unlink()
        except Exception:
            pass

    print(f"[snomed] patched preferred_term: file={p} total={total} missing_before={missing} patched={patched}")
    return patched


def resolve_source_paths(
    patient_dir: Path,
    patient: str,
    side: str,
    sources: List[str],
    default_name: str,
) -> List[Path]:
    out: List[Path] = []
    if sources:
        for s in sources:
            ss = s
            if "{patient}" in ss or "{side}" in ss:
                ss = ss.format(patient=patient, side=side)
            p = Path(ss).expanduser()
            if not p.is_absolute():
                p = (patient_dir / p)
            out.append(p.resolve())
        return out

    p = (patient_dir / default_name).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"default source not found: {p} (pass --source-jsonl to specify sources)")
    return [p]


def prepare_not_certain_canonical(
    patient_dir: Path,
    source_paths: List[Path],
    *,
    target_name: str = "not_certain_canonical.jsonl",
) -> Path:
    merged: List[Dict[str, Any]] = []
    for sp in source_paths:
        if not sp.is_file():
            raise FileNotFoundError(f"source not found: {sp}")
        merged.extend(_read_jsonl_objects(sp))

    by_key: Dict[Tuple, Dict[str, Any]] = {}
    order: List[Tuple] = []

    for obj in merged:
        evn = obj.get("entity_variable_name") or ""
        if _is_diag(str(evn)):
            v = _boolish(obj.get(CAN_EXC_KEY))
            if v is True:
                obj[CAN_EXC_KEY] = True
            elif v is False:
                obj[CAN_EXC_KEY] = False
            else:
                pass
        else:
            obj.pop(CAN_EXC_KEY, None)

        k = _dedupe_key(obj)
        if k not in by_key:
            by_key[k] = obj
            order.append(k)
        else:
            cur = by_key[k]
            _merge_can_exc(cur, obj)
            for fld in (
                "conceptId", "preferred_term", "fully_specified_name",
                "type", "fact_id", "template", "span_match",
                "entity_variable_meaning", "extracted_value", "mapping",
                "unit", "timeframe",
            ):
                if cur.get(fld) in (None, "") and obj.get(fld) not in (None, ""):
                    cur[fld] = obj.get(fld)

    dedup_rows = [by_key[k] for k in order]
    target_path = patient_dir / target_name
    _write_jsonl_objects(target_path, dedup_rows)
    print(f"[step1] wrote {len(dedup_rows)} dedup rows -> {target_path}")
    return target_path


def discover_patients(patient_root: Path, specified: Optional[str]) -> List[str]:
    if specified:
        return [specified]
    if not patient_root.is_dir():
        return []
    return sorted(p.name for p in patient_root.iterdir() if p.is_dir())


def resolve_certain_file_path(
    certain_tmpl: str,
    *,
    patient: str,
    side: str,
    project_root: Path,
) -> Path:
    patient_side = patient if patient.endswith(f"_{side}") else f"{patient}_{side}"

    s = certain_tmpl
    if ("{patient}" in s) or ("{side}" in s) or ("{patient_side}" in s):
        s = s.format(patient=patient, side=side, patient_side=patient_side)

    p = Path(s).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    else:
        p = p.resolve()

    if p.is_dir():
        for cand in (patient, patient_side):
            q = p / "patient_facts_export" / cand / side / "canonical.jsonl"
            if q.is_file():
                return q.resolve()
        tried = "\n  - " + "\n  - ".join(
            str((p / "patient_facts_export" / cand / side / "canonical.jsonl").resolve()) for cand in (patient, patient_side)
        )
        raise FileNotFoundError(f"--certain-file-path points to dir but canonical.jsonl not found. Tried:{tried}")

    if not p.is_file():
        raise FileNotFoundError(f"certain canonical file not found: {p}")

    return p


def _load_var_set_from_certain(certain_jsonl: Path) -> Set[str]:
    """
    Load a set of entity_variable_name from certain canonical.jsonl.
    We store lowercase names for case-insensitive match.
    """
    names: Set[str] = set()
    with certain_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            n = obj.get("entity_variable_name")
            if isinstance(n, str) and n.strip():
                names.add(n.strip().lower())
    return names


def upgrade_not_certain_with_certain(
    not_certain_path: Path,
    certain_jsonl: Path,
) -> Tuple[int, int]:
    """
    Step2 (NEW):
    If a not_certain row's entity_variable_name appears in certain,
    then for DIAGNOSIS rows force can_be_used_for_exclusion=True.

    Returns (diag_rows_seen, rows_upgraded_to_true).
    """
    certain_names = _load_var_set_from_certain(certain_jsonl)

    tmp_path = not_certain_path.with_suffix(not_certain_path.suffix + ".tmp_upgrade")
    diag_seen = 0
    upgraded = 0

    with not_certain_path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            evn = obj.get("entity_variable_name") or ""
            evn_l = str(evn).strip().lower()

            if _is_diag(evn_l):
                diag_seen += 1
                if evn_l in certain_names:
                    if _boolish(obj.get(CAN_EXC_KEY)) is not True:
                        obj[CAN_EXC_KEY] = True
                        upgraded += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    tmp_path.replace(not_certain_path)
    print(f"[step2] compared with certain: diag_seen={diag_seen}, upgraded_to_true={upgraded} -> {not_certain_path}")
    return diag_seen, upgraded


def count_missing_can_exc_in_file(p: Path) -> Tuple[int, int]:
    """
    For any JSONL file:
      returns (diag_total, missing_can_exc)
    missing means can_be_used_for_exclusion is missing/unparseable on diagnosis rows.
    """
    diag_total = 0
    missing = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            evn = (obj.get("entity_variable_name") or "").lower()
            if evn.startswith("patient_has_diagnosis_of_"):
                diag_total += 1
                if _boolish(obj.get(CAN_EXC_KEY)) is None:
                    missing += 1
    return diag_total, missing


def has_missing_can_exc(patched_jsonl: Path) -> bool:
    _, missing = count_missing_can_exc_in_file(patched_jsonl)
    return missing > 0


def main() -> None:
    ap = argparse.ArgumentParser(description="patient_coded_fact_entailment_pipeline_diagnose_classifed (6 steps)")

    ap.add_argument("--build-root", required=True)
    ap.add_argument(
        "--certain-file-path",
        required=True,
        help="Path (or template) to certain canonical.jsonl. Supports {patient},{side},{patient_side}. "
             "If you pass a directory, it will be treated as a build root.",
    )
    ap.add_argument("--passes", type=int, required=True)
    ap.add_argument("--side", choices=["inclusion", "exclusion"], default="inclusion")
    ap.add_argument("--patient", default=None)

    ap.add_argument("--source-jsonl", action="append", default=[],
                    help="Source jsonl(s) to build not_certain_canonical.jsonl (repeatable). Supports {patient},{side}.")
    ap.add_argument("--not-certain-name", default="not_certain_canonical.jsonl")

    # scripts: run_multi + patch are in SAME folder as this script by default
    ap.add_argument("--scripts-dir", default=None,
                    help="Optional override dir for run_multi/patch scripts (default: same dir as this script)")
    ap.add_argument("--run-multi-script", default="run_multi_pass_pipeline_diagnose_classified.py")
    ap.add_argument("--patch-script", default="patch_uncertain_can_exclusion_from_certain.py")

    # run_patient_diagnose_exclusion.py default is <repo>/psrc/run_patient_diagnose_exclusion.py
    ap.add_argument(
        "--diagnose-exclusion-script",
        default="",
        help="Optional override path to run_patient_diagnose_exclusion.py. "
             "Default: <repo>/psrc/run_patient_diagnose_exclusion.py",
    )

    ap.add_argument("--note-text-file", default="", help="Forward to run_patient_diagnose_exclusion.py")
    ap.add_argument("--chunk-size", type=int, default=10, help="Forward to run_patient_diagnose_exclusion.py")
    ap.add_argument("--log-root", default="", help="Forward to run_patient_diagnose_exclusion.py")

    ap.add_argument("--snowstorm-url", default="http://localhost:8080",
                    help="Snowstorm base URL (same as before). Default: http://localhost:8080. Set '' to disable.")
    ap.add_argument("--snowstorm-branch", default="MAIN",
                    help="Snowstorm branch (same as before). Default: MAIN.")
    ap.add_argument("--snowstorm-timeout", type=float, default=8.0,
                    help="Snowstorm HTTP timeout seconds (default: 8).")

    args = ap.parse_args()

    if args.passes < 1:
        raise SystemExit("--passes must be >= 1")

    snomed = SnowstormClient(
        snowstorm_url=args.snowstorm_url,
        branch=args.snowstorm_branch,
        timeout=args.snowstorm_timeout,
        verbose=True,
    )

    script_dir = Path(__file__).resolve().parent                       # <repo>/irsrc/patient_side
    repo_root = _find_repo_root(script_dir)                            # <repo>

    scripts_dir = Path(args.scripts_dir).expanduser().resolve() if args.scripts_dir else script_dir
    project_root = repo_root

    build_root = Path(args.build_root).expanduser()
    build_root = (project_root / build_root).resolve() if not build_root.is_absolute() else build_root.resolve()

    # ensure build_root + patient_coded_results exists
    build_root.mkdir(parents=True, exist_ok=True)
    patient_root = build_root / "patient_coded_results"
    patient_root.mkdir(parents=True, exist_ok=True)
    if args.patient:
        (patient_root / args.patient).mkdir(parents=True, exist_ok=True)

    run_multi = (scripts_dir / args.run_multi_script).resolve()
    patch_py = (scripts_dir / args.patch_script).resolve()

    if args.diagnose_exclusion_script:
        diag_excl = Path(args.diagnose_exclusion_script).expanduser()
        if not diag_excl.is_absolute():
            diag_excl = (scripts_dir / diag_excl).resolve()
        else:
            diag_excl = diag_excl.resolve()
    else:
        diag_excl = (repo_root / "psrc" / "run_patient_diagnose_exclusion.py").resolve()

    for pth in (run_multi, patch_py, diag_excl):
        if not pth.is_file():
            raise SystemExit(f"[err] script not found: {pth}")

    patients = discover_patients(patient_root, args.patient)
    if not patients:
        raise SystemExit(f"[err] no patients found under {patient_root}")

    for patient in patients:
        print(f"\n===== patient={patient} side={args.side} =====")
        pdir = patient_root / patient

        # Resolve certain canonical early (used in step2 and step5)
        try:
            certain_export = resolve_certain_file_path(
                args.certain_file_path,
                patient=patient,
                side=args.side,
                project_root=project_root,
            )
        except Exception as e:
            print(f"[err] {patient}: failed to resolve certain canonical: {e}")
            continue

        note_id = _note_id_base(patient)

        # ---------------- Step1 ----------------
        try:
            src_paths = resolve_source_paths(pdir, patient, args.side, args.source_jsonl, args.not_certain_name)
            not_certain_path = prepare_not_certain_canonical(pdir, src_paths, target_name=args.not_certain_name)
            if args.snowstorm_url:
                fill_preferred_term_in_jsonl_inplace(not_certain_path, snomed)

        except Exception as e:
            print(f"[skip] {patient}: failed step1 prepare_not_certain_canonical: {e}")
            continue

        # ---------------- Step2 (NEW) ----------------
        try:
            upgrade_not_certain_with_certain(not_certain_path, certain_export)
        except Exception as e:
            print(f"[warn] {patient}: step2 upgrade failed (continuing): {e}")

        # ---------------- Step3 (NEW) ----------------
        # Run diagnosis exclusion classifier on not_certain file, overwrite in place
        if args.snowstorm_url:
            fill_preferred_term_in_jsonl_inplace(not_certain_path, snomed)

        diag_total, missing_before = count_missing_can_exc_in_file(not_certain_path)
        print(f"[step3-pre] not_certain diagnosis_total={diag_total}, missing_can_exc={missing_before}")
        if missing_before > 0:
            cmd_fill_nc = [
                sys.executable, str(diag_excl),
                note_id,
                "--diagnosis-jsonl", str(not_certain_path),
                "--inplace",
                "--chunk-size", str(args.chunk_size),
            ]
            if args.note_text_file:
                cmd_fill_nc += ["--note-text-file", args.note_text_file]
            if args.log_root:
                cmd_fill_nc += ["--log-root", args.log_root]
            _run(cmd_fill_nc, cwd=diag_excl.parent)

            diag_total2, missing_after = count_missing_can_exc_in_file(not_certain_path)
            print(f"[step3-post] not_certain diagnosis_total={diag_total2}, missing_can_exc={missing_after}")
        else:
            print("[step3] not_certain already has can_exc for all diagnosis; skipping classifier")

        # ---------------- Step4 (was old Step2) ----------------
        cmd_multi = [
            sys.executable, str(run_multi),
            "--passes", str(args.passes),
            "--side", args.side,
            "--build-root", str(build_root),
            "--patient", patient,
        ]
        _run(cmd_multi, cwd=run_multi.parent)

        uncertain_export = build_root / "patient_facts_export" / patient / args.side / "canonical.jsonl"
        if not uncertain_export.is_file():
            print(f"[err] missing uncertain export canonical: {uncertain_export}")
            continue

        # ---------------- Step5 (was old Step3) ----------------
        cmd_patch = [
            sys.executable, str(patch_py),
            note_id,
            "--uncertain-jsonl", str(uncertain_export),
            "--certain-jsonl", str(certain_export),
        ]
        _run(cmd_patch, cwd=patch_py.parent)

        patched = uncertain_export.with_name("canonical.patched.jsonl")
        if not patched.is_file():
            patched = uncertain_export.with_name(uncertain_export.stem + ".patched" + uncertain_export.suffix)
        if not patched.is_file():
            print(f"[err] expected patched file not found next to {uncertain_export}")
            continue

        # ---------------- Step6 (was old Step4) ----------------
        if args.snowstorm_url:
            fill_preferred_term_in_jsonl_inplace(patched, snomed)

        diag_total_p, missing_p = count_missing_can_exc_in_file(patched)
        print(f"[step6-pre] patched diagnosis_total={diag_total_p}, missing_can_exc={missing_p}")

        complete = patched.with_name("canonical.patched.notdealtwith_complete.jsonl")
        if missing_p > 0:
            cmd_fill = [
                sys.executable, str(diag_excl),
                note_id,
                "--diagnosis-jsonl", str(patched),
                "--out", str(complete),
                "--chunk-size", str(args.chunk_size),
            ]
            if args.note_text_file:
                cmd_fill += ["--note-text-file", args.note_text_file]
            if args.log_root:
                cmd_fill += ["--log-root", args.log_root]
            _run(cmd_fill, cwd=diag_excl.parent)
        else:
            shutil.copyfile(patched, complete)
            print(f"[step6] no missing {CAN_EXC_KEY}; copied patched -> {complete}")

        print(f"[DONE] patient={patient}\n  certain={certain_export}\n  not_certain={not_certain_path}\n  patched={patched}\n  complete={complete}")


if __name__ == "__main__":
    main()
