#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hier_edge_cache.py
Incremental, persistent hierarchy cache for SNOMED Snowstorm.

Features
--------
• children/parents edge store with "complete" markers
• descendants_cached(): BFS over cached children; fetches only frontier nodes once
• parents_cached() / ancestors_cached(): analogous for up-edges
• tiny HTTP client with retries + env-configurable base/branch/form defaults
• WAL SQLite; safe to share across runs

Env knobs
---------
SNOWSTORM_BASE          (default: http://localhost:8080)
SNOWSTORM_BRANCH        (default: MAIN)
ISA_HIER_CACHE_DIR      (default: ~/.trialgpt_hiercache)
ISA_HTTP_TIMEOUT_S      (default: 10)
ISA_HTTP_TRIES          (default: 3)
ISA_HTTP_BACKOFF        (default: 0.8)

Schema
------
TABLE children(branch, form, parent, child, PRIMARY KEY(...))
TABLE parents (branch, form, child, parent, PRIMARY KEY(...))
TABLE meta    (branch, form, cid, children_complete, parents_complete, PRIMARY KEY(...))
TABLE mini    (cid TEXT PRIMARY KEY, json TEXT)  # optional lightweight detail memo

"""

from __future__ import annotations
import os, json, time, sqlite3
from typing import List, Dict, Any, Optional, Tuple
import urllib.request, urllib.parse, urllib.error

# =========================
# Config
# =========================
snowstorm_base  = os.getenv("SNOWSTORM_BASE", "http://localhost:8080")
snowstorm_branch= os.getenv("SNOWSTORM_BRANCH", "MAIN")
DEFAULT_FORM    = os.getenv("SNOWSTORM_FORM", "inferred")  # inferred | stated

HIER_CACHE_DIR  = os.getenv("ISA_HIER_CACHE_DIR", os.path.expanduser("~/.trialgpt_hiercache"))
HIER_CACHE_DB   = os.path.join(HIER_CACHE_DIR, "hier.sqlite")
os.makedirs(HIER_CACHE_DIR, exist_ok=True)

# tighter, but still overrideable via env
HTTP_TIMEOUT_S  = int(os.getenv("ISA_HTTP_TIMEOUT_S", "10"))
HTTP_TRIES      = int(os.getenv("ISA_HTTP_TRIES", "3"))
HTTP_BACKOFF    = float(os.getenv("ISA_HTTP_BACKOFF", "0.8"))

EDGE_SKIP_LOG   = bool(int(os.getenv("ISA_EDGE_SKIP_LOG", "0")))   # set 1 for debug prints
ISA_MARK_FAILED_COMPLETE = bool(int(os.getenv("ISA_MARK_FAILED_COMPLETE", "0")))
# if 1: hard failures (Snowstorm unreachable) are treated as leaf nodes for this run

def _log(msg: str):
    if EDGE_SKIP_LOG:
        print(msg)

# =========================
# HTTP (simple, dependency-free)
# =========================
def _http_get_json(url: str, params: Dict[str, Any] | None = None) -> Any:
    q = urllib.parse.urlencode(params or {})
    full = f"{url}?{q}" if q else url
    last_err = None
    for k in range(HTTP_TRIES):
        try:
            with urllib.request.urlopen(full, timeout=HTTP_TIMEOUT_S) as r:
                if r.status == 200:
                    data = r.read()
                    return json.loads(data.decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Snowstorm: invalid/inactive concepts or bad ECL often → 400/404.
            # Treat as "semantically empty" result so callers just see no edges.
            if e.code in (400, 404):
                return {}
            last_err = e
        except Exception as e:
            last_err = e
        time.sleep(HTTP_BACKOFF * (2 ** k))
    if last_err:
        _log(f"[http] GET failed {full}: {last_err}")
        return {"__error": str(last_err)}
    return {"__error": "unknown"}


# 允许通过环境变量调控“自旋重试次数/退避”
FETCH_MAX_SPINS  = int(os.getenv("ISA_FETCH_MAX_SPINS", "8"))        # 本次调用内，最多再试几轮
FETCH_SPIN_BACKOFF = float(os.getenv("ISA_FETCH_SPIN_BACKOFF", "0.5"))  # 每轮递增退避秒

def _spin_until_ok(fetch_url: str, params: Dict[str, Any]) -> Optional[Any]:
    """本次调用内重试，直到 _http_get_json 不再返回 __error；失败到上限则返回 None。"""
    for i in range(FETCH_MAX_SPINS):
        j = _http_get_json(fetch_url, params)
        if not (isinstance(j, dict) and "__error" in j):
            return j  # ← 成功（可能是空结构）
        time.sleep(FETCH_SPIN_BACKOFF * (2 ** i))
    return None  # ← 本次调用内仍失败

def _get_parents_live(branch: str, cid: str, *, form: str) -> Optional[List[Dict[str, Any]]]:
    url = f"{snowstorm_base}/browser/{branch}/concepts/{cid}/parents"
    j = _spin_until_ok(url, {"form": form, "limit": 1000})
    if j is None:
        return None  # 本次调用内失败，不要持久化/不要 complete
    if isinstance(j, list): return j
    if isinstance(j, dict): return j.get("items", []) or []
    return []

def _get_ancestors_live(branch: str, cid: str, *, form: str) -> Optional[List[Dict[str, Any]]]:
    url = f"{snowstorm_base}/browser/{branch}/concepts/{cid}/ancestors"
    j = _spin_until_ok(url, {"form": form, "limit": 5000})
    if j is None:
        return None
    if isinstance(j, list): return j
    if isinstance(j, dict): return j.get("items", []) or []
    return []

def _get_children_live(branch: str, cid: str, *, form: str) -> Optional[List[Dict[str, Any]]]:
    url = f"{snowstorm_base}/browser/{branch}/concepts/{cid}/children"
    j = _spin_until_ok(url, {"form": form, "limit": 1000})
    if j is None:
        return None
    if isinstance(j, list): return j
    if isinstance(j, dict): return j.get("items", []) or []
    return []


# Optional one-shot by id (for filling minimal minis)
def _get_by_id_live(branch: str, cid: str) -> Dict[str, Any]:
    j = _http_get_json(f"{snowstorm_base}/browser/{branch}/concepts/{cid}", {})
    return j if isinstance(j, dict) else {}

# =========================
# Mini conversion
# =========================
def mini_to_row(m: Dict[str, Any]) -> Dict[str, Any]:
    # Same structure as your existing helper; keep it compatible.
    return {
        "conceptId": str(m.get("conceptId") or m.get("concept", {}).get("conceptId") or ""),
        "preferred_term": (m.get("pt", {}) or {}).get("term"),
        "fully_specified_name": (m.get("fsn", {}) or {}).get("term"),
        "active": m.get("active"),
        "definitionStatus": m.get("definitionStatus"),
    }

# =========================
# DB init
# =========================
def _conn():
    c = sqlite3.connect(HIER_CACHE_DB)
    x = c.cursor()
    x.execute("PRAGMA journal_mode=WAL;")
    x.execute("PRAGMA synchronous=OFF;")
    x.execute("""
      CREATE TABLE IF NOT EXISTS children(
        branch TEXT, form TEXT, parent TEXT, child TEXT,
        PRIMARY KEY(branch, form, parent, child)
      )
    """)
    x.execute("""
      CREATE TABLE IF NOT EXISTS parents(
        branch TEXT, form TEXT, child TEXT, parent TEXT,
        PRIMARY KEY(branch, form, child, parent)
      )
    """)
    x.execute("""
      CREATE TABLE IF NOT EXISTS meta(
        branch TEXT, form TEXT, cid TEXT,
        children_complete INTEGER DEFAULT 0,
        parents_complete  INTEGER DEFAULT 0,
        PRIMARY KEY(branch, form, cid)
      )
    """)
    x.execute("""
      CREATE TABLE IF NOT EXISTS mini(
        cid TEXT PRIMARY KEY,
        json TEXT
      )
    """)
    c.commit()
    return c

# =========================
# Meta helpers
# =========================
def _mark_children_complete(cur, branch: str, form: str, cid: str, val: int=1):
    cur.execute("""INSERT INTO meta(branch,form,cid,children_complete)
                   VALUES(?,?,?,?)
                   ON CONFLICT(branch,form,cid)
                   DO UPDATE SET children_complete=excluded.children_complete""",
                (branch, form, cid, int(val)))

def _mark_parents_complete(cur, branch: str, form: str, cid: str, val: int=1):
    cur.execute("""INSERT INTO meta(branch,form,cid,parents_complete)
                   VALUES(?,?,?,?)
                   ON CONFLICT(branch,form,cid)
                   DO UPDATE SET parents_complete=excluded.parents_complete""",
                (branch, form, cid, int(val)))

def _children_complete(cur, branch: str, form: str, cid: str) -> bool:
    cur.execute("SELECT children_complete FROM meta WHERE branch=? AND form=? AND cid=?",
                (branch, form, cid))
    row = cur.fetchone()
    return bool(row and int(row[0])==1)

def _parents_complete(cur, branch: str, form: str, cid: str) -> bool:
    cur.execute("SELECT parents_complete FROM meta WHERE branch=? AND form=? AND cid=?",
                (branch, form, cid))
    row = cur.fetchone()
    return bool(row and int(row[0])==1)

# =========================
# Edge helpers
# =========================
def _put_children(cur, branch: str, form: str, parent: str, minis: List[Dict[str, Any]]):
    for m in minis:
        cid = str(m.get("conceptId") or m.get("concept", {}).get("conceptId") or "")
        if not cid: continue
        cur.execute("INSERT OR IGNORE INTO children(branch,form,parent,child) VALUES (?,?,?,?)",
                    (branch, form, parent, cid))
        # Fill parents reverse edge too
        cur.execute("INSERT OR IGNORE INTO parents(branch,form,child,parent) VALUES (?,?,?,?)",
                    (branch, form, cid, parent))

def _put_parents(cur, branch: str, form: str, child: str, minis: List[Dict[str, Any]]):
    for m in minis:
        pid = str(m.get("conceptId") or m.get("concept", {}).get("conceptId") or "")
        if not pid: continue
        cur.execute("INSERT OR IGNORE INTO parents(branch,form,child,parent) VALUES (?,?,?,?)",
                    (branch, form, child, pid))
        # also write children edge
        cur.execute("INSERT OR IGNORE INTO children(branch,form,parent,child) VALUES (?,?,?,?)",
                    (branch, form, pid, child))

def _children_of(cur, branch: str, form: str, cid: str) -> List[str]:
    cur.execute("SELECT child FROM children WHERE branch=? AND form=? AND parent=?",
                (branch, form, cid))
    return [r[0] for r in cur.fetchall()]

def _parents_of(cur, branch: str, form: str, cid: str) -> List[str]:
    cur.execute("SELECT parent FROM parents WHERE branch=? AND form=? AND child=?",
                (branch, form, cid))
    return [r[0] for r in cur.fetchall()]

# =========================
# Mini memo (optional)
# =========================
def _mini_put(cur, cid: str, mini: Dict[str, Any]):
    try:
        cur.execute("INSERT OR REPLACE INTO mini(cid,json) VALUES(?,?)",
                    (cid, json.dumps(mini_to_row(mini), ensure_ascii=False)))
    except Exception:
        pass

def _mini_get(cur, cid: str) -> Optional[Dict[str, Any]]:
    cur.execute("SELECT json FROM mini WHERE cid=? LIMIT 1", (cid,))
    row = cur.fetchone()
    if not row: return None
    try:
        return json.loads(row[0])
    except Exception:
        return None

# =========================
# Public API
# =========================
def parents_cached(branch: str, form: str, cid: str) -> List[Dict[str, Any]]:
    cid = str(cid)
    c = _conn(); cur = c.cursor()
    if not _parents_complete(cur, branch, form, cid):
        _log(f"[edge-cache] fetch parents({cid})")
        minis = _get_parents_live(branch, cid, form=form)  # ← 可能是 None（失败）/list（成功）
        if minis is None:
            _log(f"[edge-cache] parents({cid}) spin failed; skip persist/complete")
            if ISA_MARK_FAILED_COMPLETE:
                _mark_parents_complete(cur, branch, form, cid, 1)
                c.commit()
        else:
            _put_parents(cur, branch, form, cid, minis or [])
            # 成功（哪怕空）也 complete
            _mark_parents_complete(cur, branch, form, cid, 1)
            for m in (minis or []):
                pid = str(m.get("conceptId") or "")
                if pid: _mini_put(cur, pid, m)
            c.commit()
    else:
        _log(f"[edge-cache] reuse parents({cid})")

    out = []
    for pid in sorted(_parents_of(cur, branch, form, cid)):
        mm = _mini_get(cur, pid)
        out.append(mm if mm is not None else {"conceptId": pid})
    c.close()
    return out

def ancestors_cached(branch: str, form: str, cid: str) -> List[Dict[str, Any]]:
    cid = str(cid)
    c = _conn(); cur = c.cursor()
    out_ids, seen, q = set(), {cid}, [cid]

    while q:
        cur_id = q.pop(0)
        if not _parents_complete(cur, branch, form, cur_id):
            _log(f"[edge-cache] fetch parents({cur_id})")
            minis = _get_parents_live(branch, cur_id, form=form)
            if minis is None:
                _log(f"[edge-cache] parents({cur_id}) spin failed; skip persist/complete")
                if ISA_MARK_FAILED_COMPLETE:
                    _mark_parents_complete(cur, branch, form, cur_id, 1)
                    c.commit()
            else:
                _put_parents(cur, branch, form, cur_id, minis or [])
                _mark_parents_complete(cur, branch, form, cur_id, 1)
                for m in (minis or []):
                    pid = str(m.get("conceptId") or "")
                    if pid: _mini_put(cur, pid, m)
                c.commit()
        else:
            _log(f"[edge-cache] reuse parents({cur_id})")

        for pid in _parents_of(cur, branch, form, cur_id):
            if pid in seen: continue
            seen.add(pid); out_ids.add(pid); q.append(pid)

    minis: List[Dict[str, Any]] = []
    for aid in sorted(out_ids):
        mm = _mini_get(cur, aid)
        minis.append(mm if mm is not None else {"conceptId": aid})
    c.close()
    return minis

def descendants_cached(branch: str, form: str, root: str) -> List[Dict[str, Any]]:
    root = str(root)
    c = _conn(); cur = c.cursor()
    out_ids, seen, q = set(), {root}, [root]

    while q:
        cur_id = q.pop(0)
        if not _children_complete(cur, branch, form, cur_id):
            _log(f"[edge-cache] fetch children({cur_id})")
            minis = _get_children_live(branch, cur_id, form=form)
            if minis is None:
                _log(f"[edge-cache] children({cur_id}) spin failed; skip persist/complete")
                if ISA_MARK_FAILED_COMPLETE:
                    _mark_children_complete(cur, branch, form, cur_id, 1)
                    c.commit()
            else:
                _put_children(cur, branch, form, cur_id, minis or [])
                _mark_children_complete(cur, branch, form, cur_id, 1)
                for m in (minis or []):
                    chid = str(m.get("conceptId") or "")
                    if chid: _mini_put(cur, chid, m)
                c.commit()
        else:
            _log(f"[edge-cache] reuse children({cur_id})")

        for ch in _children_of(cur, branch, form, cur_id):
            if ch in seen: continue
            seen.add(ch); out_ids.add(ch); q.append(ch)

    minis: List[Dict[str, Any]] = []
    for cid in sorted(out_ids):
        mm = _mini_get(cur, cid)
        minis.append(mm if mm is not None else {"conceptId": cid})
    c.close()
    return minis


def children_cached(branch: str, form: str, cid: str) -> List[Dict[str, Any]]:
    cid = str(cid)
    c = _conn(); cur = c.cursor()
    if not _children_complete(cur, branch, form, cid):
        _log(f"[edge-cache] fetch children({cid})")
        minis = _get_children_live(branch, cid, form=form)
        if minis is None:
            _log(f"[edge-cache] children({cid}) spin failed; skip persist/complete")
            if ISA_MARK_FAILED_COMPLETE:
                _mark_children_complete(cur, branch, form, cid, 1)
                c.commit()
        else:
            _put_children(cur, branch, form, cid, minis or [])
            _mark_children_complete(cur, branch, form, cid, 1)
            for m in (minis or []):
                chid = str(m.get("conceptId") or "")
                if chid: _mini_put(cur, chid, m)
            c.commit()
    else:
        _log(f"[edge-cache] reuse children({cid})")

    out: List[Dict[str, Any]] = []
    for ch in sorted(_children_of(cur, branch, form, cid)):
        mm = _mini_get(cur, ch)
        out.append(mm if mm is not None else {"conceptId": ch})
    c.close()
    return out


def descendants_khops_cached(branch: str, form: str, root: str,
                             max_hops: int = 1_000_000,
                             cap: Optional[int] = None
                            ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    BFS 向下，限定最大 hop（根的孩子 hop=1），并可选 cap 限制“总返回数量”的硬上限。
    返回 (minis, hops_map)；hops_map[cid] = hop。
    """
    root = str(root)
    c = _conn(); cur = c.cursor()

    out_ids: List[str] = []
    hops: Dict[str, int] = {}
    seen = {root}
    q: List[Tuple[str,int]] = [(root, 0)]

    # 规范化 cap：<=0 视为不启用
    if cap is not None and cap <= 0:
        cap = None

    while q:
        cur_id, h = q.pop(0)
        # 到达 hop 上限不再扩展
        if h >= max_hops:
            continue

        # 若已达 cap 上限，提前终止整体 BFS
        if cap is not None and len(out_ids) >= cap:
            q.clear()
            break

        if not _children_complete(cur, branch, form, cur_id):
            _log(f"[edge-cache] fetch children({cur_id})")
            minis = _get_children_live(branch, cur_id, form=form)
            if minis is None:
                _log(f"[edge-cache] children({cur_id}) spin failed; skip persist/complete")
                if ISA_MARK_FAILED_COMPLETE:
                    _mark_children_complete(cur, branch, form, cur_id, 1)
                    c.commit()
            else:
                _put_children(cur, branch, form, cur_id, minis or [])
                _mark_children_complete(cur, branch, form, cur_id, 1)
                for m in (minis or []):
                    chid = str(m.get("conceptId") or "")
                    if chid: _mini_put(cur, chid, m)
                c.commit()
        else:
            _log(f"[edge-cache] reuse children({cur_id})")

        for ch in _children_of(cur, branch, form, cur_id):
            if ch in seen:
                continue
            seen.add(ch)
            hop = h + 1
            hops[ch] = hop
            out_ids.append(ch)

            # 每插入一个就检查 cap，达到上限立即整体早停
            if cap is not None and len(out_ids) >= cap:
                q.clear()
                break

            # 只有 hop < max_hops 的节点才继续扩展
            if hop < max_hops:
                q.append((ch, hop))

    minis: List[Dict[str, Any]] = []
    for cid in sorted(out_ids):
        mm = _mini_get(cur, cid)
        minis.append(mm if mm is not None else {"conceptId": cid})
    c.close()
    return minis, hops

# Convenience: fetch-and-memo single concept (optional)
def ensure_mini(cid: str, branch: str|None=None) -> Dict[str, Any]:
    cid = str(cid)
    c = _conn(); cur = c.cursor()
    mm = _mini_get(cur, cid)
    if mm is None:
        doc = _get_by_id_live(branch or snowstorm_branch, cid) or {}
        if doc:
            _mini_put(cur, cid, doc)
            c.commit()
            mm = _mini_get(cur, cid)
    c.close()
    return mm or {"conceptId": cid}
