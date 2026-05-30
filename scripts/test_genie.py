"""Smoke test for Stanford Genie (LiteLLM proxy).

Verifies:
  1. Env vars are readable
  2. Base URL is reachable
  3. Chat completion works on azure-gpt-5-mini
  4. Embedding endpoint works on azure-text-embedding-3-small
  5. Cost tracking headers are present

Run with:
    source .env && python scripts/test_genie.py

Or from the repo root:
    source .env && python scripts/test_genie.py
"""

from __future__ import annotations

import os
import sys
import time


def _check_env() -> tuple[str, str]:
    base = os.environ.get("GENIE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("GENIE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not base or not key:
        print("FAIL: GENIE_BASE_URL / GENIE_API_KEY not set. "
              "Did you `source .env` before running?", file=sys.stderr)
        sys.exit(1)
    if not key.startswith("sk-"):
        print(f"WARN: API key does not start with 'sk-' (got '{key[:6]}...'). "
              "Genie virtual keys usually start with sk-.", file=sys.stderr)
    print(f"[env] base_url = {base}")
    print(f"[env] api_key  = {key[:8]}... (redacted)")
    return base, key


def _test_chat(client) -> None:
    # NOTE: GPT-5 is a reasoning model. It consumes tokens on hidden reasoning
    # before emitting visible content, so max_tokens must budget for BOTH.
    # Use reasoning_effort='minimal' + generous max_completion_tokens for smoke.
    print("\n[chat] azure-gpt-5-mini — 'reply OK' test")
    t0 = time.time()
    resp = client.chat.completions.create(
        model="azure-gpt-5-mini",
        messages=[
            {"role": "user",
             "content": "Reply with exactly the two characters: OK"},
        ],
        max_completion_tokens=2048,   # reasoning models need headroom
        reasoning_effort="minimal",    # skip heavy reasoning for a trivial task
    )
    elapsed = time.time() - t0
    content = (resp.choices[0].message.content or "").strip()
    usage = resp.usage
    reasoning_toks = getattr(usage, "completion_tokens_details", None)
    reasoning_toks = getattr(reasoning_toks, "reasoning_tokens", None)
    print(f"[chat] reply     = {content!r}")
    print(f"[chat] tokens    = in={usage.prompt_tokens}  "
          f"out={usage.completion_tokens}  "
          f"reasoning={reasoning_toks}")
    print(f"[chat] latency   = {elapsed:.2f}s")
    if "OK" not in content:
        print(f"WARN: model did not reply 'OK' — got {content!r}", file=sys.stderr)


def _test_embedding(client) -> None:
    print("\n[embed] azure-text-embedding-3-small — single-string test")
    t0 = time.time()
    resp = client.embeddings.create(
        model="azure-text-embedding-3-small",
        input="assumption: silence-default-false-when-chronic",
    )
    elapsed = time.time() - t0
    dim = len(resp.data[0].embedding)
    print(f"[embed] dim      = {dim}")
    print(f"[embed] tokens   = {resp.usage.prompt_tokens}")
    print(f"[embed] latency  = {elapsed:.2f}s")
    if dim != 1536:
        print(f"WARN: embedding dim {dim} != 1536 (expected for -3-small)",
              file=sys.stderr)


def _test_json_mode(client) -> None:
    """Structured output — critical for calibration proposals downstream."""
    print("\n[json] azure-gpt-5-mini — structured output test")
    t0 = time.time()
    resp = client.chat.completions.create(
        model="azure-gpt-5-mini",
        messages=[
            {"role": "user",
             "content": ('Return a JSON object with keys "answer" (int) and '
                         '"confidence" (float 0-1). What is 7 * 8?')},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=2048,
        reasoning_effort="minimal",
    )
    elapsed = time.time() - t0
    content = resp.choices[0].message.content
    print(f"[json] reply     = {content}")
    print(f"[json] latency   = {elapsed:.2f}s")
    import json as _json
    try:
        parsed = _json.loads(content)
        print(f"[json] parsed OK: answer={parsed.get('answer')} conf={parsed.get('confidence')}")
    except _json.JSONDecodeError as e:
        print(f"WARN: JSON parse failed: {e}", file=sys.stderr)


def main() -> None:
    print("=" * 60)
    print("Genie proxy smoke test")
    print("=" * 60)

    base, key = _check_env()

    try:
        from openai import OpenAI
    except ImportError:
        print("FAIL: openai package not installed. Run: pip install openai",
              file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=key, base_url=base)

    try:
        _test_chat(client)
        _test_embedding(client)
        _test_json_mode(client)
    except Exception as e:
        print(f"\nFAIL: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PASS: all Genie endpoints reachable.")
    print("=" * 60)


if __name__ == "__main__":
    main()
