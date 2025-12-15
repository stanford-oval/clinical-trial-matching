"""Internal: thin Azure OpenAI client used by the typed_policy system.

Reads OPENAI_ENDPOINT + OPENAI_API_KEY from env. Routes gpt-5/o-series
to the 2024-12-01-preview API with max_completion_tokens; everything
else to 2024-08-01-preview with max_tokens + temperature=0.
"""
import json
import os
import re
import urllib.request


def _base_endpoint():
    full = os.environ['OPENAI_ENDPOINT']
    m = re.match(r'(https://[^/]+)/openai/deployments/[^/]+', full)
    return m.group(1) if m else full


def call(prompt: str, model: str = 'gpt-4.1',
         max_tokens: int = 8000, timeout: int = 240) -> str:
    """Send a single user-turn prompt; return the raw response text.

    For gpt-5 / o-* models the budget is auto-bumped to 32k since
    reasoning tokens count against max_completion_tokens.
    """
    is_reasoning = model.startswith('gpt-5') or model.startswith('o')
    if is_reasoning and max_tokens < 32000:
        max_tokens = 32000

    base = _base_endpoint()
    key = os.environ['OPENAI_API_KEY']

    body = {
        'messages': [{'role': 'user', 'content': prompt}],
        'response_format': {'type': 'json_object'},
    }
    if is_reasoning:
        body['max_completion_tokens'] = max_tokens
        api_version = '2024-12-01-preview'
    else:
        body['max_tokens'] = max_tokens
        body['temperature'] = 0
        api_version = '2024-08-01-preview'

    url = f'{base}/openai/deployments/{model}/chat/completions?api-version={api_version}'
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={'api-key': key, 'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())['choices'][0]['message']['content']


def parse_json(s: str) -> dict:
    if not s:
        return {}
    m = re.search(r'\{[\s\S]*\}', s)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}
