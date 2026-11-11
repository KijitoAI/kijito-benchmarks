"""LLM provider layer for the benchmark harnesses: the READER (answering model) and the JUDGE.

Every harness call to an LLM goes through `_llm` (one user prompt, optional system prompt) or `_llm_chat`
(system + user with an output budget), which dispatch to a local Ollama, OpenAI, OpenRouter or Anthropic
backend and return `(text, in_tokens, out_tokens)`. Nothing here talks to Kijito. Keys come from
OPENAI_API_KEY / OPENROUTER_API_KEY / ANTHROPIC_API_KEY; Ollama needs none.

Harness-side environment knobs (all optional; none changes a published configuration):
  BENCH_WORKERS             parallel reader/judge calls (runner.pmap; default 8)
  BENCH_429_RETRIES         retry budget for rate-limit / 5xx answers (default 6)
  BENCH_OPENROUTER_PROVIDER comma list overriding OpenRouter's pinned upstream (default: the model's
                            own vendor for openai/ and anthropic/ ids, fallbacks off)
  BENCH_OLLAMA_THINK        local Ollama reader only: 0/1 sends think:false/true
  BENCH_OLLAMA_TIMEOUT      local Ollama per-call timeout in seconds (default 300)
"""
import os
import random
import re
import time

import requests

# Static price table ($/1M tokens, input/output) — ESTIMATES; harnesses report exact token counts so cost
# is recomputable if prices move. Unlisted (e.g. local) = free.
_PRICES = {
    "gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    # the official STALE judge (stale.py); OpenRouter list price 2026-10-23
    "gemini-3.1-flash-lite-preview": (0.25, 1.50), "gemini-3.1-flash-lite": (0.25, 1.50),
    "claude-opus-5-5": (4.0, 20.0), "claude-opus-4-8": (15.0, 75.0), "claude-sonnet-5": (3.0, 15.0), "claude-haiku-4-5": (1.0, 5.0),
}


def ollama_settings() -> dict:
    """The local reader/judge knobs, for the manifest. Defaults: the model's own thinking default and a
    300 s timeout. BENCH_OLLAMA_THINK=0/1 sends think:false/true; BENCH_OLLAMA_TIMEOUT raises the
    per-call timeout (a shared local Ollama queues requests, so a queued call can exceed 300 s without
    anything being wrong with it)."""
    think = os.environ.get("BENCH_OLLAMA_THINK")
    return {"think": None if think in (None, "") else think.strip().lower() in ("1", "true", "yes", "on"),
            "timeout_s": float(os.environ.get("BENCH_OLLAMA_TIMEOUT") or 300)}


def _ollama_available(model: str) -> bool:
    """True when a local Ollama answers on localhost:11434 and has `model` pulled."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            names = {m.get("name", "") for m in json.loads(r.read()).get("models", [])}
        return model in names
    except Exception:
        return False


def _messages(prompt, system=None):
    """Chat messages for the OpenAI-shaped APIs: an optional system message, then the user prompt."""
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]


def ollama_think():
    """The thinking setting a local Ollama reader runs with (None = the model's default), as recorded
    in the STALE manifest and resume sentinel (BENCH_OLLAMA_THINK, see ollama_settings)."""
    return ollama_settings()["think"]


def _ollama_call(model, prompt, system=None, json_mode=False):
    cfg = ollama_settings()
    body = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    if system:
        body["system"] = system
    if json_mode:
        # Measured on Ollama 0.34.3 + qwen3.5:9b-mlx-bf16 (2026-10-23): with format=json and thinking ON
        # the constrained JSON lands in the `thinking` field and `response` comes back EMPTY. Thinking
        # off is the only combination that returns the JSON as the answer.
        body["format"] = "json"
        body["think"] = False
    elif cfg["think"] is not None:
        body["think"] = cfg["think"]
    r = requests.post("http://localhost:11434/api/generate", json=body, timeout=cfg["timeout_s"])
    r.raise_for_status()
    d = r.json()
    return d.get("response", ""), int(d.get("prompt_eval_count", 0)), int(d.get("eval_count", 0))


def _openai_call(model, prompt, system=None, json_mode=False):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    body = {"model": model, "temperature": 0, "messages": _messages(prompt, system)}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    return d["choices"][0]["message"]["content"], int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))


def _anthropic_call(model, prompt, system=None, json_mode=False):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    # NB: newer Claude models (e.g. claude-opus-4-8) DEPRECATE `temperature` and 400 if it's sent — omit it.
    # 1024: short-answer benchmarks stay short anyway; a longer answer has room to finish.
    # json_mode has no Anthropic wire equivalent; the caller's prompt asks for JSON and parses it.
    body = {"model": model, "max_tokens": 1024, "messages": [{"role": "user", "content": prompt}]}
    if system:
        body["system"] = system
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                      json=body, timeout=120)
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
    return text, int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))


def _openrouter_call(model, prompt, system=None, json_mode=False):
    """OpenRouter speaks the OpenAI chat API. `model` is an OpenRouter id (e.g. openai/gpt-4o-2024-08-06).
    Routing is pinned to the model's own vendor with fallbacks OFF, so a leaderboard judge is the vendor's
    model and never a silent substitute; override with BENCH_OPENROUTER_PROVIDER (comma list) if needed."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    order = os.environ.get("BENCH_OPENROUTER_PROVIDER")
    vendor = {"openai": "OpenAI", "anthropic": "Anthropic"}.get(model.split("/", 1)[0])
    body = {"model": model, "temperature": 0, "messages": _messages(prompt, system)}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if order or vendor:
        body["provider"] = {"order": (order.split(",") if order else [vendor]), "allow_fallbacks": False}
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
    if r.status_code >= 400:
        # Raise with the STATUS FIRST and the body text included: _llm's retry classes read the exception
        # text (a bare raise_for_status() hides OpenRouter's reason, e.g. which upstream rate-limited).
        raise RuntimeError(f"{r.status_code} openrouter HTTP error: {(getattr(r, 'text', '') or '')[:300]}")
    d = r.json()
    if "choices" not in d:   # OpenRouter can answer 200 with an error body; never score it as an empty answer
        err = d.get("error", d)
        code = err.get("code") if isinstance(err, dict) else None
        msg = str(err.get("message", "")) if isinstance(err, dict) else str(err)
        if str(code) == "429" or "rate limit" in msg.lower() or "rate-limit" in msg.lower():
            # starts with 429 => _is_rate_limit => the patient backoff budget, not a 2-retry miss
            raise RuntimeError(f"429 openrouter rate limit (200 error body): {str(err)[:200]}")
        prefix = f"{code} " if isinstance(code, int) else ""   # a 5xx code reaches the transient class
        raise RuntimeError(f"{prefix}openrouter error body: {str(err)[:200]}")
    u = d.get("usage", {})
    content = ((d["choices"] or [{}])[0].get("message") or {}).get("content")
    if content is None or not str(content).strip():
        # An empty completion is a transport/provider failure, not an answer: scored, it would read as
        # an abstention (credited on _abs questions) or a judge 'no'. Raise so the item errors + re-runs.
        fr = (d["choices"] or [{}])[0].get("finish_reason")
        raise RuntimeError(f"openrouter empty completion (finish_reason={fr!r}) for model {model!r}")
    return content, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))


_PROVIDERS = {"ollama": _ollama_call, "openai": _openai_call, "openrouter": _openrouter_call,
              "anthropic": _anthropic_call}


def _is_rate_limit(exc) -> bool:
    """A 429 (rate-limit / quota) — the retry-with-backoff class. Detected from the exception text so it
    works for requests.HTTPError across all three providers."""
    s = str(exc)
    return "429" in s[:40] or "Too Many Requests" in s or "rate_limit" in s


_SERVER_ERROR_RE = re.compile(r"\s*(500|502|503|504)\b")


def _is_server_error(exc) -> bool:
    """A transient 5xx (500/502/503/504) — status-first exception text, as requests.HTTPError
    ("502 Server Error: ...") and _openrouter_call both produce. Retried on the patient budget."""
    return bool(_SERVER_ERROR_RE.match(str(exc)))


def _llm(provider, model, prompt, retries=2, system=None, json_mode=False):
    """Dispatch to a provider with retries. Returns (text, in_tok, out_tok).

    `system` (an optional system message) and `json_mode` (ask the API for a JSON object) are passed
    to the provider ONLY when set, so a provider function written as fn(model, prompt) — every stub in
    the test suite — keeps working, and every existing caller's request body is byte-identical.

    Rate limits (429) get a SEPARATE, more patient budget with exponential backoff + jitter: under
    concurrent benchmark streams a 429 burst is expected and transient, and a swallowed 429 becomes a
    silently-scored miss that corrupts the whole arm (transport failure scored as quality). Other errors
    keep the original small fixed retry. Transient 5xx (500/502/503/504) share the PATIENT budget: a
    provider brown-out lasts longer than two 1.5s/3s retries, and an exhausted retry is a scored miss."""
    kw = {}
    if system:
        kw["system"] = system
    if json_mode:
        kw["json_mode"] = True
    return _call_with_retries(_PROVIDERS[provider], model, prompt, retries, kw)


def _call_with_retries(fn, model, prompt, retries=2, kw=None):
    """The retry / 429+5xx-backoff loop shared by `_llm` and `_llm_chat`."""
    kw = kw or {}
    rl_budget = int(os.environ.get("BENCH_429_RETRIES", "6"))
    attempt = rl_attempt = 0
    while True:
        try:
            return fn(model, prompt, **kw)
        except Exception as e:
            if _is_rate_limit(e) or _is_server_error(e):
                rl_attempt += 1
                if rl_attempt > rl_budget:
                    raise
                # 2s, 4s, 8s ... capped at 60s, plus jitter so parallel workers desynchronize
                time.sleep(min(2 ** rl_attempt, 60) + random.uniform(0, 1.5))
                continue
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(1.5 * attempt)


def _price_key(model):
    """openai/gpt-4o-2024-08-06 -> gpt-4o: an OpenRouter id or a dated snapshot prices as its base model."""
    m = model.split("/", 1)[-1]
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", m)


def _cost(model, in_tok, out_tok):
    pin, pout = _PRICES.get(model) or _PRICES.get(_price_key(model), (0.0, 0.0))
    return in_tok / 1e6 * pin + out_tok / 1e6 * pout


# ── the harness tokenizer (context-size instrumentation) ───────────────────────────────────────
# ONE tokenizer for every "tokens of retrieved context handed to the reader" number, so two context
# sizes are never measured on different rulers. tiktoken's o200k_base (the gpt-4o/-mini vocabulary) when importable; otherwise a
# DISCLOSED deterministic fallback (ceil(chars/4)) — every consumer records tokenizer_name() next
# to the numbers it produced, so a venue without tiktoken announces itself in the scorecard
# instead of silently shifting the scale.
_tokenizer = None
_tokenizer_name = None


def _resolve_tokenizer():
    global _tokenizer, _tokenizer_name
    if _tokenizer_name is None:
        try:
            import tiktoken
            _tokenizer = tiktoken.get_encoding("o200k_base")
            _tokenizer_name = "tiktoken/o200k_base"
        except Exception:
            _tokenizer = None
            _tokenizer_name = "chars/4-approx (tiktoken unavailable)"
    return _tokenizer


def tokenizer_name() -> str:
    _resolve_tokenizer()
    return _tokenizer_name


def context_tokens(text: str) -> int:
    """Token count of `text` under the harness tokenizer (see above). Empty => 0."""
    if not text:
        return 0
    tok = _resolve_tokenizer()
    if tok is not None:
        return len(tok.encode(text))
    return max(1, -(-len(text) // 4))   # ceil(chars/4), deterministic + disclosed


# ── system+user chat calls (GroupMemBench's official reader/judge shape) ─────────────────────────
# Some official protocols (GroupMemBench: prompts/hipporag_{agent,judge}_system.txt) send a SYSTEM
# prompt plus a USER message with a per-call output budget, not one user prompt. These providers take
# a payload dict {"system", "user", "max_tokens"} and share `_llm`'s retry / 429-backoff loop.
# Budget + temperature mirror GroupMemBench llm_utils.build_chat_completion_kwargs: a reasoning model
# (gpt-5 / o1 / o3 / o4, judged on the id after any vendor prefix) gets NO temperature and
# max_completion_tokens = max(4 * max_tokens, 2000); any other model gets temperature 0.2 and
# max_tokens as given.

def _is_reasoning_model(model: str) -> bool:
    return (model or "").split("/", 1)[-1].strip().lower().startswith(("gpt-5", "o1", "o3", "o4"))


def _chat_budget(model, max_tokens):
    if max_tokens is None:
        return {}
    if _is_reasoning_model(model):
        return {"max_completion_tokens": max(int(max_tokens) * 4, 2000)}
    return {"max_tokens": int(max_tokens), "temperature": 0.2}


def _chat_messages(payload):
    msgs = []
    if payload.get("system"):
        msgs.append({"role": "system", "content": payload["system"]})
    msgs.append({"role": "user", "content": payload["user"]})
    return msgs


def _ollama_chat(model, payload):
    b = _chat_budget(model, payload.get("max_tokens"))
    opts = {"temperature": b.get("temperature", 0.2)}
    n = b.get("max_tokens") or b.get("max_completion_tokens")
    if n:
        opts["num_predict"] = n
    # think=False: a local thinking model (qwen3.5) would otherwise spend the whole output budget in a
    # hidden thinking channel and return empty content. Local runs are directional, never leaderboard.
    r = requests.post("http://localhost:11434/api/chat",
                      json={"model": model, "messages": _chat_messages(payload), "stream": False,
                            "think": False, "options": opts}, timeout=600)
    r.raise_for_status()
    d = r.json()
    return ((d.get("message") or {}).get("content", ""), int(d.get("prompt_eval_count", 0)),
            int(d.get("eval_count", 0)))


def _openai_compatible_chat(url, key, model, payload, extra=None):
    body = {"model": model, "messages": _chat_messages(payload), **_chat_budget(model, payload.get("max_tokens"))}
    body.update(extra or {})
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=300)
    r.raise_for_status()
    d = r.json()
    if "choices" not in d:   # an error body must never be scored as an empty answer
        raise RuntimeError(f"chat error body: {str(d.get('error', d))[:200]}")
    u = d.get("usage", {})
    return (d["choices"][0]["message"].get("content") or "", int(u.get("prompt_tokens", 0)),
            int(u.get("completion_tokens", 0)))


def _openai_chat(model, payload):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return _openai_compatible_chat("https://api.openai.com/v1/chat/completions", key, model, payload)


def _openrouter_chat(model, payload):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    order = os.environ.get("BENCH_OPENROUTER_PROVIDER")
    vendor = {"openai": "OpenAI", "anthropic": "Anthropic"}.get(model.split("/", 1)[0])
    extra = {}
    if order or vendor:
        extra["provider"] = {"order": (order.split(",") if order else [vendor]), "allow_fallbacks": False}
    return _openai_compatible_chat("https://openrouter.ai/api/v1/chat/completions", key, model, payload, extra)


def _anthropic_chat(model, payload):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    b = _chat_budget(model, payload.get("max_tokens"))
    body = {"model": model, "max_tokens": b.get("max_tokens") or b.get("max_completion_tokens") or 1024,
            "messages": [{"role": "user", "content": payload["user"]}]}
    if payload.get("system"):
        body["system"] = payload["system"]
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01"}, json=body, timeout=300)
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    text = "".join(c.get("text", "") for c in d.get("content", []) if c.get("type") == "text")
    return text, int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))


_CHAT_PROVIDERS = {"ollama": _ollama_chat, "openai": _openai_chat, "openrouter": _openrouter_chat,
                   "anthropic": _anthropic_chat}


def _llm_chat(provider, model, system, user, max_tokens=None, retries=2):
    """System+user chat call with `_llm`'s retry policy. Returns (text, in_tok, out_tok)."""
    return _call_with_retries(_CHAT_PROVIDERS[provider], model,
                              {"system": system, "user": user, "max_tokens": max_tokens}, retries)
