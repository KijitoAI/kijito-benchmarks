"""STALE — can a memory system tell when a memory is no longer valid? (arXiv 2605.06527)

STALE (github.com/icedreamc/STALE, dataset huggingface.co/datasets/STALEproj/STALE, CC BY 4.0) is
400 scenarios (200 Type I "co-referential", 200 Type II "propagated"), each a 50-session chat
history (~593 turns, ~152K tokens) in which a LATER observation (M_new) implicitly invalidates an
EARLIER one (M_old) without ever saying so. Distractor sessions are sampled from LongMemEval. Each
scenario carries three probes, issued separately against the same memory:

  dim1  State Resolution (SR)          "does the user still live in Seattle?"
  dim2  Premise Resistance (PR)         a question built on the stale premise
  dim3  Implicit Policy Adaptation (IPA) a task whose right answer depends on the new state

OFFICIAL PROTOCOL (what this harness reproduces; sources: paper §4.1 + Table 2 caption, repo
STALE/Evaluation/run_target_model.py + full_eval_performance.py + judge_prompts.py @ ea7d391):
- Memory systems "ingest the dialogue history once per instance according to [their] native
  protocol", then "issue the three probing queries separately against the same constructed memory"
  (fixed during probing). Backbone (reader) LLM for every memory system row, CUPMem included:
  GPT-4o-mini.
- Reader prompt: the official target-model prompts below, VERBATIM. The full-context LLM rows put
  the whole serialized haystack in `[Conversation History]`; a memory system puts what it retrieved
  there instead (here: Kijito's top-k memories, each stamped with its session time).
- Judge: Gemini-3.1-flash-lite (repo default id `gemini-3.1-flash-lite-preview`), temperature 0,
  JSON mode, ONE call per scenario grading all three responses against M_old / M_new / explanation
  with the verbatim `SYSTEM_PROMPT_ALL_IN_ONE_JUDGE`. Paper: 95.8% agreement with humans.
- Metrics: accuracy per (type, dim) = six settings; Overall = the MEAN OF THE SIX settings
  (Table 2 caption). The repo's per-type "overall" (correct/total over the three dims) is also
  reported.

DELIBERATE, DISCLOSED DIFFERENCES from the official scripts (each recorded in the manifest):
- A judge API failure is an ERROR here, never a silent fail-score (official code scores it false);
  any error makes the scorecard status "invalid".
- If the official JSON parse fails (fence-strip + json.loads) we try the outermost {...} once and
  count it as `judge_parse_fallbacks`.
- Reader temperature is 0 (official sends none = provider default), for reproducibility.

THE KIJITO ARM, through Kijito's public API (--via-api, default https://api.kijito.ai, plus
--api-key-file). Every non-empty turn becomes one note, '[<session time>] <role>: <content>', sessions in
chronological order; the notes are written with kijito_remember_batch (<= 60 items per call) into ONE
PROJECT PER SCENARIO (<prefix><uid>) of one account, and each probe is recalled with kijito_recall
(scope='project', project=<that project>, limit=k). The reader's `[Conversation History]` is the
kijito_recall reply text EXACTLY as an agent reads it (--api-ctx mcp_text, the default; --api-ctx
contents renders the parsed bodies as '- <note>' lines, to separate formatting from retrieval). Nothing
about the gold (M_old, M_new, relevant_session_index) ever reaches the system. No correction is issued:
STALE's conflicts are implicit, so plain ingest is the faithful arm. A scope canary runs before the first
live scenario and every recall is leak-checked (a memory from another project marks the run invalid).
--blind declares that the key's ACCOUNT is enrolled in the keyword index over encrypted content (an
account setting, not per project; GET /api/fts-blind/status is checked and recorded).

Run: python -m kijito_bench.stale --n 5 --api-key-file KEYFILE --api-project-prefix stale-run1- --out sc.json
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

from .llm import _llm, _cost, _ollama_available, _price_key, context_tokens, ollama_think, tokenizer_name
from .runner import ContextCache, ItemLog, pmap, run_manifest

DEFAULT_API = "https://api.kijito.ai"

_HF_REPO = "STALEproj/STALE"
_HF_FILE = "T1_T2_400_FULL.json"
_HF_REVISION = "617c51dc200b5ab09970834144c7e51c77959af0"      # dataset repo sha, 2026-05-07
_PUBLISHED_SHA256 = "5f3ec375179e20e2e94469e018189188f34e2e7e5f21cbecbd99fcfa648c1876"
_UPSTREAM_CODE = "github.com/icedreamc/STALE@ea7d391103a151927cd29d2f01d87597a782bdcb"
_SPLIT = "full"            # the dataset ships ONE file (400 scenarios); there is no dev/test split
_INGEST_VERSION = "v1-per-session-batch-turn-notes"   # bump when the note shape / batching changes
_DIMS = ("dim1", "dim2", "dim3")
_PROBE = {"dim1": "SR", "dim2": "PR", "dim3": "IPA"}
_TYPES = ("T1", "T2")

# The configuration the published memory-system rows used.
_OFFICIAL_READER = "gpt-4o-mini"                              # paper §4.1, every memory-system row
_OFFICIAL_JUDGES = ("gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite")   # repo default / paper name

# Published Table 2 (paper) = repo README "Main Results", accuracy %, (T1 SR, T1 PR, T1 IPA, T2 SR,
# T2 PR, T2 IPA, Overall). Memory systems all use the GPT-4o-mini reader. `*` = truncated context.
PUBLISHED = {
    "memory_systems_gpt-4o-mini": {
        "LightMem": (52.5, 1.0, 23.5, 21.5, 0.5, 7.5, 17.8),
        "Zep": (10.0, 0.0, 19.0, 3.0, 1.0, 3.0, 6.0),
        "LiCoMemory": (15.5, 0.5, 22.5, 1.5, 1.5, 4.0, 7.6),
        "A-mem": (13.5, 0.0, 7.5, 8.0, 0.0, 1.5, 5.1),
        "mem-0": (17.0, 1.0, 22.0, 3.5, 0.0, 6.5, 8.3),
        "CUPMem": (91.0, 78.0, 32.0, 89.0, 75.0, 43.0, 68.0),
    },
    "full_context_llms": {
        "GPT-4o-mini*": (30.0, 0.0, 11.0, 9.5, 0.0, 1.5, 8.7),
        "Gemini-3.1-pro": (92.0, 30.0, 71.0, 69.0, 14.0, 55.0, 55.2),
        "Claude-Opus-4.6": (50.5, 10.0, 42.5, 21.0, 9.0, 36.5, 28.3),
        "Qwen3.5-9B": (36.0, 1.0, 21.5, 21.5, 0.0, 7.5, 14.6),
        "Qwen3.5-27B": (76.0, 4.0, 39.0, 42.0, 3.5, 23.0, 31.3),
    },
}

# ── official prompts, VERBATIM (STALE/Evaluation @ ea7d391) ───────────────────────────────────────
# run_target_model.py build_prompts(): dim3 gets the "respond to the latest query" variant.
_READER_SYSTEM = {
    "dim1": ("You are a helpful assistant. Review the following conversation history "
             "with the user, then accurately answer the question."),
    "dim3": ("You are a helpful assistant. Review the following conversation history "
             "with the user, then respond to the user's latest query directly."),
}
_READER_SYSTEM["dim2"] = _READER_SYSTEM["dim1"]


def _reader_user_prompt(history_text: str, query_text: str, dim: str) -> str:
    if dim == "dim3":
        return f"[Conversation History]\n{history_text}\n\n[Latest Query]\n{query_text}"
    return f"[Conversation History]\n{history_text}\n\n[Question]\n{query_text}"


# judge_prompts.py SYSTEM_PROMPT_ALL_IN_ONE_JUDGE (trailing spaces included). sha256 pinned in tests.
JUDGE_SYSTEM = (
    '\n'
    'You are a strict and impartial expert evaluator for an AI Assistant Benchmark focusing on Long-Term Memory and Implicit Reasoning.\n'
    '\n'
    "You will be provided with the Ground Truth Context of a user's state. \n"
    'Then, you will evaluate THREE distinct responses generated by a Target AI Model to THREE different probing questions (Dimension 1, 2, and 3).\n'
    '\n'
    '[Ground Truth Context]:\n'
    "- M_old: The user's outdated state.\n"
    "- M_new: The user's updated state.\n"
    '- Hidden Logic: How M_new invalidates M_old.\n'
    '\n'
    'You must grade each response strictly based on whether it correctly deduces and respects the hidden updated state.\n'
    '\n'
    '### Scoring Rubrics (Boolean: true for Pass, false for Fail)\n'
    '\n'
    '* **Dimension 1 (Explicit Probing):** Score `true` ONLY IF Response 1 awares that the information in M_old is possibly NO LONGER VALID. Score `false` if it assumes M_old is still true or claims ignorance.\n'
    '\n'
    '* **Dimension 2 (Adversarial Robustness):**\n'
    '  Question 2 contains a FALSE PREMISE based on M_old. Score `true` ONLY IF Response 2 detects the trap, or acknowledges the new state in M_new. Score `false` if it blindly agrees with the false premise.\n'
    '\n'
    '* **Dimension 3 (Implicit Task):**\n'
    '  Score `true` ONLY IF the final action, plan, or recommendation in Response 3 is safely adheres to the NEW state established by M_new.  \n'
    '  Score `false` if the action complies with the outdated M_old constraints, OR if the response is too generic, vague.\n'
    '\n'
    'Evaluate strictly based on the underlying logic, not surface-level word matching.\n'
    '\n'
    '### Output Format (JSON)\n'
    '{\n'
    '  "dim1_eval": {\n'
    '    "reasoning": "Brief step-by-step reasoning against the Dim 1 rubric.",\n'
    '    "pass": true/false\n'
    '  },\n'
    '  "dim2_eval": {\n'
    '    "reasoning": "Brief step-by-step reasoning against the Dim 2 rubric.",\n'
    '    "pass": true/false\n'
    '  },\n'
    '  "dim3_eval": {\n'
    '    "reasoning": "Brief step-by-step reasoning against the Dim 3 rubric.",\n'
    '    "pass": true/false\n'
    '  }\n'
    '}\n'
)
JUDGE_SYSTEM_SHA256 = "2d2d0a52992980120bde7e297197f03853255fa6f36fa3c70df7215476fbe669"


def judge_user_prompt(item: dict, answers: dict) -> str:
    """full_eval_performance.py async_evaluate_all_dimensions() user_prompt, verbatim."""
    pq = item.get("probing_queries") or {}
    old_info = item.get("old_info", item.get("M_old", ""))
    return f"""
[Ground Truth Context]
- M_old: "{old_info}"
- M_new: "{item.get('M_new', '')}"
- Hidden Logic: {item.get('explanation', '')}

--------------------------------------------------
[Dimension 1: Explicit Probing]
Question 1: {pq.get('dim1_query', '')}
Target Model Response 1: {answers.get('dim1', '')}

--------------------------------------------------
[Dimension 2: Adversarial Robustness]
Question 2: {pq.get('dim2_query', '')}
Target Model Response 2: {answers.get('dim2', '')}

--------------------------------------------------
[Dimension 3: Implicit Task]
Question 3: {pq.get('dim3_query', '')}
Target Model Response 3: {answers.get('dim3', '')}
"""


def parse_judge(content: str):
    """(result_dict, used_fallback). The official parse first: strip a ```json fence, json.loads.
    Only if that fails, the outermost {...} once (a disclosed leniency, counted by the caller).
    Raises ValueError when neither yields a dict with the three dimN_eval entries."""
    text = content or ""
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    body = m.group(1) if m else text
    fallback = False
    try:
        res = json.loads(body)
    except Exception:
        a, b = text.find("{"), text.rfind("}")
        if a < 0 or b <= a:
            raise ValueError(f"judge output is not JSON: {text[:160]!r}")
        res = json.loads(text[a:b + 1])
        fallback = True
    if not isinstance(res, dict) or not all(isinstance(res.get(f"{d}_eval"), dict) for d in _DIMS):
        raise ValueError(f"judge JSON lacks dim1_eval/dim2_eval/dim3_eval: {str(res)[:160]!r}")
    return res, fallback


def _judge_pass(res: dict, dim: str) -> bool:
    """The official verdict read: bool(eval_res[dim_eval]['pass']). A string "false" would be truthy
    under bool(), so a string is read by value — the only deviation, and only for non-bool output."""
    v = (res.get(f"{dim}_eval") or {}).get("pass", False)
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


# ── dataset ─────────────────────────────────────────────────────────────────────────────────────

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate(item: dict) -> None:
    uid = item.get("uid")
    if not uid:
        raise ValueError("record missing uid")
    for fld in ("M_new", "haystack_session", "timestamps", "probing_queries"):
        if fld not in item:
            raise ValueError(f"record {uid} missing {fld}")
    if not (item.get("M_old") or item.get("old_info")):
        raise ValueError(f"record {uid} missing M_old")
    for d in _DIMS:
        if not isinstance(item["probing_queries"].get(f"{d}_query"), str):
            raise ValueError(f"record {uid} missing probing_queries.{d}_query")
    if len(item["timestamps"]) != len(item["haystack_session"]):
        raise ValueError(f"record {uid}: {len(item['timestamps'])} timestamps for "
                         f"{len(item['haystack_session'])} sessions")


def load_dataset(data_path: str = None):
    """(records, path, sha256). Default: the pinned HF revision of the one published file."""
    if not data_path:
        from huggingface_hub import hf_hub_download
        data_path = hf_hub_download(_HF_REPO, _HF_FILE, repo_type="dataset", revision=_HF_REVISION)
    with open(data_path, encoding="utf-8") as f:
        obj = json.load(f)
    recs = obj["data"] if isinstance(obj, dict) and isinstance(obj.get("data"), list) else obj
    if not isinstance(recs, list):
        raise ValueError(f"{data_path}: expected a JSON list (or {{'data': [...]}})")
    for r in recs:
        _validate(r)
    uids = [r["uid"] for r in recs]
    if len(set(uids)) != len(uids):
        raise ValueError(f"{data_path}: duplicate uids")
    return recs, data_path, _sha256(data_path)


def stratified_sample(data: list, n: int, seed: int) -> list:
    """Deterministic round-robin across conflict types (T1/T2) so a small --n covers both types.
    n<=0 or n>=len returns everything in file order."""
    if n <= 0 or n >= len(data):
        return list(data)
    by_type: dict = {}
    for r in data:
        by_type.setdefault(r.get("type", "?"), []).append(r)
    rng = random.Random(seed)
    for v in by_type.values():
        rng.shuffle(v)
    picked, types, i = [], sorted(by_type), 0
    while len(picked) < n and any(by_type.values()):
        t = types[i % len(types)]
        if by_type[t]:
            picked.append(by_type[t].pop())
        i += 1
    return picked


def session_notes(item: dict) -> list:
    """[[note, ...] per session] in CHRONOLOGICAL order (stable sort on the session timestamp; the
    published file is already sorted). One note per non-empty turn, '[<session time>] <role>:
    <content>' — the longmemeval._turn_notes shape, so the date travels inside the memory text."""
    sessions = item.get("haystack_session") or []
    stamps = item.get("timestamps") or []
    order = sorted(range(len(sessions)), key=lambda i: str(stamps[i]) if i < len(stamps) else "")
    out = []
    for i in order:
        ts = stamps[i] if i < len(stamps) else ""
        notes = []
        for turn in sessions[i] or []:
            content = (turn.get("content") or "").strip()
            if content:
                notes.append(f"[{ts}] {turn.get('role', '?')}: {content}")
        if notes:
            out.append(notes)
    return out


# ── the Kijito arm ──────────────────────────────────────────────────────────────────────────────

_BLIND = {"on": False}   # set by run(blind=...): the key's account is enrolled in the keyword index


def _blind_on() -> bool:
    return _BLIND["on"]


def _cache_key(cache, uid, dim, arm, k, digest, dataset_sha, api_tag=None):
    if cache is None:
        return None
    extra = (f"ds={dataset_sha}|split={_SPLIT}|ingest={_INGEST_VERSION}"
             f"|blind={'on' if _blind_on() else 'off'}")
    # The retrieval came from a SERVER (host + its /api/version commit + the project namespace + the
    # recall reply form).
    if api_tag:
        extra += f"|api={api_tag}"
    return ContextCache.make_key("stale", f"{uid}:{dim}", arm, k, digest, extra=extra)


# ── the Kijito arm through the public API ─────────────────────────────────────────────────────────
# Cache value shape: [<the kijito_recall reply text>, <json meta>] per (uid, dim). How the
# reply reaches the reader is decided at scoring time (--api-ctx), so one retrieval serves both
# renderings. meta carries the project, the recalled ids and the leak verdict, so a cache hit
# re-asserts the leak check instead of forgetting it. A completed ingest is also cached as an INGEST
# RECORD (arm 'kijito_api_ingest', keyed by host|commit|prefix without the reply form) holding the
# project and the ids written, so the same ingest can be read again in another reply form.

_API_CANARY_STATE = {"api": None}


class _APIRefused(RuntimeError):
    """The scope canary refused the run (project-scoped recall crossed projects)."""


def scenario_notes_flat(item: dict) -> list:
    """Every note of a scenario in ingest order (sessions chronological, turns in order)."""
    return [n for notes in session_notes(item) for n in notes]


def kijito_api_contexts(items, k, api, cache=None, digest=None, dataset_sha=None, api_tag=None,
                        ingest_tag=None, project_prefix="", explain=False, trace=None, leaks=None,
                        timings=None, ingest_stats=None):
    """({(uid, dim): [reply_text, meta_json]}, {(uid, dim): error}) through Kijito's MCP endpoint.
    Per scenario: refuse a project that already holds memories, write every note in chronological
    order with kijito_remember_batch, recall the three probes in dim order (kijito_recall,
    scope='project', limit=k), leak-check each reply. A scenario is served
    from the cache only when ALL THREE probes hit; a failure is an error, never cached."""
    from .client import MemoryCapReached, leak_check, parse_recall, scope_canary
    out, errors, misses = {}, {}, []
    if ingest_stats is not None:
        ingest_stats.setdefault("fresh", 0)
        ingest_stats.setdefault("reused", 0)
    for it in items:
        keys = {d: _cache_key(cache, it["uid"], d, "kijito", k, digest, dataset_sha, api_tag=api_tag)
                for d in _DIMS}
        hits = {d: (cache.get(keys[d]) if cache is not None else None) for d in _DIMS}
        if all(h is not None and len(h) == 2 for h in hits.values()):
            for d in _DIMS:
                out[(it["uid"], d)] = hits[d]
                meta = json.loads(hits[d][1])
                if meta.get("leaks") and leaks is not None:
                    leaks[f"{it['uid']}:{d}"] = meta["leaks"]
        else:
            misses.append((it, keys))
    if not misses:
        return out, errors
    scope_canary(api, project_prefix, exc=_APIRefused, label="stale")
    _API_CANARY_STATE["api"] = "passed"
    cap_error = None
    for n_done, (it, keys) in enumerate(misses, 1):
        if cap_error is not None:
            for d in _DIMS:
                errors[(it["uid"], d)] = cap_error
            continue
        uid = it["uid"]
        project = f"{project_prefix}{uid}"
        calls0, retries0 = api.stats.calls, api.stats.retries
        ikey = (_cache_key(cache, uid, "ingest", "kijito_api_ingest", 0, digest, dataset_sha,
                           api_tag=ingest_tag) if (cache is not None and ingest_tag) else None)
        irec = cache.peek(ikey) if ikey else None
        notes = scenario_notes_flat(it)
        t0 = time.time()
        reused = False
        try:
            if irec:
                im = json.loads(irec[0])
                if im.get("project") != project or im.get("n_notes") != len(notes):
                    raise RuntimeError(f"ingest record for {uid} does not match this scenario "
                                       f"({im.get('project')!r}, {im.get('n_notes')} notes)")
                ids, reused = list(im["own_ids"]), True
            else:
                if parse_recall(api.recall("benchmark namespace probe", limit=1, scope="project",
                                           project=project)):
                    raise RuntimeError(f"project {project!r} already holds memories (an earlier run?); "
                                       f"use a fresh --api-project-prefix")
                ids = api.remember_batch(notes, project) if notes else []
                if ikey:
                    cache.put(ikey, [json.dumps({"project": project, "own_ids": ids, "n_notes": len(notes)})])
        except Exception as e:
            for d in _DIMS:
                errors[(uid, d)] = f"ingest: {type(e).__name__}: {e}"
            print(f"[stale] API ingest failed for {uid}: {type(e).__name__}: {e}", file=sys.stderr)
            if isinstance(e, MemoryCapReached):
                # The account is full: every later write would be refused too. The remaining scenarios
                # are recorded as retrieval errors without writing (the run is invalid; see the README).
                cap_error = f"ingest: {type(e).__name__}: {e}"
                print("[stale] STOPPED: the account's memory cap is reached (see the README: memory "
                      "footprint per arm).", file=sys.stderr)
            continue
        t1 = time.time()
        own = set(ids)
        rec_t = {}
        recalled = {}
        for d in _DIMS:
            q = it["probing_queries"][f"{d}_query"]
            tr = time.time()
            try:
                text = api.recall(q, limit=k, scope="project", project=project, explain=explain)
                got = parse_recall(text)
                q_leaks = leak_check(api, got, own, project)
            except Exception as e:
                errors[(uid, d)] = f"recall: {type(e).__name__}: {e}"
                print(f"[stale] API recall failed for {uid}/{d}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            rec_t[d] = round(time.time() - tr, 3)
            if q_leaks:
                print(f"[stale] LEAK: {uid}/{d} recall returned memories from outside project {project!r}: "
                      f"{q_leaks} (run will be marked INVALID)", file=sys.stderr)
                if leaks is not None:
                    leaks[f"{uid}:{d}"] = {str(i): v for i, v in q_leaks.items()}
            meta = {"uid": uid, "dim": d, "project": project, "recalled_ids": [g.id for g in got],
                    "leaks": {str(i): v for i, v in q_leaks.items()}, "recall_explain": bool(explain),
                    "reply_form": sorted({g.form for g in got}), "ingest_reused": reused}
            ctx = [text, json.dumps(meta)]
            out[(uid, d)] = ctx
            recalled[d] = meta["recalled_ids"]
            if cache is not None:
                cache.put(keys[d], ctx)
        t_rec = {"n_notes": len(notes), "ingest_s": round(t1 - t0, 3), "recall_s": rec_t,
                 "total_s": round(time.time() - t0, 3), "ingest_reused": reused}
        if timings is not None:
            timings[uid] = t_rec
        if ingest_stats is not None:
            ingest_stats["reused" if reused else "fresh"] += 1
        if trace is not None:
            trace.append({"uid": uid, "project": project, **t_rec, "n_ids": len(ids), "recalled": recalled,
                          "leaks": {d: json.loads(out[(uid, d)][1])["leaks"] for d in _DIMS if (uid, d) in out},
                          "calls": api.stats.calls - calls0, "retries": api.stats.retries - retries0})
        print(f"[stale] API {n_done}/{len(misses)} {uid} {'reused ingest' if reused else f'ingested {len(ids)} notes'}, "
              f"recalled 3 probes in {time.time() - t0:.1f}s", file=sys.stderr)
    return out, errors


def api_history(ctx, api_ctx: str, ctx_order: str = "rank"):
    """(history_text, n_memories) for an API retrieval. mcp_text = the reply verbatim (what an agent
    reads; an empty reply is the server's own "No memories found." text); contents = the parsed
    bodies rendered as '- <note>' lines (render_history), honouring --ctx-order."""
    from .client import parse_recall, reply_bodies
    text = ctx[0] if ctx else ""
    if api_ctx == "contents":
        bodies = reply_bodies(text)
        return render_history(bodies, ctx_order), len(bodies)
    n_mem = len(parse_recall(text))
    return (text or "(no relevant memories found)"), n_mem


_TS_RE = re.compile(r"^\[([^\]]*)\]")


def render_history(contexts: list, order: str = "rank") -> str:
    """The retrieved notes as the `[Conversation History]` block. order='rank' keeps the system's
    own ranking (default — no harness help); 'chrono' sorts by the session time in the note."""
    ctx = list(contexts or [])
    if order == "chrono":
        ctx = sorted(ctx, key=lambda c: (_TS_RE.match(c).group(1) if _TS_RE.match(c) else ""))
    return "\n".join(f"- {c}" for c in ctx) or "(no relevant memories found)"


# ── scoring ─────────────────────────────────────────────────────────────────────────────────────

def _score_one(it, ctxs, errs, reader, judge, ctx_order, render=None):
    """One scenario: three separate reader calls, one judge call. Returns the item record.
    `render(ctx) -> (history_text, n_memories)` replaces render_history (the Kijito arm passes api_history)."""
    uid = it["uid"]
    rec = {"uid": uid, "type": it.get("type", "?"), "pass": None, "answers": {}, "reasoning": {},
           "ans_in": 0, "ans_out": 0, "ans_calls": 0, "judge_in": 0, "judge_out": 0, "judge_calls": 0,
           "ctx_tokens": {}, "n_ctx": {}, "judge_parse_fallback": False, "error": None, "error_stage": None}
    bad = [errs[(uid, d)] for d in _DIMS if (uid, d) in errs]
    if bad:
        rec.update(error=bad[0], error_stage="retrieval")
        return rec
    try:
        for d in _DIMS:
            if render is not None:
                hist, n_ctx = render(ctxs[(uid, d)])
            else:
                hist, n_ctx = render_history(ctxs[(uid, d)], ctx_order), len(ctxs[(uid, d)])
            rec["ctx_tokens"][d] = context_tokens(hist)
            rec["n_ctx"][d] = n_ctx
            text, i, o = _llm(reader[0], reader[1],
                              _reader_user_prompt(hist, it["probing_queries"][f"{d}_query"], d),
                              system=_READER_SYSTEM[d])
            rec["answers"][d] = text or ""
            rec["ans_in"] += i; rec["ans_out"] += o; rec["ans_calls"] += 1
    except Exception as e:
        rec.update(error=f"{type(e).__name__}: {e}", error_stage="reader")
        return rec
    try:
        text, i, o = _llm(judge[0], judge[1], judge_user_prompt(it, rec["answers"]),
                          system=JUDGE_SYSTEM, json_mode=True)
        rec["judge_in"] += i; rec["judge_out"] += o; rec["judge_calls"] += 1
        res, fb = parse_judge(text)
    except Exception as e:
        rec.update(error=f"{type(e).__name__}: {e}", error_stage="judge")
        return rec
    rec["judge_parse_fallback"] = fb
    rec["pass"] = {d: _judge_pass(res, d) for d in _DIMS}
    rec["reasoning"] = {d: str((res.get(f"{d}_eval") or {}).get("reasoning", ""))[:2000] for d in _DIMS}
    return rec


def score_items(items, ctxs, errs, reader, judge, ctx_order="rank", arm="kijito",
                item_log=None, resume=None, render=None):
    """Score every scenario (parallel over scenarios via pmap). A logged record WITHOUT an error is
    reused on resume; an errored one is re-attempted (an error is not an outcome)."""
    resume = resume or {}

    def _one(it):
        prev = resume.get((it["uid"], arm))
        if prev is not None and not prev.get("error") and isinstance(prev.get("pass"), dict):
            r = dict(prev)
            r["_resumed"] = True
            return r
        r = _score_one(it, ctxs, errs, reader, judge, ctx_order, render=render)
        if item_log is not None:
            ok = bool(r["pass"]) and all(r["pass"].values())
            extra = {k: v for k, v in r.items() if k not in ("uid", "type", "error")}
            item_log.append(qid=r["uid"], arm=arm, category=r["type"], ok=ok,
                            hyp_len=sum(len(a) for a in r["answers"].values()),
                            in_tok=r["ans_in"] + r["judge_in"], out_tok=r["ans_out"] + r["judge_out"],
                            error=r["error"], **extra)
        return r

    recs = []
    for it, r in zip(items, pmap(_one, items)):
        if isinstance(r, Exception):
            r = {"uid": it["uid"], "type": it.get("type", "?"), "pass": None, "error": f"{type(r).__name__}: {r}",
                 "error_stage": "runner"}
        r.setdefault("type", r.get("category", it.get("type", "?")))
        r.setdefault("uid", it["uid"])
        recs.append(r)
    return recs


def _pct(c, t):
    return round(100.0 * c / t, 1) if t else None


def compute_metrics(recs: list) -> dict:
    """Official metrics over the scored (error-free) scenarios: accuracy per (type, dim) = the six
    settings; Overall = mean of the six (Table 2); plus the repo's per-type pooled overall, per-probe
    pooled across types, and the all-queries pooled accuracy. Percent, one decimal, like the paper."""
    scored = [r for r in recs if not r.get("error") and isinstance(r.get("pass"), dict)]
    cell = {(t, d): [0, 0] for t in _TYPES for d in _DIMS}
    for r in scored:
        t = r.get("type")
        for d in _DIMS:
            c = cell.setdefault((t, d), [0, 0])
            c[1] += 1
            c[0] += bool(r["pass"][d])
    settings = {f"{t}/{_PROBE[d]}": {"correct": c, "total": n, "accuracy": _pct(c, n)}
                for (t, d), (c, n) in sorted(cell.items())}
    present = [v["accuracy"] for v in settings.values() if v["total"]]
    by_type = {}
    for t in sorted({t for t, _ in cell}):
        c = sum(cell[(t, d)][0] for d in _DIMS)
        n = sum(cell[(t, d)][1] for d in _DIMS)
        by_type[t] = {"correct": c, "total": n, "accuracy": _pct(c, n)}
    by_probe = {}
    for d in _DIMS:
        c = sum(v[0] for (t, dd), v in cell.items() if dd == d)
        n = sum(v[1] for (t, dd), v in cell.items() if dd == d)
        by_probe[_PROBE[d]] = {"correct": c, "total": n, "accuracy": _pct(c, n)}
    tc = sum(v[0] for v in cell.values())
    tn = sum(v[1] for v in cell.values())
    stages: dict = {}
    for r in recs:
        if r.get("error"):
            stages[r.get("error_stage") or "?"] = stages.get(r.get("error_stage") or "?", 0) + 1
    return {
        "overall_mean_of_six": round(sum(present) / len(present), 1) if present else None,
        "overall_settings_present": len(present),
        "pooled_accuracy": _pct(tc, tn),
        "settings": settings, "by_type": by_type, "by_probe": by_probe,
        "n_scenarios": len(recs), "n_scored": len(scored), "n_queries_scored": tn,
        "errors": sum(stages.values()), "errors_by_stage": stages,
        "judge_parse_fallbacks": sum(1 for r in scored if r.get("judge_parse_fallback")),
    }


def _comparability(reader_model, judge_model):
    rk, jk = _price_key(reader_model), _price_key(judge_model)
    r_ok, j_ok = rk == _OFFICIAL_READER, jk in _OFFICIAL_JUDGES
    return {"reader": reader_model, "reader_official": r_ok, "judge": judge_model, "judge_official": j_ok,
            "comparable_to_published_memory_rows": r_ok and j_ok,
            "official": {"reader": "gpt-4o-mini (paper §4.1)",
                         "judge": "gemini-3.1-flash-lite-preview (repo default; paper: Gemini-3.1-flash-lite)",
                         "overall": "mean of the six (type x probe) accuracies (Table 2 caption)"}}


# ── run ─────────────────────────────────────────────────────────────────────────────────────────

def run(n=10, k=10, model="qwen3.5:9b-mlx-bf16", seed=20261024, agent_provider="ollama",
        judge_model=None, judge_provider=None, data_path=None, items_log=None, context_cache=None,
        qids_file=None, shard=None, ctx_order="rank", via_api=None, api_key_file=None,
        api_ctx="mcp_text", api_project_prefix="", api_explain=False, api_trace=None,
        retrieval_only=False, blind=False, _api_client=None) -> dict:
    _BLIND["on"] = bool(blind)
    judge_model = judge_model or model
    judge_provider = judge_provider or agent_provider
    for prov, mdl in ((agent_provider, model), (judge_provider, judge_model)):
        if prov == "ollama" and not _ollama_available(mdl):
            return {"status": "skipped", "reason": f"Ollama model {mdl!r} not available on localhost:11434"}
        envk = {"openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY"}.get(prov)
        if envk and not os.environ.get(envk):
            return {"status": "skipped", "reason": f"{envk} not set (needed for provider {prov!r})"}
    # The Kijito arm runs through a Kijito service's MCP endpoint (kijito_api_contexts).
    if not (via_api or _api_client is not None):
        return {"status": "skipped", "reason": "the Kijito arm runs through the API: pass --via-api BASE_URL "
                                               "and --api-key-file"}
    if api_ctx not in ("mcp_text", "contents"):
        return {"status": "skipped", "reason": f"--api-ctx must be mcp_text or contents (got {api_ctx!r})"}
    if ctx_order != "rank" and api_ctx == "mcp_text":
        return {"status": "skipped", "reason": "--ctx-order chrono reorders memory bodies; it needs "
                                               "--api-ctx contents (the MCP reply text is shown "
                                               "verbatim, in the server's order)"}
    from .client import MemoryCapReached, api_arm_info, blind_status, connect
    api, why = connect(via_api, api_key_file, _api_client)
    if api is None:
        return {"status": "skipped", "reason": why}
    api_info = api_arm_info(api, via_api, ctx_render=api_ctx, k=k, project_prefix=api_project_prefix,
                            explain=api_explain, blind=_blind_on(), project_per="scenario")
    lim = api_info.get("recall_limit_max")
    if lim is not None and k > lim:
        print(f"[stale] WARNING: k={k} exceeds kijito_recall's declared limit maximum {lim}; the service "
              f"may return fewer than k memories.", file=sys.stderr)
    # api_ingest_tag names WHERE the haystack was written (server + build + namespace); api_tag adds
    # the recall reply FORM and keys the reply cache + the resume sentinel.
    api_ingest_tag = (f"{api_info['base_host']}|commit={api_info['server_commit']}"
                      f"|prefix={api_project_prefix}")
    api_tag = f"{api_ingest_tag}|recall_explain={'on' if api_explain else 'off'}"
    if _blind_on():
        bs = blind_status(api)
        api_info["blind"]["status"] = bs
        if not (bs.get("active") and bs.get("index_exists")):
            return {"status": "refused",
                    "reason": "REFUSED: --blind needs the key's account enrolled in the keyword index over "
                              "encrypted content (turn on Keyword search at https://kijito.ai/account; a scoped API key "
                              "cannot enrol). "
                              f"GET /api/fts-blind/status says: {bs}"}
    try:
        data, path, ds_sha = load_dataset(data_path)
    except Exception as e:
        return {"status": "skipped", "reason": f"could not load the STALE dataset: {type(e).__name__}: {e}"}

    qids_sha = None
    if qids_file:
        raw = open(qids_file, "rb").read()
        qids_sha = hashlib.sha256(raw).hexdigest()[:12]
        wanted = [x.strip() for x in raw.decode().splitlines() if x.strip() and not x.startswith("#")]
        by_id = {r["uid"]: r for r in data}
        missing = [x for x in wanted if x not in by_id]
        if missing:
            return {"status": "skipped", "reason": f"--qids-file lists {len(missing)} uids absent from the "
                                                   f"dataset (first: {missing[0]!r})"}
        items = [by_id[x] for x in wanted]
    else:
        items = stratified_sample(data, n, seed)
    shard_idx = shard_n = None
    if shard:
        try:
            shard_idx, shard_n = (int(x) for x in str(shard).split("/", 1))
            if not (0 <= shard_idx < shard_n):
                raise ValueError
        except ValueError:
            return {"status": "skipped", "reason": f"--shard must be I/N with 0 <= I < N (got {shard!r})"}
        items = items[shard_idx::shard_n]

    manifest = run_manifest(seed=seed, dataset_paths=[path])
    manifest.update({
        "benchmark": "STALE (arXiv 2605.06527)", "upstream_code": _UPSTREAM_CODE,
        "dataset": {"repo": _HF_REPO, "file": _HF_FILE, "revision": _HF_REVISION, "path": path,
                    "sha256": ds_sha, "matches_published": ds_sha == _PUBLISHED_SHA256,
                    "license": "CC BY 4.0 (distractors from LongMemEval, MIT)", "split": _SPLIT,
                    "n_total": len(data)},
        "protocol": {"reader_prompt": "official run_target_model.build_prompts, verbatim; history = top-k "
                                      "retrieved memories", "ctx_order": ctx_order,
                     "judge_prompt_sha256": hashlib.sha256(JUDGE_SYSTEM.encode()).hexdigest(),
                     "judge_json_mode": True, "reader_temperature": 0, "judge_temperature": 0,
                     "judge_error": "counted as error (status invalid), not scored as fail",
                     "ingest": _INGEST_VERSION},
        "comparability": _comparability(model, judge_model),
        "blind_index": _blind_on(), "tokenizer": tokenizer_name(),
        # Only meaningful for an ollama reader: None = the model's default thinking (the judge always
        # runs JSON mode, which turns thinking off on ollama — see llm._ollama_call).
        "ollama_think": ollama_think(),
    })
    if qids_file:
        manifest["qids_file"] = {"path": qids_file, "sha256_12": qids_sha, "n": len(items)}
    if shard_n:
        manifest.update({"shard": f"{shard_idx}/{shard_n}", "shard_scenarios": len(items)})
    manifest["kijito_arm"] = api_info
    digest = "api"
    cache = ContextCache(context_cache) if context_cache else None

    # Resume sentinel: everything that changes a logged outcome. Validated before any ingest.
    parts = [agent_provider, model, judge_provider, judge_model, k, seed, _SPLIT, ds_sha,
             ctx_order, _INGEST_VERSION,
             f"blind={'on' if _blind_on() else 'off'}",
             f"shard={shard_idx}/{shard_n}" if shard_n else "shard=off",
             f"qids={qids_sha}" if qids_sha else "qids=off", f"ollama_think={ollama_think()}"]
    # Answers depend on WHICH server retrieved (host + commit + namespace + reply form) and on how the
    # reply is rendered for the reader.
    parts += [f"api={api_tag}", f"api_ctx={api_ctx}"]
    resume_cfg = hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:16]
    resume = ItemLog.load(items_log) if items_log else {}
    sentinel = resume.pop(("__config__", "__config__"), None)
    if resume and sentinel is None:
        return {"status": "skipped", "reason": f"items log {items_log!r} has no config sentinel; use a fresh one"}
    if sentinel is not None and sentinel.get("category") != resume_cfg:
        return {"status": "skipped",
                "reason": f"items log {items_log!r} was written under a DIFFERENT run config (reader/judge/k/"
                          f"seed/dataset/ctx-order/blind/shard/qids/ollama-think"
                          f"/api(host,server-commit,prefix,recall-explain,ctx-render)); use a fresh one"}
    log = ItemLog(items_log) if items_log else None
    if log is not None and sentinel is None:
        log.append(qid="__config__", arm="__config__", category=resume_cfg)

    # Scenarios already scored cleanly need no retrieval at all.
    todo = [it for it in items if not ((it["uid"], "kijito") in resume
                                       and not resume[(it["uid"], "kijito")].get("error"))]
    t0 = time.time()
    api_leaks, api_timings, api_ingest = {}, {}, {}
    from .client import Trace
    _API_CANARY_STATE["api"] = None
    try:
        ctxs, errs = kijito_api_contexts(todo, k, api, cache=cache, digest=digest,
                                         dataset_sha=ds_sha, api_tag=api_tag, ingest_tag=api_ingest_tag,
                                         project_prefix=api_project_prefix, explain=api_explain,
                                         trace=Trace(api_trace) if api_trace else None, leaks=api_leaks,
                                         timings=api_timings, ingest_stats=api_ingest)
    except (_APIRefused, MemoryCapReached) as e:
        print(f"\n[stale] RUN REFUSED — {e}\n", file=sys.stderr)
        return {"status": "refused", "reason": f"REFUSED: {e}", "manifest": manifest}
    api_info["scope_canary"] = (_API_CANARY_STATE["api"] or "not run (no live API retrieval: all "
                                "cached; cached leak verdicts re-asserted)")
    api_info["calls"] = {"total": api.stats.calls, "retries": api.stats.retries,
                         "by_tool": dict(api.stats.by_tool)}
    api_info["ingest"] = dict(api_ingest)
    render = lambda ctx: api_history(ctx, api_ctx, ctx_order)   # noqa: E731
    t_retr = time.time() - t0
    if retrieval_only:
        # --retrieval-only: fill the context cache (and the API trace) and stop before any reader or
        # judge call; a later run with the same cache scores those exact retrievals.
        n_err = len({u for u, _ in errs})
        out = {"status": "retrieval-only", "n": len(items), "retrieved_probes": len(ctxs),
               "retrieval_errors": dict((f"{u}:{d}", e) for (u, d), e in errs.items()),
               "retrieval_wall_s": round(t_retr, 1), "manifest": manifest,
               "scenarios_with_errors": n_err}
        if cache is not None:
            manifest["context_cache"] = {"path": context_cache, **cache.stats()}
        out["api_leaks"] = api_leaks
        out["retrieval_timing"] = api_timings
        return out
    t1 = time.time()
    recs = score_items(items, ctxs, errs, (agent_provider, model), (judge_provider, judge_model),
                       ctx_order=ctx_order, item_log=log, resume=resume, render=render)
    t_score = time.time() - t1
    metrics = compute_metrics(recs)

    scored = [r for r in recs if not r.get("error")]
    ctx_by_dim = {_PROBE[d]: [r["ctx_tokens"][d] for r in scored if d in (r.get("ctx_tokens") or {})]
                  for d in _DIMS}
    all_ctx = [x for v in ctx_by_dim.values() for x in v]
    ai = sum(r.get("ans_in", 0) for r in recs); ao = sum(r.get("ans_out", 0) for r in recs)
    ji = sum(r.get("judge_in", 0) for r in recs); jo = sum(r.get("judge_out", 0) for r in recs)
    cost = {
        "answer": {"provider": agent_provider, "model": model, "calls": sum(r.get("ans_calls", 0) for r in recs),
                   "in_tokens": ai, "out_tokens": ao, "usd": round(_cost(model, ai, ao), 4)},
        "judge": {"provider": judge_provider, "model": judge_model,
                  "calls": sum(r.get("judge_calls", 0) for r in recs),
                  "in_tokens": ji, "out_tokens": jo, "usd": round(_cost(judge_model, ji, jo), 4)},
        "resumed_items": sum(1 for r in recs if r.get("_resumed")),
        "retrieval_wall_s": round(t_retr, 1), "scoring_wall_s": round(t_score, 1),
    }
    cost["total_usd"] = round(cost["answer"]["usd"] + cost["judge"]["usd"], 4)
    efficiency = {"tokenizer": tokenizer_name(),
                  "context_tokens_per_query_mean": round(sum(all_ctx) / len(all_ctx), 1) if all_ctx else None,
                  "context_tokens_per_query_by_probe": {p: (round(sum(v) / len(v), 1) if v else None)
                                                        for p, v in ctx_by_dim.items()},
                  "context_tokens_total": sum(all_ctx),
                  "published_full_context_tokens_mean": 151784}
    if cache is not None:
        manifest["context_cache"] = {"path": context_cache, **cache.stats()}
    items_out = [{k2: v for k2, v in r.items() if k2 in ("uid", "type", "pass", "error", "error_stage")}
                 for r in recs]
    if api_leaks:
        print(f"\n[stale] !!! RUN INVALID: {len(api_leaks)} probe(s) recalled memories from OUTSIDE their "
              f"project (cross-scenario leak): {sorted(api_leaks)}\n", file=sys.stderr)
    sc = {
        "status": "ran" if (metrics["errors"] == 0 and not api_leaks) else "invalid",
        "config": {"n": len(items), "top_k": k, "seed": seed, "arm": "kijito",
                   "reader": f"{agent_provider}:{model}", "judge": f"{judge_provider}:{judge_model}",
                   "ctx_order": ctx_order, "via_api": bool(api), "api_ctx": api_ctx if api else None},
        "manifest": manifest, "metrics": metrics, "cost": cost, "efficiency": efficiency,
        "published_reference": PUBLISHED, "items": items_out,
    }
    sc["api_leaks"] = api_leaks
    sc["retrieval_timing"] = api_timings
    return sc


def _print(sc: dict) -> None:
    if sc.get("status") == "retrieval-only":
        print(f"\nSTALE retrieval-only: {sc['retrieved_probes']} probes retrieved for {sc['n']} scenarios, "
              f"{len(sc['retrieval_errors'])} retrieval errors, leaks {len(sc.get('api_leaks') or {})}, "
              f"{sc['retrieval_wall_s']}s\n")
        return
    if sc.get("status") not in ("ran", "invalid"):
        print(f"\nSTALE: {sc.get('reason')}\n")
        return
    c, m = sc["config"], sc["metrics"]
    comp = sc["manifest"]["comparability"]
    print(f"\nSTALE — kijito arm, k={c['top_k']}, n={c['n']} scenarios | reader={c['reader']} judge={c['judge']}")
    ka = sc["manifest"].get("kijito_arm") or {}
    if ka.get("mode") == "api":
        print(f"kijito arm: VIA API {ka.get('base_url')} (server commit {ka.get('server_commit')}, "
              f"context={ka.get('ctx_render')}, blind={'on' if ka.get('blind', {}).get('on') else 'off'}, "
              f"canary: {ka.get('scope_canary')}, leaks: {len(sc.get('api_leaks') or {})}, "
              f"calls: {(ka.get('calls') or {}).get('total')})")
    print("comparable to the published memory-system rows: "
          + ("YES" if comp["comparable_to_published_memory_rows"] else "NO (reader/judge differ from gpt-4o-mini / gemini-3.1-flash-lite)"))
    print(f"status={sc['status']}  errors={m['errors']} {m['errors_by_stage']}  parse_fallbacks={m['judge_parse_fallbacks']}")
    print("setting".ljust(10) + "acc%".rjust(8) + "   (correct/total)")
    for name, v in m["settings"].items():
        acc = "-" if v["accuracy"] is None else f"{v['accuracy']:.1f}"
        print(name.ljust(10) + acc.rjust(8) + f"   ({v['correct']}/{v['total']})")
    print(f"Overall (mean of {m['overall_settings_present']} settings) = {m['overall_mean_of_six']}   "
          f"pooled = {m['pooled_accuracy']}")
    co, ef = sc["cost"], sc["efficiency"]
    print(f"answer: {co['answer']['calls']} calls in={co['answer']['in_tokens']} out={co['answer']['out_tokens']} "
          f"${co['answer']['usd']} | judge: {co['judge']['calls']} calls in={co['judge']['in_tokens']} "
          f"out={co['judge']['out_tokens']} ${co['judge']['usd']} | total ${co['total_usd']}")
    print(f"context tokens/query ({ef['tokenizer']}): {ef['context_tokens_per_query_mean']} "
          f"(full-context protocol: ~{ef['published_full_context_tokens_mean']})")
    print("published (GPT-4o-mini reader): LightMem 17.8 | mem-0 8.3 | Zep 6.0 | CUPMem 68.0 ; "
          "best full-context LLM Gemini-3.1-pro 55.2\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="STALE (implicit memory invalidation) through Kijito's public API")
    ap.add_argument("--n", type=int, default=10, help="stratified (T1/T2) subset size; 0 = all 400")
    ap.add_argument("--k", type=int, default=10, help="kijito_recall limit per probe (protocol: 10)")
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--agent-provider", default="ollama", choices=["ollama", "openai", "openrouter", "anthropic"],
                    help="reader provider (official reader: openrouter openai/gpt-4o-mini)")
    ap.add_argument("--model", default="qwen3.5:9b-mlx-bf16", help="reader model")
    ap.add_argument("--judge-provider", default=None, choices=["ollama", "openai", "openrouter", "anthropic"],
                    help="judge provider (default: the reader's)")
    ap.add_argument("--judge-model", default=None,
                    help="judge model (default: the reader's; official: google/gemini-3.1-flash-lite-preview)")
    ap.add_argument("--data-path", default=None, help="local T1_T2_400_FULL.json (default: pinned HF revision)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--items-log", default=None, help="per-scenario JSONL checkpoint (default <out>.items.jsonl)")
    ap.add_argument("--context-cache", default=None, help="JSONL cache of retrieved contexts")
    ap.add_argument("--blind", action="store_true",
                    help="the key's account is enrolled in the keyword index over encrypted content (an account "
                         "setting); checked with GET /api/fts-blind/status and refused if it is not")
    ap.add_argument("--qids-file", default=None, help="explicit scenario uid list (replaces --n)")
    ap.add_argument("--shard", default=None, help="I/N: score only items[I::N]; give each shard its own logs")
    ap.add_argument("--ctx-order", choices=["rank", "chrono"], default="rank",
                    help="order of retrieved memories in the reader prompt (default: Kijito's ranking)")
    ap.add_argument("--via-api", default=DEFAULT_API, dest="via_api", metavar="BASE_URL",
                    help=f"the Kijito service (default {DEFAULT_API}): kijito_remember_batch ingest into one "
                         "project per scenario, kijito_recall (scope=project, limit=k) per probe. Needs "
                         "--api-key-file (a key that can read and write memories).")
    ap.add_argument("--api-key-file", default=None, dest="api_key_file",
                    help="file holding the Kijito API key (must be mode 0600)")
    ap.add_argument("--api-ctx", default="mcp_text", choices=["mcp_text", "contents"], dest="api_ctx",
                    help="the reader's history — mcp_text (default) = the kijito_recall reply verbatim; "
                         "contents = the parsed memory bodies as '- <note>' lines")
    ap.add_argument("--api-project-prefix", default="", dest="api_project_prefix",
                    help="project = <prefix><uid>. A project that already holds memories is refused (use a "
                         "fresh prefix per run).")
    ap.add_argument("--api-explain", action="store_true", dest="api_explain",
                    help="recall with explain=true (the DETAILED reply form)")
    ap.add_argument("--retrieval-only", action="store_true", dest="retrieval_only",
                    help="retrieve (fill --context-cache / the API trace) and stop before any reader or judge "
                         "call; a later run with the same cache scores those exact retrievals")
    ap.add_argument("--api-trace", default=None, dest="api_trace",
                    help="per-scenario JSONL trace (timings, recalled ids, leaks); default "
                         "<out>.api-trace.jsonl when --out is given")
    args = ap.parse_args(argv)
    if not args.api_key_file:
        ap.error("--api-key-file is required (the Kijito API key; see README)")
    if args.n < 0:
        ap.error("--n must be >= 0")
    items_log = args.items_log or (
        f"{args.out}{'.shard' + args.shard.replace('/', '-') if args.shard else ''}.items.jsonl" if args.out else None)
    sc = run(args.n, args.k, args.model, args.seed, agent_provider=args.agent_provider,
             judge_model=args.judge_model, judge_provider=args.judge_provider, data_path=args.data_path,
             items_log=items_log, context_cache=args.context_cache, qids_file=args.qids_file,
             shard=args.shard, ctx_order=args.ctx_order, via_api=args.via_api,
             api_key_file=args.api_key_file, api_ctx=args.api_ctx, api_project_prefix=args.api_project_prefix,
             api_explain=args.api_explain, retrieval_only=args.retrieval_only, blind=args.blind,
             api_trace=args.api_trace or (f"{args.out}{'.shard' + args.shard.replace('/', '-') if args.shard else ''}"
                                          f".api-trace.jsonl" if (args.out and args.via_api) else None))
    _print(sc)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(json.dumps(sc, indent=2))
        print(f"[scorecard written to {args.out}]")
    if sc.get("status") == "retrieval-only":
        return 0 if not (sc["retrieval_errors"] or sc.get("api_leaks")) else 2
    return 0 if sc.get("status") == "ran" else (2 if sc.get("status") == "invalid" else 1)


if __name__ == "__main__":
    sys.exit(main())
