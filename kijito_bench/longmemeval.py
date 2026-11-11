"""LongMemEval (ICLR 2025, github.com/xiaowu0162/LongMemEval) through Kijito's public API.

LongMemEval scores long-term memory over multi-session chat: single-session facts, preferences,
knowledge updates, temporal reasoning, multi-session synthesis and abstention. Each question comes with
its own chat history ("haystack"); the `s` split has ~50 sessions (~490 turns) per question.

The Kijito arm (the only arm that talks to Kijito; it needs --via-api, default https://api.kijito.ai, and --api-key-file).
Per question, in ONE project of the key's account (<prefix><question_id>):
- ingest: every non-empty chat turn becomes one memory, "[<session date>] <role>: <content>", written in
  haystack order with kijito_remember_batch (<= 60 items and < 1 MiB per call; a turn over the 64 KiB
  per-memory limit is split into "(part i/n)" pieces);
- recall: ONE kijito_recall(query=<question>, limit=k, scope="project", project=<that project>);
- the reader's NOTES block is the kijito_recall reply text exactly as an agent reads it (--api-ctx
  mcp_text, the default) or the memory bodies parsed out of it (--api-ctx contents).
A scope canary runs before the first live question and every recall is leak-checked: a memory from
another project marks the run invalid.

CONTROL ARM append_md (on unless --no-append-md): the same turn notes, ranked by BM25, top-k.

OFFICIAL PROTOCOL (--official-eval): the official per-type yes/no judge prompts from LongMemEval
src/evaluation/evaluate_qa.py, verbatim, parsed the official way ('yes' in the lowercased reply), with
'_abs' question ids routed to the abstention template; judge gpt-4o-2024-08-06 (OpenRouter id
openai/gpt-4o-2024-08-06). The reader prompt is a generic non-CoT answer prompt (below). Without
--official-eval a simpler harness judge is used (directional only).

Run: python -m kijito_bench.longmemeval --split s --k 50 --official-eval --agent-provider openrouter \
       --model openai/gpt-4o-2024-08-06 --qids-file splits/lme-s-holdout.qids --no-append-md \
       --api-key-file ~/.kijito/key --api-project-prefix lme-run1- --out out/lme.json
"""
import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time

# provider dispatch (ollama/openai/openrouter/anthropic) + pricing + the ONE harness tokenizer
from .llm import _llm, _cost, _ollama_available, context_tokens, tokenizer_name
from .runner import ContextCache, ItemLog, _git, _repo_root, _sha256_file, pmap, paired_stats, run_manifest
from .baselines import bm25_rank, _Fact

DEFAULT_API = "https://api.kijito.ai"

_REPO = "xiaowu0162/longmemeval-cleaned"
# The dataset revision the published runs used (the HF snapshot whose longmemeval_s_cleaned.json has
# sha256 d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442). Pinned so a rerun reads
# the same file even if the dataset repo moves on.
_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
_ORACLE = "longmemeval_oracle.json"
_SPLITS = {"oracle": "longmemeval_oracle.json",   # evidence sessions only (15MB)
           "s": "longmemeval_s_cleaned.json",     # full distractor haystack (277MB)
           "m": "longmemeval_m_cleaned.json"}      # larger haystack (2.7GB)

# The judge the official LongMemEval evaluation (src/evaluation/evaluate_qa.py) pins. A scorecard
# judged by anything else is directional, and its manifest says so (judge_matches_official).
_OFFICIAL_JUDGE = "gpt-4o-2024-08-06"


def _stratified_sample(data: list, sample_n: int, seed: int) -> list:
    """Deterministically sample `sample_n` items stratified by question_type via round-robin across the
    types — so a small sample still covers knowledge-update/temporal/multi-session, not just the first
    bucket. Same seed -> identical sample. sample_n<=0 or >=len returns the whole set."""
    if sample_n <= 0 or sample_n >= len(data):
        return list(data)
    by_type: dict[str, list] = {}
    for q in data:
        by_type.setdefault(q.get("question_type", "?"), []).append(q)
    rng = random.Random(seed)
    for v in by_type.values():
        rng.shuffle(v)
    picked, types = [], sorted(by_type)
    i = 0
    while len(picked) < sample_n and any(by_type.values()):
        t = types[i % len(types)]
        if by_type[t]:
            picked.append(by_type[t].pop())
        i += 1
    return picked


def _load_split(split: str, sample_n: int, seed: int):
    """Download (cached) the given split at the pinned revision + return (deterministic stratified
    sample, resolved dataset path). oracle = evidence sessions only; s/m = the full distractor haystack.
    The path is threaded to run_manifest(dataset_paths=...) so the manifest and the resume sentinel carry
    the file's real digest."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(_REPO, _SPLITS[split], repo_type="dataset", revision=_REVISION)
    data = json.load(open(path))
    return _stratified_sample(data, sample_n, seed), path


def _turn_notes(q) -> list[str]:
    """Flatten a question's haystack into one note per chat turn, stamped with the session date (temporal
    questions need the date signal) and the speaker. The system must find the relevant turns itself."""
    notes = []
    sessions = q.get("haystack_sessions") or []
    dates = q.get("haystack_dates") or []
    for si, sess in enumerate(sessions):
        date = dates[si] if si < len(dates) else ""
        for turn in (sess or []):
            role = turn.get("role", "?")
            content = (turn.get("content") or "").strip()
            if content:
                notes.append(f"[{date}] {role}: {content}")
    return notes


_ANSWER_PROMPT = (
    "You are answering a question about a user from their past chat history. Use ONLY the notes below; "
    "if several notes conflict, prefer the most RECENT (by date). Answer in a few words, no explanation. "
    "If the notes don't contain the answer, reply UNKNOWN.\n\n"
    "Today's date: {today}\nNOTES:\n{ctx}\n\nQUESTION: {q}\nANSWER:"
)

_JUDGE_PROMPT = (
    "You are grading whether a model's answer matches the gold answer to a question. Be strict but accept "
    "paraphrases / equivalent values. Reply with ONLY one word: CORRECT or WRONG.\n\n"
    "QUESTION: {q}\nGOLD ANSWER: {gold}\nMODEL ANSWER: {hyp}\nVERDICT:"
)

# Official-protocol (--official-eval) answering: a faithful generic equivalent of LongMemEval's non-CoT
# answer instruction. Deliberately NON-SUPPRESSIVE — it drops the harness prompt's "answer in a few
# words"/"reply UNKNOWN" steering, because the official eval judges preference RUBRICS + off-by-one
# temporal tolerance, and a terse "UNKNOWN"-primed answer floors those personalized/reasoning types.
# The official generation is generic (no per-type instruction, no length cap, no terse steering); the
# retrieved notes block is passed verbatim (dates matter for temporal questions) and the SAME prompt is
# used for every type (the official eval does not differentiate answering by type).
_OFFICIAL_ANSWER_PROMPT = (
    "You are a personal assistant answering a question about a user. Below are user facts extracted from "
    "the chat history, each stamped with its date. Answer the question using these facts; when facts "
    "conflict, prefer the most RECENT (by date).\n\n"
    "Today's date: {today}\nNOTES:\n{ctx}\n\nQUESTION: {q}\nANSWER:"
)


# ~6% of the oracle split are ABSTENTION questions whose gold answer is "the information provided is not
# enough…": the official metric credits a model that correctly ABSTAINS (says it doesn't know). We must
# mirror that — scoring a correct abstention as wrong would both understate accuracy AND penalize honesty.
_ABSTAIN_MARKERS = ("not enough", "cannot be determined", "insufficient", "no information",
                    "don't know", "do not know", "unknown", "unable to")


def _is_abstention_gold(gold: str) -> bool:
    g = (gold or "").lower()
    return ("not enough" in g) or ("cannot be determined" in g) or ("insufficient" in g)


def _looks_abstained(hyp: str) -> bool:
    h = (hyp or "").strip().lower()
    return (not h) or any(m in h for m in _ABSTAIN_MARKERS)


def _judge(provider: str, model: str, q: str, gold: str, hyp: str,
           official: bool = False, qtype: str = None, qid: str = None) -> bool:
    """Grade hyp vs gold. Abstention questions (gold='not enough info') are correct IFF the model
    abstained. Otherwise an abstained/empty hyp is wrong, and a real answer is graded by the LLM judge —
    checking WRONG/INCORRECT FIRST because 'INCORRECT' contains the substring 'CORRECT'.

    OFFICIAL MODE (--official-eval): delegate to the LongMemEval per-type judge, which REPLACES the whole
    marker path above — including the `_is_abstention_gold`/`_looks_abstained` short-circuits (official
    abstention grading is an LLM call keyed on '_abs' in the question_id, not marker-matching). Flag-off
    (the default) is byte-for-byte the pre-existing behavior — the new kwargs are inert unless official."""
    if official:
        return _official_judge(provider, model, qtype, qid, q, gold, hyp)
    if _is_abstention_gold(gold):
        return _looks_abstained(hyp)            # correct iff the model declined to answer
    if _looks_abstained(hyp):
        return False                            # a real question the model failed to answer
    verdict = (_llm(provider, model, _JUDGE_PROMPT.format(q=q, gold=gold, hyp=hyp))[0] or "").strip().upper()
    if "INCORRECT" in verdict or "WRONG" in verdict:
        return False
    return "CORRECT" in verdict


def _judge_counted(provider, model, q, gold, hyp, official=False, qtype=None, qid=None):
    """Mirror `_judge` EXACTLY (same abstention short-circuits, same WRONG-before-CORRECT parse) but also
    return the judge's token cost: (ok, in_tok, out_tok, calls). `_judge` stays the untouched shared
    apples-to-apples contract other modules import — this only adds the token passthrough for the cost
    scorecard block, so nothing about the grading changes. A test pins the verdict equivalence (see
    test_judge_counted_matches_judge) to catch any drift between the two.

    OFFICIAL MODE mirrors `_judge`'s official branch through `_official_judge_counted` — the SAME per-type
    prompt + 'yes'-in-lowercase parse — so the two stay verdict-equivalent under --official-eval too."""
    if official:
        return _official_judge_counted(provider, model, qtype, qid, q, gold, hyp)
    if _is_abstention_gold(gold):
        return _looks_abstained(hyp), 0, 0, 0        # correct iff the model declined; no LLM call
    if _looks_abstained(hyp):
        return False, 0, 0, 0                         # a real question the model failed to answer
    text, itok, otok = _llm(provider, model, _JUDGE_PROMPT.format(q=q, gold=gold, hyp=hyp))
    verdict = (text or "").strip().upper()
    ok = ("CORRECT" in verdict) and not ("INCORRECT" in verdict or "WRONG" in verdict)
    return ok, itok, otok, 1


# ── official LongMemEval per-type judge (opt-in, --official-eval) ─────────────────────────
# The five yes/no judge templates below are VERBATIM from xiaowu0162/LongMemEval
# src/evaluation/evaluate_qa.py `get_anscheck_prompt` (fetched 2026-07-16) — do NOT reword them; the whole
# point of --official-eval is that the numbers are directly comparable to other systems' published
# LongMemEval scores, which requires the EXACT official prompt + parse (temperature 0, label = 'yes' in response.lower()).
# `.format` fills three POSITIONAL slots in the official order: (question, correct_answer/rubric/
# explanation, model_response). The per-type differences that actually move scores: temporal-reasoning
# forgives off-by-one day counts; knowledge-update accepts an updated answer alongside stale info;
# single-session-preference grades against a RUBRIC (not exact-match); abstention grades "did the model
# correctly say it's unanswerable?". Any type not in the map falls back to the base contains-the-answer
# template (the official three-type default).

_OFFICIAL_JUDGE_DEFAULT = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

_OFFICIAL_JUDGE_TEMPORAL = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

_OFFICIAL_JUDGE_KNOWLEDGE_UPDATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

_OFFICIAL_JUDGE_PREFERENCE = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

_OFFICIAL_JUDGE_ABSTENTION = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."

_OFFICIAL_JUDGE_BY_TYPE = {
    "single-session-user": _OFFICIAL_JUDGE_DEFAULT,
    "single-session-assistant": _OFFICIAL_JUDGE_DEFAULT,
    "multi-session": _OFFICIAL_JUDGE_DEFAULT,
    "temporal-reasoning": _OFFICIAL_JUDGE_TEMPORAL,
    "knowledge-update": _OFFICIAL_JUDGE_KNOWLEDGE_UPDATE,
    "single-session-preference": _OFFICIAL_JUDGE_PREFERENCE,
}


def _official_judge_prompt(qtype, qid, q, gold, hyp) -> str:
    """Assemble the official per-type yes/no judge prompt EXACTLY as the LongMemEval reference does.
    Abstention routing keys on '_abs' in the question_id (mirrors the official `abstention` flag) and
    OVERRIDES the per-type template — an '_abs' question is graded 'did the model correctly identify this
    as unanswerable?' regardless of its nominal type. This REPLACES the harness's gold-marker abstention detection:
    the routing signal is the question_id, not the gold string. Unknown types fall back to the base
    contains-the-correct-answer template."""
    if qid is not None and "_abs" in str(qid):
        template = _OFFICIAL_JUDGE_ABSTENTION
    else:
        template = _OFFICIAL_JUDGE_BY_TYPE.get(qtype, _OFFICIAL_JUDGE_DEFAULT)
    return template.format(q, gold, hyp)


def _official_judge(provider, model, qtype, qid, q, gold, hyp) -> bool:
    """Official LongMemEval grading: ONE LLM call with the per-type yes/no prompt, label = 'yes' in
    response.lower() (the official parse). No marker-based abstention short-circuit — under the official
    protocol a correct abstention is itself an LLM judgement (the '_abs' template), not string-matching."""
    text = _llm(provider, model, _official_judge_prompt(qtype, qid, q, gold, hyp))[0]
    return "yes" in (text or "").lower()


def _official_judge_counted(provider, model, qtype, qid, q, gold, hyp):
    """Token-counting mirror of `_official_judge` — SAME prompt (shared builder), SAME 'yes'-in-lowercase
    parse — plus the (in_tok, out_tok, calls) passthrough for the cost block. Verdict-equivalent to
    `_official_judge` by construction, which keeps _judge/_judge_counted equivalent under official mode."""
    text, itok, otok = _llm(provider, model, _official_judge_prompt(qtype, qid, q, gold, hyp))
    return ("yes" in (text or "").lower()), itok, otok, 1



# ── retrieval plumbing ─────────────────────────────────────────────────────────────────────────

def _cache_key(cache, qid, arm, k, digest, api_tag=None, blind=False):
    """Cache key for a (question, arm) retrieval, or None when no cache is active. The SPLIT and the
    dataset file's content digest arrive inside `digest` (run() builds it for every arm), because oracle
    and s/m share question ids. For the Kijito arm, `api_tag` binds the server (host + /api/version
    commit + project namespace + reply form) that produced the retrieval."""
    if cache is None:
        return None
    extra = "blind=on" if blind else "blind=off"
    if api_tag:
        extra += f"|api={api_tag}"
    return ContextCache.make_key("longmemeval", qid, arm, k, digest, extra=extra)


class _CanaryFailed(RuntimeError):
    """The scope canary refused the run. run() turns it into a status='refused' scorecard."""


def _retrieval_failed(errors, arm, qid, e) -> None:
    """Record ONE retrieval failure: never cached (a cached [] would replay the failure as a real empty
    context forever), tallied into the scorecard's retrieval_errors, and the run is marked invalid."""
    msg = f"{type(e).__name__}: {e}"
    print(f"[lme] {arm} retrieval FAILED for {qid} (NOT cached; run will be marked INVALID): {msg}",
          file=sys.stderr)
    if errors is not None:
        errors[qid] = msg


def _md_contexts(questions, k, cache=None, digest=None, errors=None):
    """append-markdown: the same turns as notes; BM25 top-k per question. Cache-keyed distinctly from the
    kijito arm (arm='append_md'); a hit skips the BM25 retrieval for that question. A failure is
    absent from the result + recorded in `errors`, never cached (symmetric with the kijito arm)."""
    out = {}
    for q in questions:
        qid = q["question_id"]
        key = _cache_key(cache, qid, "append_md", k, digest)
        hit = cache.get(key) if cache is not None else None
        if hit is not None:
            out[qid] = hit
            continue
        try:
            notes = _turn_notes(q)
            facts = [_Fact(i, t) for i, t in enumerate(notes)]
            by_id = {f.id: f.text for f in facts}
            ranked, _ = bm25_rank(facts, q["question"], k) if facts else ([], None)
            contexts = [by_id[i] for i in ranked]
        except Exception as e:  # symmetric with the kijito arm — one bad question doesn't abort
            _retrieval_failed(errors, "append_md", qid, e)
            continue
        out[qid] = contexts
        if cache is not None:
            cache.put(key, contexts)
    return out


# ── the Kijito arm, through the public API ──────────────────────────────────────────────────────
# The Kijito arm ingests with kijito_remember_batch and recalls with kijito_recall over the MCP endpoint
# of a Kijito service, ONE account, ONE project per question (recall scope='project' isolates questions
# inside the account). Server-side behaviour (encryption, write-time work, caps, the formatted reply) is
# whatever the service does for every user; the harness changes nothing server-side.
#
# Cache value shape (API mode): [<the kijito_recall reply text>, <json meta>]. The reply text is the
# retrieval; how it is handed to the reader is decided at scoring time (--api-ctx), so one retrieval
# serves both renderings. meta carries the project, the ids this question wrote and the leak verdict,
# so a cache hit re-asserts the leak check instead of forgetting it.

_API_CANARY_STATE = {"api": None}


def _api_render(mode: str):
    """How an API retrieval reaches the reader.
    mcp_text : EXACTLY the kijito_recall reply an agent reads (lean by default; the detailed form under
               --api-explain) — the default, and what a published number means.
    contents : the memory bodies parsed out of that same reply, in reply order, rendered like the
               append_md arm (`- <note>` per line) — isolates retrieval from formatting. A LEAN
               reply's bodies are one-line normalised by the server; their ' ⏎ ' glyphs are shown as
               line breaks again here (client.lean_unbreak — reader-facing only, see its note).
               A detailed reply's bodies are exact."""
    from .client import reply_bodies
    if mode == "contents":
        return lambda ctx: _ctx_block(reply_bodies(ctx[0] if ctx else ""))
    return lambda ctx: (ctx[0] if ctx else "")


def _api_canary(api, prefix: str) -> dict:
    """The shared scope canary (client.scope_canary), raising _CanaryFailed so run() refuses."""
    from .client import scope_canary
    res = scope_canary(api, prefix, exc=_CanaryFailed, label="lme")
    _API_CANARY_STATE["api"] = "passed"
    return res


def _api_leak_check(api, items, own_ids, project) -> dict:
    """The shared per-recall leak check (client.leak_check)."""
    from .client import leak_check
    return leak_check(api, items, own_ids, project)


def _api_contexts(questions, k, api, cache=None, digest=None, errors=None, api_tag=None,
                  project_prefix="", trace=None, timings=None, leaks=None, explain=False,
                  ingest_tag=None, ingest_stats=None, blind=False):
    """{question_id: [reply_text, meta_json]} through Kijito's MCP endpoint. Per question: refuse a
    project that already holds memories (a prior run would double the haystack), write the turn notes
    in chronological order with kijito_remember_batch, recall the question with kijito_recall
    (limit=k, scope='project', project=<prefix><qid>, explain=<explain>), then leak-check the reply. A
    failed call (after the client's retries) is a retrieval error: absent from the result, never cached.

    ONE INGEST, SEVERAL READS. Each completed ingest is also cached as an INGEST RECORD
    (arm 'kijito_api_ingest', keyed by `ingest_tag` = host|commit|prefix — no recall knobs), holding the
    project and the ids written. A reply-cache miss whose ingest record is present skips the probe and
    the ingest and runs only the recall + leak check on that project — so the same ingest can be read
    again in another reply form (--api-explain) without writing a second haystack. The reply itself is
    keyed by `api_tag`, which carries the recall form, so two forms never share a reply entry."""
    from .client import parse_recall
    out = {}
    misses = []
    if ingest_stats is not None:
        ingest_stats.setdefault("fresh", 0)
        ingest_stats.setdefault("reused", 0)
    for q in questions:
        qid = q["question_id"]
        key = _cache_key(cache, qid, "kijito", k, digest, api_tag=api_tag, blind=blind)
        hit = cache.get(key) if cache is not None else None
        if hit is not None and len(hit) == 2:
            meta = json.loads(hit[1])
            if meta.get("leaks") and leaks is not None:
                leaks[qid] = meta["leaks"]
            out[qid] = hit
        else:
            misses.append((q, key))
    if not misses:
        return out
    _api_canary(api, project_prefix)
    from .client import MemoryCapReached
    for mi, (q, key) in enumerate(misses):
        qid = q["question_id"]
        project = f"{project_prefix}{qid}"
        calls0, retries0 = api.stats.calls, api.stats.retries
        ikey = (_cache_key(cache, qid, "kijito_api_ingest", 0, digest, api_tag=ingest_tag, blind=blind)
                if (cache is not None and ingest_tag) else None)
        irec = cache.peek(ikey) if ikey else None
        reused = False
        try:
            t0 = time.time()
            notes = _turn_notes(q)
            if irec:
                im = json.loads(irec[0])
                if im.get("project") != project or im.get("n_notes") != len(notes):
                    raise RuntimeError(f"ingest record for {qid} does not match this question "
                                       f"({im.get('project')!r}, {im.get('n_notes')} notes)")
                ids, reused = list(im["own_ids"]), True
                t1 = t2 = time.time()
            else:
                probe = parse_recall(api.recall("benchmark namespace probe", limit=1, scope="project",
                                                project=project))
                if probe:
                    raise RuntimeError(f"project {project!r} already holds memories (an earlier run?); "
                                       f"use a fresh --api-project-prefix")
                t1 = time.time()
                ids = api.remember_batch(notes, project) if notes else []
                t2 = time.time()
                if ikey:
                    cache.put(ikey, [json.dumps({"project": project, "own_ids": ids,
                                                 "n_notes": len(notes)})])
            text = api.recall(q["question"], limit=k, scope="project", project=project,
                              explain=explain)
            t3 = time.time()
            items = parse_recall(text)
            q_leaks = _api_leak_check(api, items, set(ids), project)
        except Exception as e:  # one failing question must not abort the run
            _retrieval_failed(errors, "kijito", qid, e)
            if isinstance(e, MemoryCapReached):
                # The account is full: every later write would be refused too. Stop writing and record
                # the remaining questions as retrieval errors (the run is invalid; see the README).
                for q2, _ in misses[mi + 1:]:
                    _retrieval_failed(errors, "kijito", q2["question_id"], e)
                print("[lme] STOPPED: the account's memory cap is reached (see the README: memory "
                      "footprint per arm).", file=sys.stderr)
                break
            continue
        t_rec = {"n_notes": len(notes), "probe_s": round(t1 - t0, 3), "ingest_s": round(t2 - t1, 3),
                 "recall_s": round(t3 - t2, 3), "total_s": round(time.time() - t0, 3),
                 "ingest_reused": reused}
        if ingest_stats is not None:
            ingest_stats["reused" if reused else "fresh"] += 1
        if timings is not None:
            timings[qid] = t_rec
        if q_leaks:
            print(f"[lme] LEAK: {qid} recall returned memories from outside project {project!r}: "
                  f"{q_leaks} (run will be marked INVALID)", file=sys.stderr)
            if leaks is not None:
                leaks[qid] = {str(i): v for i, v in q_leaks.items()}
        meta = {"project": project, "own_ids": ids, "recalled_ids": [it.id for it in items],
                "leaks": {str(i): v for i, v in q_leaks.items()}, "recall_explain": bool(explain),
                "reply_form": sorted({it.form for it in items}), "ingest_reused": reused}
        ctx = [text, json.dumps(meta)]
        out[qid] = ctx
        if trace is not None:
            trace.append({"qid": qid, "project": project, **t_rec, "n_ids": len(ids),
                          "recalled": [{"id": it.id, "score": it.score} for it in items],
                          "leaks": meta["leaks"], "calls": api.stats.calls - calls0,
                          "retries": api.stats.retries - retries0})
        if cache is not None:
            cache.put(key, ctx)
    return out


def _Trace(path):
    """Per-question API-arm trace (the shared client.Trace)."""
    from .client import Trace
    return Trace(path)


def _api_blind_status(api) -> dict:
    """GET /api/fts-blind/status (the shared client.blind_status)."""
    from .client import blind_status
    return blind_status(api)


def _ctx_block(notes) -> str:
    """The retrieved-notes block handed to the reader (the empty-placeholder is added at prompt time)."""
    return "\n".join(f"- {c}" for c in notes)


def _score_arm(provider, model, questions, ctxs, arm=None, item_log=None, resume=None,
               official=False, judge_provider=None, judge_model=None, retr_error_qids=None,
               render=None) -> dict:
    """Answer each question from its retrieved context, judge vs gold, aggregate overall + by type.
    A per-question failure is counted as a miss (not a crash) so one bad item can't abort the run.

    OFFICIAL MODE (--official-eval, `official=True`): the non-suppressive `_OFFICIAL_ANSWER_PROMPT`
    replaces `_ANSWER_PROMPT`, and each question's (question_type, question_id) is threaded to the judge so
    it can pick the official per-type template (and route '_abs' ids to the abstention template).

    JUDGE vs READER: the judge calls go to (judge_provider, judge_model), defaulting to the reader's
    (provider, model). Answer-call and judge-call tokens are tracked separately and priced at their OWN
    model's rate (usd).

    RETRIEVAL FAILURES: a qid in `retr_error_qids` (or absent from `ctxs`) had no retrieval this run; it
    is NOT answered (no spend on a context that does not exist) and is recorded with an error, so the
    errored-record resume rule below re-runs it once retrieval works.

    - RESUME: if `resume` (from ItemLog.load) holds a CLEAN record for this (qid, arm), reuse the outcome
      WITHOUT re-calling the model — the checkpoint that survives a killed run. A record carrying an
      `error` is NOT reused (a transport failure or retrieval failure must re-run, never be replayed as
      a scored miss forever).
    - CONCURRENCY: the per-question agent+judge calls run through `pmap` (bounded threads, order-preserving,
      per-item fault isolation). Only the LLM scoring parallelizes; the upstream retrieval stays serial.
    - TOKEN ACCOUNTING: every _llm return (agent + judge) is summed for the scorecard cost block via
      _judge_counted (which mirrors _judge's verdict exactly but also returns tokens)."""
    resume = resume or {}
    # RENDER: how an arm's retrieved context becomes the reader's NOTES block. Default = the `- <note>`
    # list (append_md); the Kijito arm passes _api_render(...) (the MCP reply text by default).
    render = render or _ctx_block
    judge_provider = judge_provider or provider
    judge_model = judge_model or model
    retr_error_qids = retr_error_qids or frozenset()

    def _score_one(q):
        qid = q["question_id"]
        qtype = q.get("question_type", "?")
        # RESUME: a previously-logged CLEAN (qid, arm) is reused verbatim — no model call, same outcome.
        r = resume.get((qid, arm)) if arm is not None else None
        if r is not None and not r.get("error"):
            it, ot = r.get("in_tok", 0), r.get("out_tok", 0)
            return {"qid": qid, "arm": arm, "category": r.get("category", qtype), "ok": bool(r.get("ok")),
                    "hyp_len": r.get("hyp_len", 0), "in_tok": it, "out_tok": ot,
                    "answer_in_tok": r.get("answer_in_tok", 0), "answer_out_tok": r.get("answer_out_tok", 0),
                    "judge_in_tok": r.get("judge_in_tok", 0), "judge_out_tok": r.get("judge_out_tok", 0),
                    "error": None, "_calls": 1 if (it or ot) else 0}
        ain = aout = jin = jout = calls = hyp_len = 0
        ok, err = False, None
        hyp = None
        t_item = time.time()
        if qid in retr_error_qids or qid not in ctxs:
            err = "retrieval_failed: no context retrieved this run (not answered; re-runs on resume)"
        else:
            try:
                ctx = render(ctxs[qid]) or "(no relevant notes found)"
                answer_prompt = _OFFICIAL_ANSWER_PROMPT if official else _ANSWER_PROMPT
                hyp, itok, otok = _llm(provider, model,
                                       answer_prompt.format(today=q.get("question_date", ""), ctx=ctx,
                                                            q=q["question"]))
                ain += itok; aout += otok; calls += 1
                hyp = hyp or ""
                hyp_len = len(hyp)
                ok, jit, jot, jcalls = _judge_counted(judge_provider, judge_model, q["question"],
                                                      str(q.get("answer", "")), hyp,
                                                      official=official, qtype=qtype, qid=qid)
                jin += jit; jout += jot; calls += jcalls
            except Exception as e:
                print(f"[lme] scoring failed for {qid}: {type(e).__name__}: {e}", file=sys.stderr)
                ok, err = False, f"{type(e).__name__}: {e}"
        rec = {"qid": qid, "arm": arm, "category": qtype, "ok": bool(ok), "hyp_len": hyp_len,
               "in_tok": ain + jin, "out_tok": aout + jout, "error": err,
               "answer_in_tok": ain, "answer_out_tok": aout, "judge_in_tok": jin, "judge_out_tok": jout}
        if item_log is not None:
            # hyp + elapsed_s are diagnostics (per-question gap tracing, reader+judge wall time);
            # resume reads neither.
            item_log.append(**{fld: rec[fld] for fld in ItemLog._FIELDS},
                            answer_in_tok=ain, answer_out_tok=aout, judge_in_tok=jin, judge_out_tok=jout,
                            hyp=hyp, elapsed_s=round(time.time() - t_item, 2))
        rec["_calls"] = calls
        return rec

    # pmap keeps input order + isolates per-item failures; _score_one already traps its own exceptions, so a
    # returned Exception here would be a runner-level surprise — record it as a miss rather than aborting.
    records = []
    for q, out in zip(questions, pmap(_score_one, questions)):
        if isinstance(out, Exception):
            out = {"qid": q["question_id"], "arm": arm, "category": q.get("question_type", "?"), "ok": False,
                   "hyp_len": 0, "in_tok": 0, "out_tok": 0, "answer_in_tok": 0, "answer_out_tok": 0,
                   "judge_in_tok": 0, "judge_out_tok": 0,
                   "error": f"{type(out).__name__}: {out}", "_calls": 0}
        records.append(out)

    by_type: dict[str, list] = {}
    correct = errors = in_tok = out_tok = llm_calls = 0
    ain = aout = jin = jout = 0
    for r in records:
        by_type.setdefault(r["category"], []).append(r["ok"])
        correct += r["ok"]
        if r["error"]:
            errors += 1
        in_tok += r["in_tok"]; out_tok += r["out_tok"]; llm_calls += r.get("_calls", 0)
        ain += r.get("answer_in_tok", 0); aout += r.get("answer_out_tok", 0)
        jin += r.get("judge_in_tok", 0); jout += r.get("judge_out_tok", 0)

    # The official print_qa_metrics.py views: task-averaged = the unweighted mean of the per-type
    # accuracies; abstention = accuracy over the '_abs' questions (which ALSO count in their type).
    type_accs = [sum(v) / len(v) for v in by_type.values()]
    abs_oks = [r["ok"] for r in records if str(r["qid"]).endswith("_abs")]
    # Efficiency: tokens of the retrieved context actually handed to the reader, measured on the ONE
    # harness tokenizer (recorded alongside). Failed retrievals have no context and are excluded.
    ctx_toks = [context_tokens(render(ctxs[q["question_id"]])) for q in questions
                if q["question_id"] in ctxs and q["question_id"] not in retr_error_qids]

    return {
        "accuracy": round(correct / len(questions), 4) if questions else 0.0,
        "task_averaged_accuracy": round(sum(type_accs) / len(type_accs), 4) if type_accs else 0.0,
        "abstention_accuracy": round(sum(abs_oks) / len(abs_oks), 4) if abs_oks else None,
        "abstention_n": len(abs_oks),
        "correct": correct, "n": len(questions), "errors": errors,
        "by_type": {t: round(sum(v) / len(v), 4) for t, v in sorted(by_type.items())},
        "in_tok": in_tok, "out_tok": out_tok, "llm_calls": llm_calls,
        "answer_in_tok": ain, "answer_out_tok": aout, "judge_in_tok": jin, "judge_out_tok": jout,
        "usd": round(_cost(model, ain, aout) + _cost(judge_model, jin, jout), 4),
        "ctx_tokens_mean": round(sum(ctx_toks) / len(ctx_toks), 1) if ctx_toks else 0.0,
        "ctx_tokens_p50": statistics.median(ctx_toks) if ctx_toks else 0,
        "ctx_tokenizer": tokenizer_name(),
        # per-item (qid, ok) kept for the paired stats block; stripped before the scorecard is written
        "_items": [{"qid": r["qid"], "ok": r["ok"]} for r in records],
    }


def _prompt_digest(official: bool) -> str:
    """Digest of the answer + judge prompt TEXTS this run actually uses (resume-sentinel provenance):
    a resume across a prompt edit would otherwise mix answers/verdicts from two prompts under one
    manifest. Official mode uses the official answer prompt + every per-type template (+ abstention)."""
    if official:
        texts = [_OFFICIAL_ANSWER_PROMPT, json.dumps(_OFFICIAL_JUDGE_BY_TYPE, sort_keys=True),
                 _OFFICIAL_JUDGE_DEFAULT, _OFFICIAL_JUDGE_ABSTENTION]
    else:
        texts = [_ANSWER_PROMPT, _JUDGE_PROMPT]
    return hashlib.sha256("\x1f".join(texts).encode()).hexdigest()[:12]




def _code_state() -> str:
    """'clean', or 'dirty:<digest>' over the TRACKED diff vs HEAD plus untracked *.py files (name +
    content) of this harness checkout — the code half of the resume-sentinel binding. Deliberately NOT
    the manifest's git_dirty bool, which also counts untracked non-code files: writing this run's own
    items log / scorecard inside the checkout would flip it between a run and its resume. A digest (not
    a bare flag) also refuses a resume across two DIFFERENT dirty states."""
    root = _repo_root()
    if not root:
        return "not-a-git-checkout"
    diff = _git(["diff", "HEAD"], cwd=root)
    untracked = sorted(x for x in _git(["ls-files", "--others", "--exclude-standard", "--", "*.py"],
                                       cwd=root).splitlines() if x.strip())
    if not diff and not untracked:
        return "clean"
    h = hashlib.sha256(diff.encode())
    for path in untracked:
        h.update(path.encode()); h.update(str(_sha256_file(os.path.join(root, path))).encode())
    return f"dirty:{h.hexdigest()[:12]}"


def run(n: int, k: int, model: str, seed: int, split: str = "oracle",
        agent_provider: str = "ollama", items_log: str = None, context_cache: str = None,
        official_eval: bool = False, shard: str = None, qids_file: str = None, judge_model: str = None,
        judge_provider: str = None, via_api: str = None, api_key_file: str = None,
        api_ctx: str = "mcp_text", api_project_prefix: str = "", api_trace: str = None,
        no_append_md: bool = False, api_explain: bool = False, blind: bool = False,
        _api_client=None) -> dict:
    judge_model = judge_model or model
    judge_provider = judge_provider or agent_provider
    # Both the reader AND the judge must be reachable (a judge that cannot run scores nothing).
    for role, prov, mdl in (("agent", agent_provider, model), ("judge", judge_provider, judge_model)):
        if prov == "ollama" and not _ollama_available(mdl):
            return {"status": "skipped",
                    "reason": f"Ollama model {mdl!r} not available on localhost:11434 (needed as the {role})"}
        envk = {"openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY"}.get(prov)
        if envk and not os.environ.get(envk):
            return {"status": "skipped", "reason": f"{envk} not set (needed for {role} provider {prov!r})"}
    # The Kijito arm runs through a Kijito service's MCP endpoint (see _api_contexts).
    if not (via_api or _api_client is not None):
        return {"status": "skipped", "reason": "the Kijito arm runs through the API: pass --via-api BASE_URL "
                                               "and --api-key-file"}
    if api_ctx not in ("mcp_text", "contents"):
        return {"status": "skipped", "reason": f"--api-ctx must be mcp_text or contents (got {api_ctx!r})"}
    from .client import KijitoMCPClient, MemoryCapReached, load_key, host_of, recall_limit_max
    api = None
    try:
        api = _api_client or KijitoMCPClient(via_api, load_key(api_key_file))
        ver = api.version()
        tools = api.tools_list()
        tool_names = {t.get("name") for t in tools}
    except Exception as e:
        return {"status": "skipped",
                "reason": f"--via-api {via_api or getattr(api, 'base', None)!r} unusable: "
                          f"{type(e).__name__}: {e}"}
    need = {"kijito_remember_batch", "kijito_recall", "kijito_get"} - tool_names
    if need:
        return {"status": "skipped", "reason": f"the service's key cannot call {sorted(need)}"}
    api_info = {"mode": "api", "base_host": host_of(api.base), "server_commit": ver.get("commit"),
                "server_version": ver.get("version"), "server_started_at": ver.get("started_at"),
                "ingest_tool": "kijito_remember_batch", "recall_tool": "kijito_recall",
                "recall_scope": "project", "project_prefix": api_project_prefix,
                "ctx_render": api_ctx, "recall_explain": bool(api_explain),
                "recall_limit_max": recall_limit_max(tools),
                "note": "the server's build is server_commit (from GET /api/version); this harness sets "
                        "nothing server-side."}
    lim = api_info["recall_limit_max"]
    if lim is not None and k > lim:
        print(f"[longmemeval] WARNING: k={k} exceeds kijito_recall's declared limit maximum {lim}; the "
              f"service may return fewer than k memories.", file=sys.stderr)
    # api_ingest_tag names WHERE the haystack was written (server + build + namespace); api_tag adds
    # the recall reply FORM, and keys the reply cache + the resume sentinel. So a lean and an
    # explain read of one ingest share the ingest record but never a reply or an items log.
    api_ingest_tag = (f"{api_info['base_host']}|commit={api_info['server_commit']}"
                      f"|prefix={api_project_prefix}")
    api_tag = f"{api_ingest_tag}|recall_explain={'on' if api_explain else 'off'}"
    if blind:
        bs = _api_blind_status(api)
        api_info["blind_status"] = bs
        if not (bs.get("active") and bs.get("index_exists")):
            return {"status": "refused",
                    "reason": "REFUSED: --blind needs the key's account enrolled in the keyword index over "
                              "encrypted content (turn on Keyword search at https://kijito.ai/account; a scoped API key "
                              "cannot enrol). "
                              f"GET /api/fts-blind/status says: {bs}"}
    try:
        questions, split_path = _load_split(split, 0 if qids_file else n, seed)
    except Exception as e:
        return {"status": "skipped", "reason": f"could not load {_REPO}/{_SPLITS.get(split, split)}: {type(e).__name__}: {e}"}
    # --qids-file: an explicit, committed question list (a frozen dev or held-out split) REPLACES the
    # stratified sample; --n is ignored. Order follows the file. An id missing from the split refuses.
    qids_sha = None
    if qids_file:
        raw = open(qids_file, "rb").read()
        qids_sha = hashlib.sha256(raw).hexdigest()[:12]
        wanted = [x.strip() for x in raw.decode().splitlines() if x.strip() and not x.startswith("#")]
        by_id = {q.get("question_id"): q for q in questions}
        missing = [x for x in wanted if x not in by_id]
        if missing:
            return {"status": "skipped",
                    "reason": f"--qids-file lists {len(missing)} ids absent from split {split!r} "
                              f"(first: {missing[0]!r})"}
        questions = [by_id[x] for x in wanted]

    # --shard I/N: run only every N-th question (index I of N slices of the SAME deterministic list).
    # Each question is isolated in its own project, so slices are independent and N shard processes
    # divide the ingest wall clock by ~N. Each shard uses its OWN --items-log (the sentinel folds the
    # shard spec, so logs can't cross-resume); context-cache keys are per question and shard-independent.
    shard_idx = shard_n = None
    if shard:
        try:
            shard_idx, shard_n = (int(x) for x in str(shard).split("/", 1))
            if not (0 <= shard_idx < shard_n):
                raise ValueError
        except ValueError:
            return {"status": "skipped",
                    "reason": f"--shard must be I/N with 0 <= I < N (got {shard!r})"}
        questions = questions[shard_idx::shard_n]

    manifest = run_manifest(seed=seed, dataset_paths=[split_path])
    manifest["dataset"] = {"repo": _REPO, "file": _SPLITS[split], "revision": _REVISION,
                           "license": "MIT (LongMemEval)"}
    if qids_file:
        manifest["qids_file"] = {"path": qids_file, "sha256_12": qids_sha, "n": len(questions)}
    manifest["blind_index"] = bool(blind)
    manifest["kijito_arm"] = api_info
    if "ollama" in (agent_provider, judge_provider):
        from .llm import ollama_settings
        manifest["ollama"] = ollama_settings()
    if no_append_md:
        manifest["append_md_arm"] = "skipped (--no-append-md)"
    manifest["judge_model"] = judge_model
    manifest["judge_provider"] = judge_provider
    manifest["official_judge_model"] = _OFFICIAL_JUDGE
    # judge != the official pin => absolute numbers are directional (an OpenRouter id
    # 'openai/gpt-4o-2024-08-06' IS the pin, routed vendor-only with fallbacks off).
    manifest["judge_matches_official"] = judge_model.split("/", 1)[-1] == _OFFICIAL_JUDGE
    # The cache digest: the oracle and s/m splits SHARE question ids, so the split and the dataset
    # file's CONTENT digest (not its machine-specific path) key every cached context.
    ds_digest = hashlib.sha256("|".join(sorted(
        str(v) for v in (manifest.get("dataset_sha256") or {}).values())).encode()).hexdigest()[:16]
    digest = f"split:{split}|dataset:{ds_digest}"
    cache = ContextCache(context_cache) if context_cache else None
    # Official-protocol provenance: every scorecard produced under the official per-type judge carries
    # this marker, so official and harness-judge numbers can never be silently mixed.
    if official_eval:
        manifest = {**manifest, "official_eval": True, "protocol": "v2-official-pertype"}
    # Shard provenance: the scorecard's aggregates cover the shard's question subset; items must be
    # merged across all N shards for run-level numbers.
    if shard_n:
        manifest = {**manifest, "shard": f"{shard_idx}/{shard_n}", "shard_questions": len(questions)}

    # Checkpoint/resume: validated BEFORE the retrieval phase (a mismatched log must refuse cheaply,
    # not after the full ingest). The log is BOUND to everything that changes a logged outcome.
    resume_cfg_parts = [
        model, agent_provider, k, seed, split, bool(official_eval),
        json.dumps(manifest.get("dataset_sha256") or {}, sort_keys=True),
        # The judge grades every logged `ok`, and the prompt texts + the harness code produce every
        # logged answer/verdict: a resume across any of them would ship a mixed-provenance arm.
        f"judge={judge_provider}:{judge_model}",
        f"prompts={_prompt_digest(official_eval)}",
        f"code={manifest.get('git_commit')}:{_code_state()}",
    ]
    cfg_shard = f"shard={shard_idx}/{shard_n}" if shard_n else "shard=off"
    if shard_n:
        resume_cfg_parts += [cfg_shard]
    # A question list is provenance: a log written for one list never resumes under another.
    if qids_sha:
        resume_cfg_parts += [f"qids={qids_sha}"]
    if blind:
        resume_cfg_parts += ["blind=on"]
    # Answers depend on WHICH server retrieved (host + commit + namespace + reply form) and on how its
    # reply is rendered for the reader.
    resume_cfg_parts += [f"api={api_tag}", f"api_ctx={api_ctx}"]
    cfg_fields = (f"model/agent-provider/k/seed/split/official-eval/dataset/judge-model/judge-provider/"
                  f"prompts/git-commit+dirty/{cfg_shard}")
    if qids_sha:
        cfg_fields += "/qids-file"
    cfg_fields += "/api(host,server-commit,prefix,recall-explain,ctx-render)"
    resume_cfg = hashlib.sha256(
        "\x1f".join(map(str, resume_cfg_parts)).encode()).hexdigest()[:16]
    resume = ItemLog.load(items_log) if items_log else {}
    sentinel = resume.pop(("__config__", "__config__"), None)
    # Errored records (transport failure, retrieval failure) are NOT reused — they re-run (the
    # _score_arm resume rule); say how many so a resume's spend is not a surprise.
    n_errored = sum(1 for r in resume.values() if r.get("error"))
    if n_errored:
        print(f"[longmemeval] resume: {n_errored} errored record(s) in {items_log!r} will RE-RUN "
              f"(only clean records are reused)", file=sys.stderr)
    if resume and sentinel is None:
        return {"status": "skipped",
                "reason": f"items log {items_log!r} has no config sentinel; refusing to reuse records "
                          f"of unknown provenance — use a fresh --items-log"}
    if sentinel is not None and sentinel.get("category") != resume_cfg:
        return {"status": "skipped",
                "reason": f"items log {items_log!r} was written under a DIFFERENT run config "
                          f"({cfg_fields}); resuming would serve stale scores under this manifest "
                          f"— use a fresh --items-log"}
    log = ItemLog(items_log) if items_log else None
    if log is not None and sentinel is None:
        log.append(qid="__config__", arm="__config__", category=resume_cfg)

    # Retrieval is serial; only the reader/judge scoring loop parallelizes.
    # retr_errors[arm] = {qid: message} for every retrieval that RAISED (never cached, never answered).
    retr_errors = {"kijito": {}, "append_md": {}}
    _API_CANARY_STATE["api"] = None
    timings = {}
    api_leaks = {}
    api_ingest_stats = {}
    renders = {}
    trace = _Trace(api_trace) if api_trace else None
    try:
        kctx = _api_contexts(questions, k, api, cache=cache, digest=digest,
                             errors=retr_errors["kijito"], api_tag=api_tag,
                             project_prefix=api_project_prefix, trace=trace, timings=timings,
                             leaks=api_leaks, explain=api_explain, ingest_tag=api_ingest_tag,
                             ingest_stats=api_ingest_stats, blind=blind)
        renders["kijito"] = _api_render(api_ctx)
        ctx = {"kijito": kctx}
        if not no_append_md:
            ctx["append_md"] = _md_contexts(questions, k, cache=cache, digest=digest,
                                            errors=retr_errors["append_md"])
    except (_CanaryFailed, MemoryCapReached) as e:
        print(f"\n[longmemeval] RUN REFUSED — {e}\n", file=sys.stderr)
        return {"status": "refused", "reason": f"REFUSED: {e}", "manifest": manifest}
    api_info["scope_canary"] = (_API_CANARY_STATE["api"]
                                or "not run (no live API retrieval: all cached; cached leak verdicts re-asserted)")
    api_info["calls"] = {"total": api.stats.calls, "retries": api.stats.retries,
                         "by_tool": dict(api.stats.by_tool)}
    api_info["ingest"] = dict(api_ingest_stats)   # {fresh, reused} among the live retrievals

    # `resume`/`log` were loaded + config-validated above (before retrieval); new records append as we
    # go so a killed run resumes exactly where it stopped without re-spending on completed items.
    t0 = time.time()
    arms = {arm: _score_arm(agent_provider, model, questions, c, arm=arm, item_log=log, resume=resume,
                            official=official_eval, judge_provider=judge_provider,
                            judge_model=judge_model, retr_error_qids=set(retr_errors.get(arm, {})),
                            render=renders.get(arm))
            for arm, c in ctx.items()}
    wall = time.time() - t0
    for arm, m in arms.items():
        m["retrieval_errors"] = len(retr_errors.get(arm, {}))
    total_retr_errors = sum(m["retrieval_errors"] for m in arms.values())
    if api_leaks:
        print(f"\n[longmemeval] !!! RUN INVALID: {len(api_leaks)} question(s) recalled memories from "
              f"OUTSIDE their project (cross-question leak): {sorted(api_leaks)}\n", file=sys.stderr)
    if total_retr_errors:
        print(f"\n[longmemeval] !!! RUN INVALID: {total_retr_errors} retrieval error(s) "
              f"({', '.join(a + '=' + str(m['retrieval_errors']) for a, m in arms.items() if m['retrieval_errors'])})"
              f" — those questions had NO retrieval and are scored as misses; the scorecard is written "
              f"with status='invalid' and must NOT be published. Fix and resume (they re-run).\n",
              file=sys.stderr)

    # Cost block: answer-call and judge-call tokens tracked separately and priced at their OWN model's
    # rate (a judge different from the reader would otherwise be priced at the reader's rate).
    ans_in = sum(a["answer_in_tok"] for a in arms.values())
    ans_out = sum(a["answer_out_tok"] for a in arms.values())
    jin = sum(a["judge_in_tok"] for a in arms.values())
    jout = sum(a["judge_out_tok"] for a in arms.values())
    cost = {"llm_calls": sum(a["llm_calls"] for a in arms.values()),
            "in_tokens": ans_in + jin, "out_tokens": ans_out + jout,
            "answer_in_tokens": ans_in, "answer_out_tokens": ans_out,
            "judge_in_tokens": jin, "judge_out_tokens": jout,
            "answer_usd": round(_cost(model, ans_in, ans_out), 4),
            "judge_usd": round(_cost(judge_model, jin, jout), 4),
            "usd": round(_cost(model, ans_in, ans_out) + _cost(judge_model, jin, jout), 4),
            "wall_clock_s": round(wall, 2)}

    # Paired kijito-vs-append_md stats, aligned by qid so McNemar/bootstrap see the same questions.
    stats = None
    if "append_md" in arms:
        kb = {it["qid"]: it["ok"] for it in arms["kijito"]["_items"]}
        mb = {it["qid"]: it["ok"] for it in arms["append_md"]["_items"]}
        qids = [q for q in (it["qid"] for it in arms["kijito"]["_items"]) if q in kb and q in mb]
        stats = paired_stats([kb[q] for q in qids], [mb[q] for q in qids], seed=seed)

    for a in arms.values():   # _items was internal scaffolding for the stats block — don't ship it
        a.pop("_items", None)

    # When a cache is active, disclose it in the manifest so a scorecard always states whether retrieval
    # was live or replayed (path + hit/miss counts).
    if cache is not None:
        manifest = {**manifest, "context_cache": {"path": context_cache, **cache.stats()}}

    sc = {
        "status": "invalid" if (total_retr_errors or api_leaks) else "ran",
        "config": {"split": split, "n": len(questions), "top_k": k, "model": model, "seed": seed,
                   "dream": False,
                   "agent_provider": agent_provider,
                   "judge_model": judge_model, "judge_provider": judge_provider,
                   "judge": f"{judge_provider}:{judge_model}"},
        "manifest": manifest,
        "arms": arms,
        "retrieval_errors": total_retr_errors,
        # per-question retrieval wall time of the kijito arm (namespace probe + ingest + recall).
        # Reader+judge wall time per item is in the items log (elapsed_s).
        "retrieval_timing": {"kijito": timings},
        "cost": cost,
        "stats": stats,   # paired kijito-vs-append_md: McNemar exact + bootstrap CI
    }
    sc["api_leaks"] = api_leaks
    return sc


def _print(sc: dict) -> None:
    if sc.get("status") not in ("ran", "invalid"):
        print(f"\nLONGMEMEVAL: {sc.get('reason')}\n"); return
    c = sc["config"]
    if sc.get("status") == "invalid":
        print(f"\n{'!' * 78}\nLONGMEMEVAL RUN INVALID — {sc.get('retrieval_errors')} retrieval error(s); "
              f"DO NOT PUBLISH these numbers.\n{'!' * 78}")
    print(f"\nLONGMEMEVAL ({c['split']} split) — Kijito vs append-markdown, "
          f"reader={c.get('agent_provider')}:{c['model']}")
    local = c.get("judge_provider", c.get("agent_provider", "ollama")) == "ollama"
    judge_note = ("LOCAL judge (directional, not leaderboard)" if local else
                  f"judge={c.get('judge')} ({'official per-type prompts' if (sc.get('manifest') or {}).get('official_eval') else 'harness v1 prompt'})")
    print(f"n={c['n']} questions | top_k={c['top_k']} | {judge_note}\n")
    man = sc.get("manifest") or {}
    ka = man.get("kijito_arm") or {}
    if ka.get("mode") == "api":
        print(f"kijito arm: VIA API {ka.get('base_host')} (server commit {ka.get('server_commit')}, "
              f"recall {'explain=true' if ka.get('recall_explain') else 'default form'}, "
              f"reader gets {ka.get('ctx_render')}) | scope canary: {ka.get('scope_canary')} | "
              f"leaks: {len(sc.get('api_leaks') or {})} | calls: {(ka.get('calls') or {}).get('total')} "
              f"(retries {(ka.get('calls') or {}).get('retries')})")
    print("arm".ljust(14) + "accuracy".rjust(9) + "  task-avg" + "  abstain(n)" + "   (correct/n)"
          + "  retr-err" + "  ctx-tok mean/p50")
    print("-" * 96)
    for arm, m in sc["arms"].items():
        ab = m.get("abstention_accuracy")
        ab_s = f"{ab:.4f}({m.get('abstention_n', 0)})" if ab is not None else "   -"
        print(arm.ljust(14) + f"{m['accuracy']:9.4f}" + f"{m.get('task_averaged_accuracy', 0.0):10.4f}"
              + f"  {ab_s:>10s}" + f"   ({m['correct']}/{m['n']})".ljust(14)
              + f"{m.get('retrieval_errors', 0):8d}"
              + f"  {m.get('ctx_tokens_mean', 0)}/{m.get('ctx_tokens_p50', 0)}")
    tk = next(iter(sc["arms"].values()), {}).get("ctx_tokenizer")
    if tk:
        print(f"(ctx tokens measured with {tk})")
    print("\nby question_type (kijito vs append_md):")
    ktypes = sc["arms"]["kijito"]["by_type"]
    mtypes = (sc["arms"].get("append_md") or {}).get("by_type", {})
    for t in sorted(set(ktypes) | set(mtypes)):
        md_s = f"{mtypes[t]:.2f}" if t in mtypes else "  -"
        print(f"  {t:26s} kijito {ktypes.get(t, 0.0):.2f}  vs  append_md {md_s}")
    st = sc.get("stats")
    if st:
        print(f"\npaired kijito vs append_md (n={st['n']}): delta={st['delta']:+.4f}  95%CI={st['ci95']}  "
              f"McNemar p={st['mcnemar_p']:.4f} (b={st['b']}, c={st['c']})")
    co = sc.get("cost")
    if co:
        print(f"cost: {co['llm_calls']} llm calls | in={co['in_tokens']} out={co['out_tokens']} tok | "
              f"${co['usd']:.4f} | {co['wall_clock_s']:.1f}s wall")
        if "answer_in_tokens" in co:
            print(f"  answer calls: in={co['answer_in_tokens']} out={co['answer_out_tokens']} "
                  f"${co.get('answer_usd', 0.0):.4f} | judge calls: in={co['judge_in_tokens']} "
                  f"out={co['judge_out_tokens']} ${co.get('judge_usd', 0.0):.4f}")
    cc = (sc.get("manifest") or {}).get("context_cache")
    if cc:
        print(f"context cache: {cc['path']} (hits={cc['hits']}, misses={cc['misses']}) "
              "— retrieval was REPLAYED for the hits, not live")
    if local:
        print("\n(LOCAL judge => directional + internally comparable, NOT the official gpt-4o judge.)\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="LongMemEval through Kijito's public API")
    ap.add_argument("--n", type=int, default=15, help="sample size (<=500; stratified by question_type)")
    ap.add_argument("--k", type=int, default=8, help="top-k retrieval budget (kijito_recall limit)")
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--model", default="qwen3.5:9b-mlx-bf16", help="reader model")
    ap.add_argument("--out", default=None)
    ap.add_argument("--items-log", default=None,
                    help="per-item JSONL checkpoint for resume (default: <out>.items.jsonl when --out is "
                         "given). On restart, already-logged (qid,arm) pairs are skipped + their outcomes reused.")
    ap.add_argument("--context-cache", default=None,
                    help="JSONL cache of retrieved contexts (keyed by dataset/qid/arm/k/server). A hit skips "
                         "that question's retrieval, so a scoring pass can be repeated over the same retrievals.")
    ap.add_argument("--agent-provider", default="ollama", choices=["ollama", "openai", "openrouter", "anthropic"],
                    help="reader provider (and the judge's, unless --judge-provider). Default local ollama.")
    ap.add_argument("--judge-model", default=None, dest="judge_model",
                    help=f"judge model (default: --model). The official LongMemEval judge is {_OFFICIAL_JUDGE} "
                         f"(OpenRouter id openai/{_OFFICIAL_JUDGE}); the manifest records judge_matches_official.")
    ap.add_argument("--judge-provider", default=None, dest="judge_provider",
                    choices=["ollama", "openai", "openrouter", "anthropic"],
                    help="judge provider (default: --agent-provider)")
    ap.add_argument("--split", choices=["oracle", "s", "m"], default="oracle",
                    help="oracle = evidence sessions only; s/m = the full distractor haystack")
    ap.add_argument("--shard", default=None,
                    help="I/N — run only questions[I::N] of the deterministic question list (0-indexed). "
                         "Parallelizes the serial per-question ingest across processes: each shard needs its "
                         "OWN --items-log (sentinel-bound to the shard spec) and its OWN --context-cache file "
                         "(per-question keys concatenate cleanly afterwards). A defaulted items-log folds the "
                         "shard into its name. Scorecard aggregates cover only the shard.")
    ap.add_argument("--blind", action="store_true",
                    help="the key's account is enrolled in the keyword index over encrypted content (an account "
                         "setting); checked with GET /api/fts-blind/status and refused if it is not")
    ap.add_argument("--qids-file", default=None,
                    help="explicit question-id list (one per line, '#' comments) replacing the stratified "
                         "sample; --n is ignored. Use it for a committed dev / held-out split (splits/). The "
                         "file's digest is bound into the resume sentinel and recorded in the manifest.")
    ap.add_argument("--official-eval", action="store_true",
                    help="grade with the OFFICIAL LongMemEval per-type yes/no judge + a non-suppressive answer "
                         "prompt, and route '_abs' question ids to the official abstention template. Every "
                         "scorecard so produced is manifest-tagged official_eval/protocol.")
    ap.add_argument("--via-api", default=DEFAULT_API, dest="via_api", metavar="BASE_URL",
                    help=f"the Kijito service (default {DEFAULT_API}): ingest with kijito_remember_batch and "
                         "recall with kijito_recall over BASE_URL/mcp/ (one project per question, recall "
                         "scope='project'). Needs --api-key-file (a key that can read and write memories).")
    ap.add_argument("--api-key-file", default=None, dest="api_key_file",
                    help="file holding the Kijito API key (must be mode 0600)")
    ap.add_argument("--api-ctx", default="mcp_text", choices=["mcp_text", "contents"], dest="api_ctx",
                    help="what the reader gets — mcp_text (default) = exactly the kijito_recall reply an agent "
                         "reads; contents = the memory bodies from that reply, in order, as '- <note>' lines. "
                         "Same retrieval (cache-shared) either way.")
    ap.add_argument("--api-project-prefix", default="", dest="api_project_prefix",
                    help="project = <prefix><question_id>. A project that already holds memories is refused "
                         "(it would double the haystack), so a re-ingest needs a fresh prefix.")
    ap.add_argument("--api-explain", action="store_true", dest="api_explain",
                    help="recall with explain=true (the DETAILED reply form: score/scope/importance header, "
                         "belief line, full UNTRUSTED fence, scoring breakdown) instead of the default (lean) "
                         "form. Bound into the manifest, the resume sentinel and the reply cache key; with "
                         "--context-cache an ingest already written by a run of the other form is REUSED.")
    ap.add_argument("--no-append-md", action="store_true", dest="no_append_md",
                    help="skip the append_md (BM25 over the same notes) control arm; no paired stats")
    ap.add_argument("--api-trace", default=None, dest="api_trace",
                    help="per-question JSONL trace (timings, recalled ids, leak verdicts); "
                         "default <out>.api-trace.jsonl when --out is given")
    args = ap.parse_args(argv)
    if not args.api_key_file:
        ap.error("--api-key-file is required (the Kijito API key; see README)")
    if args.n < 1:
        ap.error("--n must be >= 1")
    # Default the checkpoint alongside the scorecard so `--out foo.json` is resumable out of the box.
    # Shard-aware default: N shard processes sharing one --out would otherwise all default to the SAME
    # .items.jsonl. An EXPLICIT --items-log is honored as given.
    if args.items_log:
        items_log = args.items_log
    elif args.out:
        _shard_tag = f".shard{args.shard.replace('/', '-')}" if args.shard else ""
        items_log = f"{args.out}{_shard_tag}.items.jsonl"
    else:
        items_log = None
    sc = run(args.n, args.k, args.model, args.seed, split=args.split,
             agent_provider=args.agent_provider, items_log=items_log, context_cache=args.context_cache,
             official_eval=args.official_eval, shard=args.shard,
             qids_file=args.qids_file, judge_model=args.judge_model, judge_provider=args.judge_provider,
             via_api=args.via_api, api_key_file=args.api_key_file, api_ctx=args.api_ctx,
             api_project_prefix=args.api_project_prefix, no_append_md=args.no_append_md,
             api_explain=args.api_explain, blind=args.blind,
             api_trace=args.api_trace or (f"{args.out}.api-trace.jsonl" if (args.out and args.via_api) else None))
    _print(sc)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w").write(json.dumps(sc, indent=2))
        print(f"[scorecard written to {args.out}]")
    # An invalid (retrieval errors) or refused (canary) run exits non-zero so a driver script sees it.
    return 1 if sc.get("status") in ("invalid", "refused") else 0


if __name__ == "__main__":
    sys.exit(main())
