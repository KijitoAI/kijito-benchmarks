"""GroupMemBench — memory over multi-party enterprise chat (arXiv 2605.14498, NeurIPS 2026).

Repo: github.com/UCSB-NLP-Chang/GroupMemBench (release e2682e0). Four domains (Finance, Technology,
Healthcare, Manufacturing), each ONE synthetic multi-channel group log of 30,000 messages; 745 questions
in six types (multi_hop, knowledge_update, temporal, user_implicit, term_ambiguity, abstention). Every
question of a domain is asked against that domain's whole log.

OFFICIAL PROTOCOL (repo baselines/rag_common/eval_lib.py + paper §4.1 / App. I), mirrored here:
- Corpus = the domain JSON, flattened and sorted by (channel, timestamp, msg_node).
- Retrieval query = f"{asking_user_id} {question}"; top-k = 10 (RETRIEVE_TOP_K).
- Passage shown to the reader = "[user=… / speaker_role=… / channel=… / phase_name=… / topic=… /
  timestamp=… / reply_to=… / msg_node=…]\\n<content>", numbered "[i] " and joined by blank lines.
- Reader = system prompt prompts/hipporag_agent_system.txt + user message "Asking user: …\\n\\nQuestion:
  …\\n\\nRetrieved passages:\\n…\\n\\nAnswer the question using the retrieved passages.", max_tokens 512
  (gpt-5: max_completion_tokens 2048, no temperature). The "Final:" line is the answer.
- Judge = system prompt prompts/hipporag_judge_system.txt + "Question:\\n…\\n\\nGold Answer:\\n…\\n\\n
  Agent Answer:\\n<final line>\\n", max_tokens 256 (gpt-5: 2000). Reader and judge default to gpt-5.
- Score = the paper's App. I rule (eval_lib.parse_judgment on the judge's "Final" line: negatives first,
  then "correct", else Unclear) with Unclear EXCLUDED from the denominator; per-type accuracy pools the
  domains and the overall number pools questions. Table 2's "Average" is that pooled number (BM25
  309/715 = 43.22, Hindsight 329/715 = 46.01, MemGPT 201/714 = 28.15 with one Unclear dropped —
  checked by back-computing every cell). The repo's summarize_typed_eval.py rule (strict "correct",
  Unclear counted as a miss) is reported alongside.

KNOWN GAP vs the published table: Table 2 pools 715 questions; the repo ships 745. The 30 extra are all
abstention (Healthcare 34 vs the paper's 19, Manufacturing 48 vs 33 — the README says those two domains
ship the unfiltered set; which ids were filtered is not published). Every non-abstention column, and all
of Finance + Technology, match the paper's denominators exactly. The scorecard therefore reports the
pooled 745 number, the exactly-comparable columns, and a disclosed reweighted estimate of the 715 pool.

Arms (same reader, judge, prompts and k; only retrieval differs):
- kijito : through Kijito's public API (--via-api, default https://api.kijito.ai, plus --api-key-file).
           Every message, rendered as the official reader passage (--ingest-format passage, the default),
           in corpus order, is written with kijito_remember_batch (<= 60 items per call) into ONE PROJECT
           PER DOMAIN (<prefix><Domain>) of one account; each question is one kijito_recall (query = the
           official search query, scope='project', limit=k). The reader's "Retrieved passages:" block is
           the kijito_recall reply text EXACTLY as an agent reads it (--api-ctx mcp_text, the default);
           --api-ctx contents numbers the parsed bodies the official way, to separate formatting from
           retrieval. A scope canary runs before the first live domain and every recall is leak-checked
           (a memory from another project marks the run invalid). --blind declares that the key's ACCOUNT
           is enrolled in the keyword index over encrypted content (an account setting, not per project;
           GET /api/fts-blind/status is checked and recorded).
- bm25   : the repo's own BM25 baseline (rank_bm25.BM25Okapi over content-only tokens, re-implemented
           here byte-for-byte so no untrusted code runs and no new dependency is needed).

Run: python -m kijito_bench.groupmembench --data-root <clone> --api-key-file KEYFILE --api-project-prefix gmb-run1- [--n 5] …
"""
import argparse
import hashlib
import html
import json
import math
import os
import random
import re
import statistics
import sys
import time

from .llm import _cost, _llm_chat, context_tokens, tokenizer_name
from .runner import ContextCache, ItemLog, _sha256_file, pmap, paired_stats, run_manifest

DOMAINS = ("Finance", "Technology", "Healthcare", "Manufacturing")
QTYPES = ("multi_hop", "knowledge_update", "temporal", "user_implicit", "term_ambiguity", "abstention")
DEFAULT_API = "https://api.kijito.ai"
# A clone of github.com/UCSB-NLP-Chang/GroupMemBench at REPO_COMMIT (see the README).
DEFAULT_ROOT = os.environ.get("GROUPMEMBENCH_ROOT", os.path.expanduser("~/.cache/groupmembench"))
OFFICIAL_READER = "gpt-5"
OFFICIAL_JUDGE = "gpt-5"
OFFICIAL_K = 10
REPO_COMMIT = "e2682e01ff490acfe4fac2940159dce60307dfc9"

# Verbatim prompts (repo prompts/*.txt, read with .strip() by eval_lib.read_text). At run time they are
# compared against the clone's files and the result is recorded in the manifest.
AGENT_SYSTEM = (
    "You are a careful QA agent. Use only the retrieved conversation passages to answer the question.\n"
    "First provide a chain-of-thought reasoning paragraph. Then provide the final answer on a new line "
    "using the format:\nFinal: <answer>\n"
    "If the answer is not supported by the retrieved passages, say so in the Final line.")
JUDGE_SYSTEM = (
    "You are a strict judge evaluating whether an agent's answer matches the gold answer for a question.\n"
    "Consider paraphrases correct if they have the same meaning as the gold answer.\n"
    "First provide a brief reasoning paragraph. Then provide the final judgment on a new line using the "
    "format:\nFinal: Correct\nor\nFinal: Incorrect")
AGENT_MAX_TOKENS = 512
JUDGE_MAX_TOKENS = 256

# Paper Table 2 (cross-domain, GPT-5 reader + GPT-5 judge, gpt-4o-mini ingestion LLM for the memory
# systems, text-embedding-3-large for dense parts). Per-type pooled accuracy, percent.
PUBLISHED_TABLE2 = {
    "BM25": {"multi_hop": 40.11, "knowledge_update": 25.23, "term_ambiguity": 14.15,
             "user_implicit": 40.82, "temporal": 54.94, "abstention": 77.98, "average": 43.22},
    "text-embedding-3-large": {"multi_hop": 36.26, "knowledge_update": 23.36, "term_ambiguity": 21.70,
                               "user_implicit": 46.94, "temporal": 32.72, "abstention": 75.23,
                               "average": 38.04},
    "GraphRAG": {"multi_hop": 12.09, "knowledge_update": 14.02, "term_ambiguity": 19.81,
                 "user_implicit": 14.29, "temporal": 5.56, "abstention": 66.97, "average": 20.56},
    "Mem0": {"multi_hop": 21.98, "knowledge_update": 4.67, "term_ambiguity": 11.32,
             "user_implicit": 20.41, "temporal": 16.67, "abstention": 82.57, "average": 25.73},
    "MemGPT": {"multi_hop": 22.53, "knowledge_update": 17.76, "term_ambiguity": 20.75,
               "user_implicit": 28.57, "temporal": 12.42, "abstention": 77.98, "average": 28.15},
    "A-Mem": {"multi_hop": 35.16, "knowledge_update": 22.43, "term_ambiguity": 26.42,
              "user_implicit": 46.94, "temporal": 23.46, "abstention": 67.89, "average": 35.10},
    "HippoRAG": {"multi_hop": 39.56, "knowledge_update": 27.10, "term_ambiguity": 30.19,
                 "user_implicit": 42.86, "temporal": 29.63, "abstention": 75.23, "average": 39.72},
    "Hindsight": {"multi_hop": 42.31, "knowledge_update": 17.76, "term_ambiguity": 37.74,
                  "user_implicit": 40.82, "temporal": 54.94, "abstention": 77.06, "average": 46.01},
}
# Denominators the paper's Table 2 pooled (back-computed; they reproduce every published cell).
PAPER_N_BY_TYPE = {"multi_hop": 182, "knowledge_update": 107, "term_ambiguity": 106,
                   "user_implicit": 49, "temporal": 162, "abstention": 109}
PAPER_ABSTENTION_N = {"Technology": 28, "Finance": 29, "Healthcare": 19, "Manufacturing": 33}
PAPER_TOTAL = 715
REPO_TOTAL = 745

# $/1M tokens (input, output) from openrouter.ai/api/v1/models, 2026-10-23. Unlisted => _cost's table.
_PRICES = {"gpt-5": (1.25, 10.0), "gpt-5-mini": (0.25, 2.0), "gpt-4o": (2.5, 10.0),
           "gpt-4o-mini": (0.15, 0.6)}


def _usd(model, in_tok, out_tok):
    base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", (model or "").split("/", 1)[-1])
    if base in _PRICES:
        pin, pout = _PRICES[base]
        return in_tok / 1e6 * pin + out_tok / 1e6 * pout
    return _cost(model, in_tok, out_tok)


# ── official loaders + formatting (re-implemented from baselines/rag_common/eval_lib.py) ──────────

def load_conversation_messages(path):
    """eval_lib.load_conversation_messages: every channel's messages, tagged _channel, sorted by
    (channel, timestamp, msg_node). Message indices into this list are the retrieval unit."""
    data = json.load(open(path, encoding="utf-8"))
    messages = []
    if isinstance(data, dict):
        for channel, msgs in data.items():
            if not isinstance(msgs, list):
                continue
            for m in msgs:
                if isinstance(m, dict):
                    m2 = dict(m)
                    m2["_channel"] = channel
                    messages.append(m2)
    messages.sort(key=lambda m: (m.get("_channel", ""), m.get("timestamp", ""), m.get("msg_node", "")))
    return messages


def message_index_text(message):
    content = message.get("content", "")
    if not isinstance(content, str):
        return ""
    return html.unescape(content).strip()


def format_retrieved_message(message):
    author = message.get("author") or "?"
    tags = [f"user={author}"]
    for src_key, label in (("role", "speaker_role"), ("_channel", "channel"), ("phase_name", "phase_name"),
                           ("topic", "topic"), ("timestamp", "timestamp"), ("reply_to", "reply_to"),
                           ("msg_node", "msg_node")):
        value = message.get(src_key)
        if value not in (None, ""):
            tags.append(f"{label}={value}")
    body = message.get("content", "")
    body = html.unescape(body).strip() if isinstance(body, str) else ""
    return f"[{' / '.join(tags)}]\n{body}".strip()


def format_attributed_message(message):
    """The --ingest-format attributed alternative: who / where / when + content only (the paper's
    memory-system ingestion tuple, plus channel), without the generator's phase/topic labels."""
    return (f"[{message.get('timestamp', '')}] #{message.get('_channel', '')} "
            f"{message.get('author') or '?'} ({message.get('role') or '?'}): {message_index_text(message)}")


INGEST_FORMATS = {"passage": format_retrieved_message, "attributed": format_attributed_message}


def extract_final(text):
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        low = ln.lower()
        if low.startswith("final:") or low.startswith("final answer:"):
            return ln.split(":", 1)[1].strip()
    return lines[-1] if lines else text.strip()


def split_reasoning_and_final(text):
    if not text:
        return "", ""
    lines = [ln.rstrip() for ln in text.splitlines()]
    final_idx = None
    for i in range(len(lines) - 1, -1, -1):
        low = lines[i].strip().lower()
        if low.startswith("final:") or low.startswith("final answer:"):
            final_idx = i
            break
    if final_idx is None:
        return text.strip(), extract_final(text)
    return "\n".join(lines[:final_idx]).strip(), lines[final_idx].split(":", 1)[1].strip()


def parse_judgment(text):
    """eval_lib.parse_judgment (the paper's App. I rule): negatives first, then 'correct', else None."""
    final = extract_final(text).strip().lower()
    if "incorrect" in final or "wrong" in final or "not correct" in final:
        return False
    if "correct" in final:
        return True
    return None


_NORM_RE = re.compile(r"[^a-zA-Z]+")


def summarizer_is_correct(judge_answer):
    """task_synthesis/summarize_typed_eval._is_correct — the rule that produces the accuracy tables."""
    return _NORM_RE.sub("", (judge_answer or "")).lower() == "correct"


def search_query(q):
    uid = q.get("asking_user_id") or ""
    return f"{uid} {q['question']}" if uid else q["question"]


def numbered_passages(retrieved_docs):
    """eval_lib's passage block: "[i] <doc>" joined by blank lines."""
    return "\n\n".join(f"[{i}] {doc}" for i, doc in enumerate(retrieved_docs, start=1))


def agent_user_prompt(q, retrieved_docs):
    return agent_user_prompt_block(q, numbered_passages(retrieved_docs))


def agent_user_prompt_block(q, passages):
    """The official reader user message around an already-rendered passages block (the --via-api arm
    hands the kijito_recall reply text here instead of numbered passages)."""
    uid = q.get("asking_user_id") or ""
    asker = f"Asking user: {uid}\n\n" if uid else ""
    return (f"{asker}Question:\n{q['question']}\n\nRetrieved passages:\n{passages}\n\n"
            "Answer the question using the retrieved passages.")


def judge_user_prompt(q, agent_final):
    return f"Question:\n{q['question']}\n\nGold Answer:\n{q.get('answer', '')}\n\nAgent Answer:\n{agent_final}\n"


# ── official BM25 (rank_bm25 0.2.2 BM25Okapi, k1=1.5 b=0.75 epsilon=0.25) ─────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text):
    return [t.lower() for t in _TOKEN_RE.findall(text)] if text else []


class OfficialBM25:
    """rank_bm25.BM25Okapi semantics: idf = ln(N - n + .5) - ln(n + .5); negative idfs are replaced by
    epsilon * mean(idf); a repeated query token scores once per occurrence. Ranking = the baseline's
    sorted(range(N), key=score, reverse=True)[:k] — stable, so ties (incl. all-zero) keep corpus order."""

    def __init__(self, corpus_tokens, k1=1.5, b=0.75, epsilon=0.25):
        self.k1, self.b = k1, b
        self.n = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0
        self.postings = {}
        nd = {}
        for i, doc in enumerate(corpus_tokens):
            freqs = {}
            for w in doc:
                freqs[w] = freqs.get(w, 0) + 1
            for w, f in freqs.items():
                self.postings.setdefault(w, []).append((i, f))
                nd[w] = nd.get(w, 0) + 1
        self.idf, neg, idf_sum = {}, [], 0.0
        for w, f in nd.items():
            v = math.log(self.n - f + 0.5) - math.log(f + 0.5)
            self.idf[w] = v
            idf_sum += v
            if v < 0:
                neg.append(w)
        eps = epsilon * (idf_sum / len(self.idf)) if self.idf else 0.0
        for w in neg:
            self.idf[w] = eps

    def get_scores(self, query_tokens):
        scores = [0.0] * self.n
        for q in query_tokens:
            idf = self.idf.get(q) or 0
            for i, f in self.postings.get(q, ()):
                scores[i] += idf * (f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)))
        return scores

    def top_k(self, query_tokens, k):
        if not query_tokens:
            return []   # the baseline returns no passages for an empty query
        s = self.get_scores(query_tokens)
        return sorted(range(self.n), key=lambda i: s[i], reverse=True)[:k]


# ── dataset ───────────────────────────────────────────────────────────────────────────────────

def conversation_path(root, domain):
    return os.path.join(root, "data", "final", domain, f"synthetic_domain_channels_rolevariants_{domain}.json")


def question_path(root, domain, qtype):
    return os.path.join(root, "questions", domain, f"{qtype}.jsonl")


def load_questions(root, domains=DOMAINS, qtypes=QTYPES):
    out = []
    for d in domains:
        for t in qtypes:
            p = question_path(root, d, t)
            if not os.path.exists(p):
                continue
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    q = json.loads(line)
                    out.append({**q, "domain": d, "qtype": t, "qid": f"{d}/{q['id']}"})
    return out


def stratified_sample(questions, n, seed):
    """Deterministic round-robin over question types (domains interleaved within each type), so a small
    n covers the types and spreads over domains. Same seed -> same sample. n<=0 or >=len => everything,
    in file order."""
    if n <= 0 or n >= len(questions):
        return list(questions)
    rng = random.Random(seed)
    by_type = {}
    for q in questions:
        by_type.setdefault(q["qtype"], {}).setdefault(q["domain"], []).append(q)
    queues = {}
    for t, doms in by_type.items():
        order = [d for d in DOMAINS if d in doms] + sorted(d for d in doms if d not in DOMAINS)
        for d in order:
            rng.shuffle(doms[d])
        rng.shuffle(order)
        lists, inter = [doms[d] for d in order], []
        while any(lists):
            for lst in lists:
                if lst:
                    inter.append(lst.pop())
        queues[t] = inter
    types = [t for t in QTYPES if t in queues] + sorted(t for t in queues if t not in QTYPES)
    i, picked = rng.randrange(len(types)), []
    while len(picked) < n:
        t = types[i % len(types)]
        if queues[t]:
            picked.append(queues[t].pop(0))
        i += 1
    return picked


def dataset_digest(root, domains):
    """sha256 over the conversation + question files of the domains in play (content, not mtime)."""
    parts = {}
    for d in domains:
        parts[conversation_path(root, d)] = _sha256_file(conversation_path(root, d))
        for t in QTYPES:
            p = question_path(root, d, t)
            if os.path.exists(p):
                parts[p] = _sha256_file(p)
    h = hashlib.sha256(json.dumps(sorted((os.path.relpath(k, root), v) for k, v in parts.items())).encode())
    return h.hexdigest(), parts


# ── retrieval arms ────────────────────────────────────────────────────────────────────────────

# ── the Kijito arm, through the public API ──────────────────────────────────────────────────────────
# Cache value shape: [<the kijito_recall reply text>, <json meta>] per question. How the reply
# reaches the reader is decided at scoring time (--api-ctx). meta carries the project, the recalled ids and
# the leak verdict, so a cache hit re-asserts the leak check. A completed domain ingest is also cached as
# an INGEST RECORD (keyed by host|commit|prefix + ingest format, without the reply form) holding the
# project and the ids written: a crash after the (long) ingest resumes with recalls only, and the same
# ingest can be read again in another reply form.

_API_CANARY_STATE = {"api": None}


class _APIRefused(RuntimeError):
    """The scope canary refused the run (project-scoped recall crossed projects)."""


def kijito_api_contexts(questions, messages_by_domain, k, api, fmt="passage", cache=None, key_fn=None,
                        ingest_key_fn=None, project_prefix="", explain=False, trace=None, leaks=None,
                        timings=None, ingest_stats=None):
    """({qid: [reply_text, meta_json]}, {qid: error}) through Kijito's MCP endpoint. Per domain with a
    cache miss: refuse a project that already holds memories, write every message (INGEST_FORMATS[fmt],
    corpus order) with kijito_remember_batch, then per missed question one kijito_recall (official search
    query, scope='project', limit=k) and a leak check. Errors are returned, never cached."""
    from .client import MemoryCapReached, leak_check, parse_recall, scope_canary
    out, errors, misses = {}, {}, {}
    if ingest_stats is not None:
        ingest_stats.setdefault("fresh", 0)
        ingest_stats.setdefault("reused", 0)
    for q in questions:
        key = key_fn(q, "kijito") if (cache is not None and key_fn) else None
        hit = cache.get(key) if key is not None else None
        if hit is not None and len(hit) == 2:
            out[q["qid"]] = hit
            meta = json.loads(hit[1])
            if meta.get("leaks") and leaks is not None:
                leaks[q["qid"]] = meta["leaks"]
        else:
            misses.setdefault(q["domain"], []).append((q, key))
    if not misses:
        return out, errors
    scope_canary(api, project_prefix, exc=_APIRefused, label="gmb")
    _API_CANARY_STATE["api"] = "passed"
    cap_msg = None
    for domain, items in misses.items():
        if cap_msg is not None:
            for q, _ in items:
                errors[q["qid"]] = cap_msg
            continue
        project = f"{project_prefix}{domain}"
        calls0, retries0 = api.stats.calls, api.stats.retries
        ikey = ingest_key_fn(domain) if (cache is not None and ingest_key_fn) else None
        irec = cache.peek(ikey) if ikey else None
        t0 = time.time()
        reused = False
        try:
            if irec:
                im = json.loads(irec[0])
                if im.get("project") != project or im.get("n_notes") != len(messages_by_domain[domain]):
                    raise RuntimeError(f"ingest record for {domain} does not match this run "
                                       f"({im.get('project')!r}, {im.get('n_notes')} notes)")
                ids, reused = list(im["own_ids"]), True
            else:
                if parse_recall(api.recall("benchmark namespace probe", limit=1, scope="project",
                                           project=project)):
                    raise RuntimeError(f"project {project!r} already holds memories (an earlier run?); "
                                       f"use a fresh --api-project-prefix")
                texts = [INGEST_FORMATS[fmt](m) for m in messages_by_domain[domain]]
                ids = []
                step = 3000   # progress lines only; remember_batch splits into tool-legal batches
                for s0 in range(0, len(texts), step):
                    ids += api.remember_batch(texts[s0:s0 + step], project)
                    print(f"[gmb] API ingest {domain}: {len(ids)}/{len(texts)} in {time.time() - t0:.0f}s",
                          file=sys.stderr, flush=True)
                if len(ids) != len(texts):
                    raise RuntimeError(f"stored {len(ids)} of {len(texts)} messages")
                if ikey:
                    cache.put(ikey, [json.dumps({"project": project, "own_ids": ids, "n_notes": len(texts)})])
        except Exception as e:
            msg = f"API ingest {domain}: {type(e).__name__}: {e}"
            print(f"[gmb] kijito {msg}", file=sys.stderr)
            for q, _ in items:
                errors[q["qid"]] = msg
            if isinstance(e, MemoryCapReached):
                # The account is full: the remaining domains are recorded as errors without writing.
                cap_msg = msg
                print("[gmb] STOPPED: the account's memory cap is reached (see the README: memory "
                      "footprint per arm).", file=sys.stderr)
            continue
        t_ing = time.time() - t0
        if ingest_stats is not None:
            ingest_stats["reused" if reused else "fresh"] += 1
        if timings is not None:
            timings[f"ingest:{domain}"] = {"n_notes": len(ids), "ingest_s": round(t_ing, 1), "reused": reused}
        print(f"[gmb] kijito API {'reused ingest of' if reused else 'ingested'} {domain}: {len(ids)} memories "
              f"in {t_ing:.0f}s", file=sys.stderr)
        own = set(ids)
        for q, ckey in items:
            tr = time.time()
            try:
                text = api.recall(search_query(q), limit=k, scope="project", project=project, explain=explain)
                got = parse_recall(text)
                q_leaks = leak_check(api, got, own, project)
            except Exception as e:
                errors[q["qid"]] = f"recall: {type(e).__name__}: {e}"
                print(f"[gmb] kijito API recall failed for {q['qid']}: {errors[q['qid']]}", file=sys.stderr)
                continue
            if q_leaks:
                print(f"[gmb] LEAK: {q['qid']} recall returned memories from outside project {project!r}: "
                      f"{q_leaks} (run will be marked INVALID)", file=sys.stderr)
                if leaks is not None:
                    leaks[q["qid"]] = {str(i): v for i, v in q_leaks.items()}
            meta = {"qid": q["qid"], "project": project, "recalled_ids": [g.id for g in got],
                    "leaks": {str(i): v for i, v in q_leaks.items()}, "recall_explain": bool(explain),
                    "reply_form": sorted({g.form for g in got}), "ingest_reused": reused}
            ctx = [text, json.dumps(meta)]
            out[q["qid"]] = ctx
            if timings is not None:
                timings[q["qid"]] = {"recall_s": round(time.time() - tr, 3)}
            if trace is not None:
                trace.append({"qid": q["qid"], "project": project, "recalled": meta["recalled_ids"],
                              "leaks": meta["leaks"], "recall_s": round(time.time() - tr, 3)})
            if ckey is not None:
                cache.put(ckey, ctx)
        if trace is not None:
            trace.append({"domain": domain, "project": project, "ingest_s": round(t_ing, 1), "n_ids": len(ids),
                          "ingest_reused": reused, "calls": api.stats.calls - calls0,
                          "retries": api.stats.retries - retries0})
    return out, errors


def api_render(mode):
    """render(q, ctx) -> (reader user message, passages block, n memories, docs for --results-dir) for an
    API retrieval. mcp_text: the passages block is the kijito_recall reply verbatim; contents: the parsed
    bodies, numbered the official way (numbered_passages) — the bm25 arm's rendering."""
    from .client import reply_bodies

    def render(q, ctx):
        text = ctx[0] if ctx else ""
        bodies = reply_bodies(text)
        block = numbered_passages(bodies) if mode == "contents" else text
        return agent_user_prompt_block(q, block), block, len(bodies), bodies
    return render


def bm25_contexts(questions, messages_by_domain, k, cache=None, key_fn=None):
    out, errors, indexes = {}, {}, {}
    for q in questions:
        key = key_fn(q, "bm25") if (cache is not None and key_fn) else None
        hit = cache.get(key) if key is not None else None
        if hit is not None:
            out[q["qid"]] = hit
            continue
        try:
            msgs = messages_by_domain[q["domain"]]
            if q["domain"] not in indexes:
                indexes[q["domain"]] = OfficialBM25([tokenize(message_index_text(m)) for m in msgs])
            idx = indexes[q["domain"]].top_k(tokenize(search_query(q)), k)
            got = [format_retrieved_message(msgs[i]) for i in idx]
        except Exception as e:
            errors[q["qid"]] = f"bm25: {type(e).__name__}: {e}"
            continue
        out[q["qid"]] = got
        if key is not None:
            cache.put(key, got)
    return out, errors


# ── scoring ───────────────────────────────────────────────────────────────────────────────────

_EXTRA = ("domain", "agent_answer", "agent_reasoning", "judge_answer", "judge_reasoning", "verdict",
          "ans_in", "ans_out", "judge_in", "judge_out", "ctx_tokens", "n_ctx")


def score_arm(arm, questions, ctxs, retrieval_errors, agent_provider, model, judge_provider, judge_model,
              item_log=None, resume=None, render=None):
    """Reader + judge per question, the official way. Retrieval-error and scoring-error items are
    recorded with `error`, scored as misses, and NOT written to the items log, so a rerun retries them.
    `render(q, ctx)` (api_render) replaces the numbered-passages rendering for the --via-api arm."""
    resume = resume or {}

    def one(q):
        qid = q["qid"]
        base = {"qid": qid, "arm": arm, "category": q["qtype"], "domain": q["domain"]}
        if qid in retrieval_errors:
            return {**base, "ok": False, "verdict": None, "error": f"retrieval: {retrieval_errors[qid]}",
                    "ans_in": 0, "ans_out": 0, "judge_in": 0, "judge_out": 0, "ctx_tokens": 0, "n_ctx": 0}
        if (qid, arm) in resume:
            r = resume[(qid, arm)]
            return {**base, **{f: r.get(f) for f in _EXTRA if f in r}, "ok": bool(r.get("ok")),
                    "error": None, "resumed": True}
        if render is not None:
            user, block, n_ctx, _ = render(q, ctxs[qid])
        else:
            docs = ctxs[qid]
            user, block, n_ctx = agent_user_prompt(q, docs), numbered_passages(docs), len(docs)
        rec = {**base, "ctx_tokens": context_tokens(block),
               "n_ctx": n_ctx, "ans_in": 0, "ans_out": 0, "judge_in": 0, "judge_out": 0}
        try:
            text, ai, ao = _llm_chat(agent_provider, model, AGENT_SYSTEM, user, max_tokens=AGENT_MAX_TOKENS)
            reasoning, final = split_reasoning_and_final(text or "")
            rec.update(ans_in=ai, ans_out=ao, agent_reasoning=reasoning, agent_answer=final)
            jtext, ji, jo = _llm_chat(judge_provider, judge_model, JUDGE_SYSTEM, judge_user_prompt(q, final),
                                      max_tokens=JUDGE_MAX_TOKENS)
            jreason, jfinal = split_reasoning_and_final(jtext or "")
            v = parse_judgment(jfinal)
            rec.update(judge_in=ji, judge_out=jo, judge_reasoning=jreason, judge_answer=jfinal,
                       verdict={True: "Correct", False: "Incorrect"}.get(v, "Unclear"),
                       ok=summarizer_is_correct(jfinal), error=None)
        except Exception as e:
            print(f"[gmb] scoring failed for {qid}/{arm}: {type(e).__name__}: {e}", file=sys.stderr)
            rec.update(ok=False, verdict=None, error=f"scoring: {type(e).__name__}: {e}")
            return rec
        if item_log is not None:
            item_log.append(qid=qid, arm=arm, category=q["qtype"], ok=rec["ok"],
                            hyp_len=len(rec.get("agent_answer") or ""),
                            in_tok=rec["ans_in"] + rec["judge_in"], out_tok=rec["ans_out"] + rec["judge_out"],
                            error=None, **{f: rec.get(f) for f in _EXTRA})
        return rec

    recs = []
    for q, r in zip(questions, pmap(one, questions)):
        if isinstance(r, Exception):
            r = {"qid": q["qid"], "arm": arm, "category": q["qtype"], "domain": q["domain"], "ok": False,
                 "verdict": None, "error": f"runner: {type(r).__name__}: {r}"}
        recs.append(r)
    return recs


def _acc(c, n):
    return round(c / n, 4) if n else None


def official_metrics(records):
    """The paper's scoring (App. I, and what reproduces Table 2): verdict by eval_lib.parse_judgment;
    "Unclear" is EXCLUDED from the denominator; per-type accuracy pools domains; the overall number pools
    questions (Table 2 "Average"). Evidence the exclusion is real: MemGPT's Temporal 12.42 = 20/161 and
    its Average 28.15 = 201/714, one Unclear dropped. Also reported: the repo summarizer's rule (strict
    "correct" match, everything else a miss over all n), per-domain tables, macro-over-types, and — only
    when the full 745 set was scored — a disclosed reweighted estimate of the paper's 715 pool."""
    cell = {}
    for r in records:
        c = cell.setdefault((r["domain"], r["category"]), {"c": 0, "n": 0, "unclear": 0, "c_sum": 0})
        c["n"] += 1
        c["c_sum"] += bool(r["ok"])
        if r.get("verdict") == "Unclear":
            c["unclear"] += 1
        if r.get("verdict") == "Correct":
            c["c"] += 1

    def pool(pred):
        s = {"c": 0, "n": 0, "unclear": 0, "c_sum": 0}
        for (d, t), v in cell.items():
            if pred(d, t):
                for f in s:
                    s[f] += v[f]
        return s

    def block(s):
        return {"accuracy": _acc(s["c"], s["n"] - s["unclear"]), "correct": s["c"],
                "scored": s["n"] - s["unclear"], "n": s["n"], "unclear": s["unclear"],
                "summarizer_rule_accuracy": _acc(s["c_sum"], s["n"])}

    by_type = {t: block(pool(lambda d, tt, t=t: tt == t)) for t in QTYPES
               if pool(lambda d, tt, t=t: tt == t)["n"]}
    allp = pool(lambda d, t: True)
    by_domain = {}
    for d in DOMAINS:
        s = pool(lambda dd, t, d=d: dd == d)
        if s["n"]:
            by_domain[d] = {**block(s), "by_type": {t: block(v) for (dd, t), v in sorted(cell.items()) if dd == d}}
    accs = [v["accuracy"] for v in by_type.values() if v["accuracy"] is not None]
    out = {"overall_pooled": _acc(allp["c"], allp["n"] - allp["unclear"]), "correct": allp["c"],
           "scored": allp["n"] - allp["unclear"], "n": allp["n"], "unclear": allp["unclear"],
           "overall_summarizer_rule": _acc(allp["c_sum"], allp["n"]),
           "macro_over_types": round(statistics.mean(accs), 4) if accs else None,
           "by_type": by_type, "by_domain": by_domain}
    if allp["n"] == REPO_TOTAL and all(t in by_type for t in QTYPES):
        exact = pool(lambda d, t: not (t == "abstention" and d in ("Healthcare", "Manufacturing")))
        c_est, n_est = float(exact["c"]), float(exact["n"] - exact["unclear"])
        for d in ("Healthcare", "Manufacturing"):
            v = cell.get((d, "abstention"))
            if v and v["n"] - v["unclear"]:
                c_est += v["c"] / (v["n"] - v["unclear"]) * PAPER_ABSTENTION_N[d]
                n_est += PAPER_ABSTENTION_N[d]
        ab = pool(lambda d, t: t == "abstention" and d in ("Finance", "Technology"))
        ab_c, ab_n = float(ab["c"]), float(ab["n"] - ab["unclear"])
        for d in ("Healthcare", "Manufacturing"):
            v = cell.get((d, "abstention"))
            if v and v["n"] - v["unclear"]:
                ab_c += v["c"] / (v["n"] - v["unclear"]) * PAPER_ABSTENTION_N[d]
                ab_n += PAPER_ABSTENTION_N[d]
        out["paper_pool_estimate"] = {
            "average_715": round(c_est / n_est, 4) if n_est else None,
            "abstention_109": round(ab_c / ab_n, 4) if ab_n else None,
            "note": ("ESTIMATE: the repo ships 34/48 Healthcare/Manufacturing abstention questions where the "
                     "paper pooled 19/33 and the filtered ids are unpublished; HC/Mfg abstention accuracy is "
                     "applied to the paper's counts. Every other cell is exactly the paper's question set.")}
    return out


def _token_block(records, model, judge_model):
    ai = sum(r.get("ans_in") or 0 for r in records)
    ao = sum(r.get("ans_out") or 0 for r in records)
    ji = sum(r.get("judge_in") or 0 for r in records)
    jo = sum(r.get("judge_out") or 0 for r in records)
    ctx = [r["ctx_tokens"] for r in records if r.get("ctx_tokens") is not None and not r.get("error")]
    return {
        "answer": {"in_tokens": ai, "out_tokens": ao, "usd": round(_usd(model, ai, ao), 4)},
        "judge": {"in_tokens": ji, "out_tokens": jo, "usd": round(_usd(judge_model, ji, jo), 4)},
        "context_tokens_per_query": ({"mean": round(statistics.mean(ctx), 1), "median": statistics.median(ctx),
                                      "max": max(ctx), "tokenizer": tokenizer_name()} if ctx else None),
    }


# ── run ───────────────────────────────────────────────────────────────────────────────────────

def _base_model(model):
    return (model or "").split("/", 1)[-1]


def _prompt_check(root):
    res = {}
    for name, const in (("hipporag_agent_system.txt", AGENT_SYSTEM), ("hipporag_judge_system.txt", JUDGE_SYSTEM)):
        p = os.path.join(root, "prompts", name)
        res[name] = (open(p, encoding="utf-8").read().strip() == const) if os.path.exists(p) else None
    return res


def run(root=DEFAULT_ROOT, n=0, k=OFFICIAL_K, model=OFFICIAL_READER, judge_model=None,
        agent_provider="openrouter", judge_provider=None, seed=20261024, domains=DOMAINS, qtypes=QTYPES,
        arms=("kijito", "bm25"), items_log=None, context_cache=None, blind=False, ingest_format="passage",
        shard=None, qids_file=None, results_dir=None, via_api=None, api_key_file=None,
        api_ctx="mcp_text", api_project_prefix="", api_explain=False, api_trace=None, retrieval_only=False,
        _api_client=None):
    judge_model = judge_model or model
    judge_provider = judge_provider or agent_provider
    for prov in {agent_provider, judge_provider}:
        envk = {"openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY"}.get(prov)
        if envk and not os.environ.get(envk):
            return {"status": "skipped", "reason": f"{envk} not set (needed for provider {prov!r})"}
    if not os.path.isdir(os.path.join(root, "questions")):
        return {"status": "skipped", "reason": f"GroupMemBench clone not found at {root!r}"}
    # The kijito arm runs through a Kijito service's MCP endpoint (kijito_api_contexts).
    api = api_info = api_tag = api_ingest_tag = None
    if "kijito" in arms:
        if not (via_api or _api_client is not None):
            return {"status": "skipped", "reason": "the kijito arm runs through the API: pass --via-api BASE_URL "
                                                   "and --api-key-file (or --arms bm25)"}
        if api_ctx not in ("mcp_text", "contents"):
            return {"status": "skipped", "reason": f"--api-ctx must be mcp_text or contents (got {api_ctx!r})"}
        from .client import api_arm_info, blind_status, connect
        api, why = connect(via_api, api_key_file, _api_client)
        if api is None:
            return {"status": "skipped", "reason": why}
        api_info = api_arm_info(api, via_api, ctx_render=api_ctx, k=k, project_prefix=api_project_prefix,
                                explain=api_explain, blind=blind, project_per="domain")
        lim = api_info.get("recall_limit_max")
        if lim is not None and k > lim:
            print(f"[gmb] WARNING: k={k} exceeds kijito_recall's declared limit maximum {lim}; the service "
                  f"may return fewer than k memories.", file=sys.stderr)
        api_ingest_tag = (f"{api_info['base_host']}|commit={api_info['server_commit']}"
                          f"|prefix={api_project_prefix}")
        api_tag = f"{api_ingest_tag}|recall_explain={'on' if api_explain else 'off'}"
        if blind:
            bs = blind_status(api)
            api_info["blind"]["status"] = bs
            if not (bs.get("active") and bs.get("index_exists")):
                return {"status": "refused",
                        "reason": "REFUSED: --blind needs the key's account enrolled in the keyword index over "
                                  "encrypted content (turn on Keyword search at https://kijito.ai/account; a scoped API key "
                                  "cannot enrol). "
                                  f"GET /api/fts-blind/status says: {bs}"}

    questions = load_questions(root, domains, qtypes)
    qids_sha = None
    if qids_file:
        raw = open(qids_file, "rb").read()
        qids_sha = hashlib.sha256(raw).hexdigest()[:12]
        wanted = [x.strip() for x in raw.decode().splitlines() if x.strip() and not x.startswith("#")]
        by_id = {q["qid"]: q for q in questions}
        missing = [x for x in wanted if x not in by_id]
        if missing:
            return {"status": "skipped", "reason": f"--qids-file lists {len(missing)} unknown ids (first {missing[0]!r})"}
        questions = [by_id[x] for x in wanted]
    else:
        questions = stratified_sample(questions, n, seed)
    shard_idx = shard_n = None
    if shard:
        try:
            shard_idx, shard_n = (int(x) for x in str(shard).split("/", 1))
            if not 0 <= shard_idx < shard_n:
                raise ValueError
        except ValueError:
            return {"status": "skipped", "reason": f"--shard must be I/N with 0 <= I < N (got {shard!r})"}
        questions = questions[shard_idx::shard_n]
    if not questions:
        return {"status": "skipped", "reason": "no questions selected"}

    used_domains = sorted({q["domain"] for q in questions}, key=DOMAINS.index)
    ds_sha, ds_files = dataset_digest(root, used_domains)
    manifest = run_manifest(seed=seed, dataset_paths=list(ds_files))
    retrieval_extra = f"fmt={ingest_format}|blind={'on' if blind else 'off'}"
    cache = ContextCache(context_cache) if context_cache else None

    def key_fn(q, arm):
        # bm25 contexts depend only on the data; the kijito arm's key binds the ingest format and the
        # SERVER (host + commit + namespace + reply form) that produced the retrieval.
        extra = retrieval_extra if arm == "kijito" else "official-bm25-okapi"
        if arm == "kijito" and api_tag:
            extra += f"|api={api_tag}"
        return ContextCache.make_key("groupmembench", q["qid"], arm, k, ds_sha[:16], extra=extra)

    def ingest_key_fn(domain):
        return ContextCache.make_key("groupmembench", f"ingest:{domain}", "kijito_api_ingest", 0, ds_sha[:16],
                                     extra=f"fmt={ingest_format}|api={api_ingest_tag}")

    # Resume sentinel: every input that changes an answer or a verdict.
    parts = [model, agent_provider, judge_model, judge_provider, k, ingest_format, bool(blind), ds_sha,
             f"shard={shard}" if shard else "shard=off", f"qids={qids_sha}" if qids_sha else "qids=off",
             hashlib.sha256((AGENT_SYSTEM + JUDGE_SYSTEM).encode()).hexdigest()[:12]]
    # Answers depend on WHICH server retrieved and on how its reply is rendered for the reader.
    if api_tag:
        parts += [f"api={api_tag}", f"api_ctx={api_ctx}"]
    cfg = hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:16]
    resume = ItemLog.load(items_log) if items_log else {}
    sentinel = resume.pop(("__config__", "__config__"), None)
    if resume and sentinel is None:
        return {"status": "skipped", "reason": f"items log {items_log!r} has no config sentinel; use a fresh log"}
    if sentinel is not None and sentinel.get("category") != cfg:
        return {"status": "skipped", "reason": (f"items log {items_log!r} was written under a different config "
                                                "(model/providers/k/ingest-format/blind/dataset/"
                                                "shard/qids/prompts"
                                                f"{'/api(host,server-commit,prefix,recall-explain,ctx-render)' if api_tag else ''}"
                                                "); use a fresh --items-log")}
    log = ItemLog(items_log) if items_log else None
    if log is not None and sentinel is None:
        log.append(qid="__config__", arm="__config__", category=cfg)

    # Load conversations only for domains whose contexts are not all cached.
    need = set()
    for q in questions:
        for arm in arms:
            if cache is None or key_fn(q, arm) not in cache._mem:
                need.add(q["domain"])
    messages_by_domain = {d: load_conversation_messages(conversation_path(root, d)) for d in sorted(need)}

    t0 = time.time()
    ctx, rerr, renders = {}, {}, {}
    api_leaks, api_timings, api_ingest = {}, {}, {}
    if "kijito" in arms and api is not None:
        from .client import MemoryCapReached, Trace
        _API_CANARY_STATE["api"] = None
        try:
            ctx["kijito"], rerr["kijito"] = kijito_api_contexts(
                questions, messages_by_domain, k, api, fmt=ingest_format, cache=cache, key_fn=key_fn,
                ingest_key_fn=ingest_key_fn, project_prefix=api_project_prefix, explain=api_explain,
                trace=Trace(api_trace) if api_trace else None, leaks=api_leaks, timings=api_timings,
                ingest_stats=api_ingest)
        except (_APIRefused, MemoryCapReached) as e:
            print(f"\n[gmb] RUN REFUSED — {e}\n", file=sys.stderr)
            return {"status": "refused", "reason": f"REFUSED: {e}", "manifest": manifest}
        api_info["scope_canary"] = (_API_CANARY_STATE["api"] or "not run (no live API retrieval: all "
                                    "cached; cached leak verdicts re-asserted)")
        api_info["calls"] = {"total": api.stats.calls, "retries": api.stats.retries,
                             "by_tool": dict(api.stats.by_tool)}
        api_info["ingest"] = dict(api_ingest)
        renders["kijito"] = api_render(api_ctx)
    if "bm25" in arms:
        ctx["bm25"], rerr["bm25"] = bm25_contexts(questions, messages_by_domain, k, cache=cache, key_fn=key_fn)
    retrieval_wall = time.time() - t0
    if retrieval_only:
        # --retrieval-only: fill the context cache (and the API trace) and stop before any reader or judge
        # call; a later run with the same cache scores those exact retrievals.
        if cache is not None:
            manifest["context_cache"] = {"path": context_cache, **cache.stats()}
        if api_info:
            manifest["kijito_arm"] = api_info
        out = {"status": "retrieval-only", "n": len(questions),
               "retrieved": {a: len(c) for a, c in ctx.items()}, "retrieval_errors": rerr,
               "retrieval_wall_s": round(retrieval_wall, 1), "manifest": manifest}
        if api is not None:
            out["api_leaks"] = api_leaks
            out["retrieval_timing"] = api_timings
        return out

    t1 = time.time()
    records = {a: score_arm(a, questions, ctx[a], rerr[a], agent_provider, model, judge_provider, judge_model,
                            item_log=log, resume=resume, render=renders.get(a)) for a in arms}
    scoring_wall = time.time() - t1

    arms_out = {}
    for a, recs in records.items():
        arms_out[a] = {**official_metrics(recs),
                       "retrieval_errors": len(rerr[a]),
                       "scoring_errors": sum(1 for r in recs if (r.get("error") or "").startswith(("scoring", "runner"))),
                       "resumed": sum(1 for r in recs if r.get("resumed")),
                       "tokens": _token_block(recs, model, judge_model)}
    stats = None
    if "kijito" in records and "bm25" in records:
        kb = {r["qid"]: r["ok"] for r in records["kijito"]}
        bb = {r["qid"]: r["ok"] for r in records["bm25"]}
        common = [q["qid"] for q in questions if q["qid"] in kb and q["qid"] in bb]
        stats = paired_stats([kb[x] for x in common], [bb[x] for x in common], seed=seed)

    if results_dir:   # official output schema, consumable by task_synthesis/summarize_typed_eval.py
        qmap, files = {q["qid"]: q for q in questions}, {}
        for a, recs in records.items():
            for r in recs:
                q = qmap[r["qid"]]
                p = os.path.join(results_dir, q["domain"], f"{a}__{q['qtype']}.jsonl")
                docs, extra_cols = ctx[a].get(r["qid"], []), {}
                if a in renders and docs:   # API arm: the parsed bodies, plus the reply the reader saw
                    extra_cols = {"mcp_reply": docs[0]}
                    docs = renders[a](q, docs)[3]
                files.setdefault(p, []).append({
                    "id": q["id"], "query": q["question"], "asking_user_id": q.get("asking_user_id") or "",
                    "retrieved_docs": docs, **extra_cols,
                    "agent_reasoning": r.get("agent_reasoning") or "", "agent_answer": r.get("agent_answer") or "",
                    "judge_reasoning": r.get("judge_reasoning") or "", "judge_answer": r.get("judge_answer") or "",
                    "error": r.get("error")})
        for p, rows in files.items():
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    n_err = sum(a["retrieval_errors"] + a["scoring_errors"] for a in arms_out.values())
    if api_leaks:
        print(f"\n[gmb] !!! RUN INVALID: {len(api_leaks)} question(s) recalled memories from OUTSIDE their "
              f"project: {sorted(api_leaks)}\n", file=sys.stderr)
    manifest.update({
        "benchmark": {"name": "GroupMemBench", "paper": "arXiv 2605.14498", "repo": "UCSB-NLP-Chang/GroupMemBench",
                      "repo_commit_expected": REPO_COMMIT, "data_root": root, "dataset_sha256": ds_sha,
                      "domains": used_domains, "qtypes": list(qtypes)},
        "comparability": {
            "reader_matches_official": _base_model(model) == OFFICIAL_READER,
            "judge_matches_official": _base_model(judge_model) == OFFICIAL_JUDGE,
            "k_matches_official": k == OFFICIAL_K,
            "prompts_match_repo": _prompt_check(root),
            "question_set": f"{len(questions)} of the repo's {REPO_TOTAL}; the paper's Table 2 pools {PAPER_TOTAL}",
            "paper_ingestion_llm": "gpt-4o-mini (memory systems); the kijito arm's harness makes no LLM call "
                                   "at ingest (messages are written verbatim)",
            "reader_reasoning_effort": "provider default (not set)",
        },
        "kijito_arm": ({**api_info, "ingest": "kijito_remember_batch, one memory per message",
                        "ingest_format": ingest_format, "ingest_records": api_info.get("ingest")} if api_info else
                       {"mode": "not run (--arms without kijito)"}),
        "resume_config": cfg,
    })
    if qids_sha:
        manifest["qids_file"] = {"path": qids_file, "sha256_12": qids_sha}
    if shard_n:
        manifest["shard"] = f"{shard_idx}/{shard_n}"
    if cache is not None:
        manifest["context_cache"] = {"path": context_cache, **cache.stats()}
    reasons = []
    if n_err:
        reasons.append(f"{n_err} retrieval/scoring error(s) — scored as misses; rerun with the same "
                       "--items-log/--context-cache to retry only those")
    if api_leaks:
        reasons.append(f"{len(api_leaks)} question(s) recalled memories from outside their project")
    extra_out = {"api_leaks": api_leaks, "retrieval_timing": api_timings} if api is not None else {}
    return {
        **extra_out,
        "status": "invalid" if reasons else "ran",
        "invalid_reason": "; ".join(reasons) or None,
        "config": {"n": len(questions), "k": k, "model": model, "agent_provider": agent_provider,
                   "judge_model": judge_model, "judge_provider": judge_provider, "seed": seed,
                   "arms": list(arms), "ingest_format": ingest_format, "blind": bool(blind),
                   "via_api": bool(api), "api_ctx": api_ctx if api else None},
        "manifest": manifest, "arms": arms_out, "stats_kijito_vs_bm25": stats,
        "wall_clock_s": {"retrieval": round(retrieval_wall, 1), "scoring": round(scoring_wall, 1)},
        "published_table2": PUBLISHED_TABLE2,
    }


def _print(sc):
    if sc.get("status") == "retrieval-only":
        print(f"\nGROUPMEMBENCH retrieval-only: n={sc['n']} retrieved={sc['retrieved']} "
              f"errors={ {a: len(e) for a, e in sc['retrieval_errors'].items()} } "
              f"leaks={len(sc.get('api_leaks') or {})} {sc['retrieval_wall_s']}s\n")
        return
    if sc.get("status") not in ("ran", "invalid"):
        print(f"\nGROUPMEMBENCH: {sc.get('reason')}\n")
        return
    c = sc["config"]
    comp = sc["manifest"]["comparability"]
    print(f"\nGROUPMEMBENCH status={sc['status']}  n={c['n']} k={c['k']}  reader={c['agent_provider']}:{c['model']}"
          f"  judge={c['judge_provider']}:{c['judge_model']}")
    print(f"reader official={comp['reader_matches_official']} judge official={comp['judge_matches_official']}"
          f" prompts={comp['prompts_match_repo']}")
    ka = sc["manifest"].get("kijito_arm") or {}
    if ka.get("mode") == "api":
        print(f"kijito arm: VIA API {ka.get('base_url')} (server commit {ka.get('server_commit')}, "
              f"context={ka.get('ctx_render')}, blind={'on' if ka.get('blind', {}).get('on') else 'off'}, "
              f"canary: {ka.get('scope_canary')}, leaks: {len(sc.get('api_leaks') or {})}, "
              f"calls: {(ka.get('calls') or {}).get('total')})")
    if sc.get("invalid_reason"):
        print("INVALID:", sc["invalid_reason"])
    hdr = "arm".ljust(8) + "".join(t[:10].rjust(12) for t in QTYPES) + "     pooled"
    print(hdr)
    for a, m in sc["arms"].items():
        row = a.ljust(8)
        for t in QTYPES:
            v = m["by_type"].get(t)
            row += (f"{v['correct']}/{v['n']}".rjust(12) if v else "—".rjust(12))
        row += f"   {m['overall_pooled']:.4f} ({m['correct']}/{m['n']})"
        print(row)
        tb = m["tokens"]
        print(f"         answer {tb['answer']}  judge {tb['judge']}  ctx {tb['context_tokens_per_query']}")
        if m.get("paper_pool_estimate"):
            print(f"         paper-pool estimate: {m['paper_pool_estimate']['average_715']}")
    if sc.get("stats_kijito_vs_bm25"):
        print("paired kijito vs bm25:", sc["stats_kijito_vs_bm25"])
    print("published Table 2 averages:", {k: v["average"] for k, v in PUBLISHED_TABLE2.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description="GroupMemBench through Kijito's public API (official protocol)")
    ap.add_argument("--data-root", default=DEFAULT_ROOT,
                    help="GroupMemBench clone (data/, questions/, prompts/); default $GROUPMEMBENCH_ROOT or "
                         "~/.cache/groupmembench")
    ap.add_argument("--n", type=int, default=0, help="stratified subset size over (type, domain); 0 = all")
    ap.add_argument("--k", type=int, default=OFFICIAL_K)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--agent-provider", default="openrouter", choices=["ollama", "openai", "openrouter", "anthropic"])
    ap.add_argument("--model", default=f"openai/{OFFICIAL_READER}", help="reader model (official: gpt-5)")
    ap.add_argument("--judge-provider", default=None, choices=["ollama", "openai", "openrouter", "anthropic"],
                    help="default: --agent-provider")
    ap.add_argument("--judge-model", default=None, help="default: --model (official: gpt-5)")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--qtypes", default=",".join(QTYPES))
    ap.add_argument("--arms", default="kijito,bm25")
    ap.add_argument("--ingest-format", choices=sorted(INGEST_FORMATS), default="passage",
                    help="passage = the official reader passage (what the reader sees is exactly what Kijito "
                         "stored and returned); attributed = timestamp/channel/author/role + content")
    ap.add_argument("--blind", action="store_true",
                    help="the key's account is enrolled in the keyword index over encrypted content (an account "
                         "setting); checked with GET /api/fts-blind/status and refused if it is not")
    ap.add_argument("--qids-file", default=None, help="explicit 'Domain/id' list; replaces --n")
    ap.add_argument("--shard", default=None, help="I/N: run questions[I::N]")
    ap.add_argument("--out", default=None)
    ap.add_argument("--items-log", default=None, help="default <out>[.shardI-N].items.jsonl")
    ap.add_argument("--context-cache", default=None)
    ap.add_argument("--results-dir", default=None, help="also write official-schema results/<Domain>/<arm>__<qtype>.jsonl")
    ap.add_argument("--via-api", default=DEFAULT_API, dest="via_api", metavar="BASE_URL",
                    help=f"the Kijito service (default {DEFAULT_API}): kijito_remember_batch ingest into one "
                         "project per domain, kijito_recall (scope=project, limit=k) per question. Needs "
                         "--api-key-file.")
    ap.add_argument("--api-key-file", default=None, dest="api_key_file",
                    help="file holding the Kijito API key (a key that can read and write memories; mode 0600)")
    ap.add_argument("--api-ctx", default="mcp_text", choices=["mcp_text", "contents"], dest="api_ctx",
                    help="the reader's passages — mcp_text (default) = the kijito_recall reply verbatim; "
                         "contents = the parsed bodies, numbered the official way (like the bm25 arm)")
    ap.add_argument("--api-project-prefix", default="", dest="api_project_prefix",
                    help="project = <prefix><Domain>; a project that already holds memories is refused")
    ap.add_argument("--api-explain", action="store_true", dest="api_explain",
                    help="recall with explain=true (the DETAILED reply form)")
    ap.add_argument("--api-trace", default=None, dest="api_trace",
                    help="JSONL trace (per-question recalled ids/leaks, per-domain ingest); default "
                         "<out>.api-trace.jsonl when --out is given")
    ap.add_argument("--retrieval-only", action="store_true", dest="retrieval_only",
                    help="retrieve (fill --context-cache) and stop before any reader or judge call")
    args = ap.parse_args(argv)
    if "kijito" in args.arms.split(",") and not args.api_key_file:
        ap.error("--api-key-file is required for the kijito arm (the Kijito API key; see README)")
    if args.n < 0:
        ap.error("--n must be >= 0")
    domains = tuple(d for d in args.domains.split(",") if d)
    qtypes = tuple(t for t in args.qtypes.split(",") if t)
    arms = tuple(a for a in args.arms.split(",") if a)
    bad = [d for d in domains if d not in DOMAINS] + [t for t in qtypes if t not in QTYPES] + \
          [a for a in arms if a not in ("kijito", "bm25")]
    if bad:
        ap.error(f"unknown domain/qtype/arm: {bad}")
    items_log = args.items_log
    if not items_log and args.out:
        items_log = f"{args.out}{'.shard' + args.shard.replace('/', '-') if args.shard else ''}.items.jsonl"
    sc = run(root=args.data_root, n=args.n, k=args.k, model=args.model, judge_model=args.judge_model,
             agent_provider=args.agent_provider, judge_provider=args.judge_provider, seed=args.seed,
             domains=domains, qtypes=qtypes, arms=arms, items_log=items_log, context_cache=args.context_cache,
             blind=args.blind, ingest_format=args.ingest_format, shard=args.shard, qids_file=args.qids_file,
             results_dir=args.results_dir, via_api=args.via_api,
             api_key_file=args.api_key_file, api_ctx=args.api_ctx, api_project_prefix=args.api_project_prefix,
             api_explain=args.api_explain, retrieval_only=args.retrieval_only,
             api_trace=args.api_trace or (f"{args.out}{'.shard' + args.shard.replace('/', '-') if args.shard else ''}"
                                          f".api-trace.jsonl" if (args.out and args.via_api) else None))
    _print(sc)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w").write(json.dumps(sc, indent=2))
        print(f"[scorecard written to {args.out}]")
    if sc.get("status") == "retrieval-only":
        return 0 if not (any(sc["retrieval_errors"].values()) or sc.get("api_leaks")) else 2
    return 0 if sc.get("status") == "ran" else (2 if sc.get("status") == "invalid" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
