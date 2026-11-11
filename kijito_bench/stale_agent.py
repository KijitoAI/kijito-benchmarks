"""STALE, AGENT-CURATED arm through Kijito's public API (a disclosed arm beside the raw-ingest one, stale.py).

How Kijito is used: an agent reads each conversation and saves what is worth remembering. Here an
ingest agent (the protocol's backbone model, gpt-4o-mini by default) distills every session into dated
memories, saves them with kijito_remember_batch, reads Kijito's write reply, and — when the reply carries
update hints ("↻ may UPDATE older memory [N] …") and hints are ON — decides which hinted older memories
the new ones replace and links them with kijito_relate(..., "supersedes"). Probes then recall through
kijito_recall and are scored with the official reader/judge exactly as in stale.py.

Paired design: distillations are cached per (scenario, session) and shared by every arm, so the hints-on
and hints-off arms store IDENTICAL memories and differ only in whether the agent acts on the hints.
--mode both also stores the raw turns (as the raw arm does) before the agent's memories; hints raised by
raw-turn batches are never acted on (the agent reviews only what it wrote itself).

Nothing about the gold (M_old, M_new, relevant_session_index) reaches the agent or the system.
Run: python -m kijito_bench.stale_agent --api-key-file KEYFILE --n 20 --mode curated --hints on \
       --project-prefix P --out sc.json --distill-cache d.jsonl
"""
import argparse
import json
import os
import re
import sys
import threading
import time

from . import stale as S
from .client import BATCH_MAX_ITEMS, KijitoMCPClient, MemoryCapReached, load_key, parse_remembered_ids
from .llm import _llm
from .runner import ItemLog

DISTILL = ("You keep long-term memory for an AI assistant. Below is one chat session between the user and "
           "the assistant, dated {date}. Write the memories worth keeping, one per line:\n"
           "- facts about the USER: their situation, history, events and activities (with WHEN they "
           "happened), preferences, plans, people, places, possessions, numbers;\n"
           "- specific things the ASSISTANT told the user that they may ask about later (recommendations, "
           "names, figures, lists).\n"
           "Each line must stand alone. Start it with the date the thing happened or was said, as "
           "[YYYY-MM-DD], resolving relative dates against the session date. Keep exact names and numbers. "
           "When something about the user has changed, say what changed (e.g. 'moved from X to Y'). Skip "
           "small talk and generic advice. If nothing is worth keeping, write NONE.\n\n"
           "SESSION ({date}):\n{s}")

DECIDE = ("You just saved these memories (id: text):\n{new}\n\nThe memory system flagged older memories that "
          "some of them may UPDATE:\n{hints}\n\nLink a pair ONLY if the new memory is about the SAME specific "
          "thing as the older one (the same attribute of the same person, place or object: where the user "
          "lives, a schedule, a health condition, a job, a preference about X, a plan) AND it makes the "
          "older statement no longer true now. Two memories that merely share a topic, a day or a context "
          "are NOT a replacement; an added detail is NOT a replacement. When unsure, do not link. Reply "
          "with one line per replacement in the form 'NEW_ID -> OLD_ID', or NONE.")

_HINT_RE = re.compile(r"↻ may UPDATE older memory \[(\d+)\](?: \(([^)]*)\))?: (.*?) — if \[(\d+)\] replaces it")
_PAIR_RE = re.compile(r"\b(\d+)\s*->\s*(\d+)\b")
_lock = threading.Lock()


def sessions_chrono(item):
    sess = item.get("haystack_session") or []
    stamps = item.get("timestamps") or []
    order = sorted(range(len(sess)), key=lambda i: str(stamps[i]) if i < len(stamps) else "")
    return [(i, stamps[i] if i < len(stamps) else "", sess[i] or []) for i in order]


def load_distill_cache(path):
    c = {}
    if path and os.path.exists(path):
        for line in open(path):
            try:
                j = json.loads(line)
                c[(j["uid"], j["si"])] = j["mems"]
            except Exception:
                pass
    return c


def distill(item, si, date, sess, model, cache, cache_path, cost):
    key = (item["uid"], si)
    if key in cache:
        return cache[key]
    text = "\n".join(f"{t.get('role', '?')}: {(t.get('content') or '').strip()}" for t in sess
                     if (t.get("content") or "").strip())[:40000]
    out, i, o = _llm("openrouter", model, DISTILL.format(date=date, s=text))
    mems = [l.strip("-• ").strip() for l in (out or "").splitlines()]
    mems = [m for m in mems if m and m.upper() != "NONE"]
    with _lock:
        cost["distill_in"] += i; cost["distill_out"] += o
        cache[key] = mems
        if cache_path:
            with open(cache_path, "a") as f:
                f.write(json.dumps({"uid": item["uid"], "si": si, "mems": mems}) + "\n")
    return mems


def ingest_scenario(api, item, project, model, mode, hints_on, cache, cache_path, cost):
    stats = {"memories": 0, "raw": 0, "hints": 0, "hints_foreign": 0, "relates": 0, "decide_calls": 0}
    own = set()
    existing = api.recall("anything", limit=1, project=project, scope="project")
    if not existing.startswith("No memories") and "Found 0" not in existing:
        raise RuntimeError(f"project {project!r} already holds memories; use a fresh --project-prefix")
    for si, date, sess in sessions_chrono(item):
        if mode == "both":
            raw = [f"[{date}] {t.get('role', '?')}: {(t.get('content') or '').strip()}" for t in sess
                   if (t.get("content") or "").strip()]
            if raw:
                rid = api.remember_batch(raw, project)
                own.update(rid)
                stats["raw"] += len(rid)
        mems = distill(item, si, date, sess, model, cache, cache_path, cost)
        if not mems:
            continue
        items = [{"content": m, "importance": 0.5, "scope": "project", "project": project} for m in mems]
        # kijito_remember_batch takes at most BATCH_MAX_ITEMS memories per call (the tool's schema);
        # a long distilled session is written in consecutive chunks and the replies read together.
        ids, replies = [], []
        for k in range(0, len(items), BATCH_MAX_ITEMS):
            chunk = items[k:k + BATCH_MAX_ITEMS]
            part = api.call_tool("kijito_remember_batch", {"memories": chunk, "project": project})
            got = parse_remembered_ids(part)
            if len(got) != len(chunk):
                raise RuntimeError(f"remember_batch stored {len(got)} of {len(chunk)}: {part[:200]}")
            ids += got
            replies.append(part)
        reply = "\n".join(replies)
        stats["memories"] += len(ids)
        own.update(ids)
        hints = _HINT_RE.findall(reply)
        # Act only on hints that name a memory this scenario wrote: an update hint can point at a memory in
        # another project of the same account (another scenario), which is never this scenario's state.
        n_all = len(hints)
        hints = [h for h in hints if int(h[0]) in own]
        stats["hints_foreign"] += n_all - len(hints)
        stats["hints"] += len(hints)
        if not (hints_on and hints):
            continue
        new_txt = "\n".join(f"{i}: {m}" for i, m in zip(ids, mems))
        hint_txt = "\n".join(f"- older [{old}] ({d}): {snip} — may be updated by [{new}]"
                             for old, d, snip, new in hints)
        out, i, o = _llm("openrouter", model, DECIDE.format(new=new_txt, hints=hint_txt))
        with _lock:
            cost["decide_in"] += i; cost["decide_out"] += o
        stats["decide_calls"] += 1
        allowed = {(int(new), int(old)) for old, _d, _s, new in hints}
        for a, b in _PAIR_RE.findall(out or ""):
            pair = (int(a), int(b))
            if pair in allowed:
                api.call_tool("kijito_relate", {"from_id": pair[0], "to_id": pair[1], "rel_type": "supersedes"})
                stats["relates"] += 1
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="STALE, agent-curated arm, through Kijito's public API")
    ap.add_argument("--via-api", default=S.DEFAULT_API, help=f"the Kijito service (default {S.DEFAULT_API})")
    ap.add_argument("--api-key-file", required=True, help="file holding the Kijito API key (mode 0600)")
    ap.add_argument("--n", type=int, default=20, help="stratified (T1/T2) subset size; 0 = all 400")
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--shard", default="0/1", help="I/N: run items whose index %% N == I")
    ap.add_argument("--k", type=int, default=10, help="kijito_recall limit per probe")
    ap.add_argument("--mode", choices=["curated", "both"], default="both",
                    help="curated = only the agent's distilled memories; both = the raw turns too")
    ap.add_argument("--hints", choices=["on", "off"], default="on",
                    help="whether the ingest agent acts on the update hints in kijito_remember_batch replies")
    ap.add_argument("--ingest-model", default="openai/gpt-4o-mini",
                    help="the ingest agent's model (OpenRouter id; distillation and hint decisions)")
    ap.add_argument("--model", default="openai/gpt-4o-mini", help="reader (protocol: gpt-4o-mini)")
    ap.add_argument("--judge-model", default="google/gemini-3.1-flash-lite-preview", help="judge (protocol)")
    ap.add_argument("--agent-provider", default="openrouter", help="reader backend (anthropic for a disclosed Claude arm)")
    ap.add_argument("--judge-provider", default="openrouter")
    ap.add_argument("--project-prefix", required=True,
                    help="project = <prefix><uid>; a project that already holds memories is refused")
    ap.add_argument("--distill-cache", required=True,
                    help="JSONL cache of per-session distillations, shared by the hints-on and hints-off arms")
    ap.add_argument("--out", required=True, help="scorecard path; also writes <out>.ctx.jsonl and "
                                                 "<out>.items.jsonl")
    ap.add_argument("--uids-file", default="", help="only these scenario uids (one per line), after --n/--shard")
    ap.add_argument("--reuse-ctx", default="", help="score from a saved <out>.ctx.jsonl (no ingest/recall)")
    a = ap.parse_args(argv)
    recs, path, ds_sha = S.load_dataset()
    items = S.stratified_sample(recs, a.n, a.seed)
    si, sn = (int(x) for x in a.shard.split("/"))
    items = [it for j, it in enumerate(items) if j % sn == si]
    if a.uids_file:
        keep = {x.strip() for x in open(a.uids_file) if x.strip() and not x.startswith("#")}
        items = [it for it in items if it["uid"] in keep]
    api = KijitoMCPClient(a.via_api, load_key(a.api_key_file))
    server = api.version().get("commit")
    cache = load_distill_cache(a.distill_cache)
    cost = {"distill_in": 0, "distill_out": 0, "decide_in": 0, "decide_out": 0}
    ctxs, errs, ingest_stats = {}, {}, {}
    t0 = time.time()
    if a.reuse_ctx:
        for line in open(a.reuse_ctx):
            j = json.loads(line)
            ctxs[(j["uid"], j["dim"])] = [j["reply"]]
        items = [it for it in items if all((it["uid"], d) in ctxs for d in S._DIMS)]
    cap_error = None
    for n_done, it in enumerate([] if a.reuse_ctx else items, 1):
        if cap_error is not None:
            for d in S._DIMS:
                errs[(it["uid"], d)] = cap_error
            continue
        project = f"{a.project_prefix}{it['uid']}"
        try:
            ingest_stats[it["uid"]] = ingest_scenario(api, it, project, a.ingest_model, a.mode,
                                                      a.hints == "on", cache, a.distill_cache, cost)
            for d in S._DIMS:
                ctxs[(it["uid"], d)] = [api.recall(it["probing_queries"][f"{d}_query"], limit=a.k,
                                                   project=project, scope="project")]
        except Exception as e:
            for d in S._DIMS:
                errs[(it["uid"], d)] = f"{type(e).__name__}: {e}"
            print(f"[stale-agent] {it['uid']} failed: {type(e).__name__}: {e}", file=sys.stderr)
            if isinstance(e, MemoryCapReached):
                # The account is full: the remaining scenarios are recorded as errors without writing.
                cap_error = f"{type(e).__name__}: {e}"
                print("[stale-agent] STOPPED: the account's memory cap is reached (see the README: memory "
                      "footprint per arm).", file=sys.stderr)
        print(f"[stale-agent] {n_done}/{len(items)} {it['uid']} {ingest_stats.get(it['uid'])} "
              f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    if not a.reuse_ctx:
        with open(a.out + ".ctx.jsonl", "w") as f:
            for (u, d), c in ctxs.items():
                f.write(json.dumps({"uid": u, "dim": d, "reply": c[0]}) + "\n")
    log = ItemLog(a.out + ".items.jsonl")
    scored = S.score_items(items, ctxs, errs, (a.agent_provider, a.model), (a.judge_provider, a.judge_model),
                           item_log=log, render=lambda ctx: S.api_history(ctx, "mcp_text"))
    metrics = S.compute_metrics(scored)
    out = {"config": vars(a), "server_commit": server, "dataset_sha256": ds_sha, "ingest_cost_tokens": cost,
           "ingest_stats": ingest_stats, "metrics": metrics,
           "errors": {f"{u}:{d}": e for (u, d), e in errs.items()}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"overall": metrics.get("overall_mean_of_six"), "pooled": metrics.get("pooled_accuracy"), "settings": {k: v["accuracy"] for k, v in
                      metrics.get("settings", {}).items()}, "errors": len(errs)}, indent=1))


if __name__ == "__main__":
    main()
