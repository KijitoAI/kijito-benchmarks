"""GroupMemBench harness (kijito_bench/groupmembench.py): the official passage / prompt / parse / scoring
rules, the BM25 control, and the Kijito arm against the in-memory fake MCP endpoint, with a fake
reader/judge. No network, no LLM, no dataset download (a tiny synthetic clone is built per test)."""
import json
import os
import statistics

import pytest

import kijito_bench.groupmembench as g
from kijito_bench import llm as P
from conftest import FakeKijito, make_client

CLONE = os.environ.get("GROUPMEMBENCH_ROOT", os.path.expanduser("~/.cache/groupmembench"))


def _msg(node, content, author="User_1", role="Analyst", ts="2025-07-01T09:00:00", reply=None, **kw):
    return {"msg_node": node, "content": content, "author": author, "role": role, "timestamp": ts,
            "reply_to": reply, "phase_name": kw.get("phase", "Kickoff"), "topic": kw.get("topic", "Scope"),
            "is_noise": False}


def _make_root(tmp_path, domains=("Finance", "Healthcare")):
    root = tmp_path / "gmb"
    for d in domains:
        conv = {
            "Alpha": [_msg("Msg_2", "Budget moved to 40k &amp; approved", "User_2", ts="2025-07-03T10:00:00"),
                      _msg("Msg_1", "Budget is 30k for the vendor", "User_1", ts="2025-07-01T10:00:00")],
            "Beta": [_msg("Msg_3", "Kickoff on 2025-07-10 in room B", "User_3", ts="2025-07-02T10:00:00")],
        }
        p = root / "data" / "final" / d
        p.mkdir(parents=True)
        (p / f"synthetic_domain_channels_rolevariants_{d}.json").write_text(json.dumps(conv))
        qd = root / "questions" / d
        qd.mkdir(parents=True)
        for t in g.QTYPES:
            rows = [{"id": f"{t}_{i}", "question": f"{t} question {i} about budget",
                     "answer": "40k" if t != "abstention" else "There is no information available.",
                     "asking_user_id": "User_7"} for i in (1, 2)]
            (qd / f"{t}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    pr = root / "prompts"
    pr.mkdir()
    (pr / "hipporag_agent_system.txt").write_text(g.AGENT_SYSTEM + "\n")
    (pr / "hipporag_judge_system.txt").write_text(g.JUDGE_SYSTEM + "\n")
    return str(root)


class _FakeLLM:
    """Reader answers '40k'; judge says Correct iff the agent answer contains 40k."""

    def __init__(self):
        self.calls = []

    def __call__(self, provider, model, system, user, max_tokens=None):
        self.calls.append((provider, model, system, user, max_tokens))
        if system == g.AGENT_SYSTEM:
            return ("The later message overrides.\nFinal: 40k", 100, 10)
        ans = user.rsplit("Agent Answer:\n", 1)[1]
        return (f"Reasoning.\nFinal: {'Correct' if '40k' in ans else 'Incorrect'}", 50, 5)


# ── official formatting / prompts / parsing ──────────────────────────────────────────────────

def test_loader_sorts_by_channel_timestamp_and_tags_channel(tmp_path):
    root = _make_root(tmp_path)
    msgs = g.load_conversation_messages(g.conversation_path(root, "Finance"))
    assert [m["msg_node"] for m in msgs] == ["Msg_1", "Msg_2", "Msg_3"]
    assert msgs[0]["_channel"] == "Alpha" and msgs[2]["_channel"] == "Beta"


def test_passage_format_is_the_official_header():
    m = {**_msg("Msg_9", " a &amp; b ", "User_4", role="Health IT", reply="Msg_8"), "_channel": "EHR"}
    assert g.format_retrieved_message(m) == (
        "[user=User_4 / speaker_role=Health IT / channel=EHR / phase_name=Kickoff / topic=Scope / "
        "timestamp=2025-07-01T09:00:00 / reply_to=Msg_8 / msg_node=Msg_9]\na & b")
    m2 = {"content": "x", "author": None, "reply_to": None}
    assert g.format_retrieved_message(m2) == "[user=?]\nx"


def test_reader_and_judge_prompts_are_verbatim():
    q = {"question": "Q?", "answer": "A", "asking_user_id": "User_7"}
    assert g.search_query(q) == "User_7 Q?"
    assert g.search_query({"question": "Q?"}) == "Q?"
    assert g.agent_user_prompt(q, ["d1", "d2"]) == (
        "Asking user: User_7\n\nQuestion:\nQ?\n\nRetrieved passages:\n[1] d1\n\n[2] d2\n\n"
        "Answer the question using the retrieved passages.")
    assert g.judge_user_prompt(q, "B") == "Question:\nQ?\n\nGold Answer:\nA\n\nAgent Answer:\nB\n"


@pytest.mark.skipif(not os.path.isdir(os.path.join(CLONE, "prompts")), reason="GroupMemBench clone absent")
def test_embedded_prompts_match_the_clone():
    assert g._prompt_check(CLONE) == {"hipporag_agent_system.txt": True, "hipporag_judge_system.txt": True}


def test_final_line_parsing_and_both_judgment_rules():
    assert g.split_reasoning_and_final("think\nmore\nFinal: 2025-07-18") == ("think\nmore", "2025-07-18")
    assert g.split_reasoning_and_final("no final line\nlast") == ("no final line\nlast", "last")
    assert g.split_reasoning_and_final("") == ("", "")
    assert g.parse_judgment("Incorrect") is False and g.parse_judgment("not correct") is False
    assert g.parse_judgment("Correct") is True and g.parse_judgment("Maybe") is None
    # the summarizer (which produces the accuracy tables) is stricter than parse_judgment
    assert g.summarizer_is_correct("Correct.") and g.summarizer_is_correct(" correct ")
    assert not g.summarizer_is_correct("Correct (paraphrase)")
    assert not g.summarizer_is_correct("Incorrect") and not g.summarizer_is_correct("")


# ── official BM25 ────────────────────────────────────────────────────────────────────────────

def test_bm25_tokenizer_and_okapi_scores():
    assert g.tokenize("User_7 Budget, 30K!") == ["user_7", "budget", "30k"]
    corpus = [["a", "b"], ["a", "c"], ["a", "a", "d"]]
    bm = g.OfficialBM25(corpus)
    # 'a' is in all 3 docs => negative idf ln(0.5)-ln(3.5), replaced by 0.25 * mean idf
    import math
    idf_b = math.log(3 - 1 + 0.5) - math.log(1 + 0.5)
    mean = (math.log(0.5) - math.log(3.5) + 3 * idf_b) / 4
    assert bm.idf["a"] == pytest.approx(0.25 * mean)
    assert bm.idf["b"] == pytest.approx(idf_b)
    # repeated query token counts twice; ties keep corpus order; empty query returns nothing
    s1, s2 = bm.get_scores(["b"]), bm.get_scores(["b", "b"])
    assert s2[0] == pytest.approx(2 * s1[0])
    assert bm.top_k(["zzz"], 2) == [0, 1]
    assert bm.top_k([], 2) == []
    assert bm.top_k(["c"], 1) == [1]


# ── sampling ─────────────────────────────────────────────────────────────────────────────────

def test_stratified_sample_deterministic_and_covers_types(tmp_path):
    qs = g.load_questions(_make_root(tmp_path))
    assert len(qs) == 2 * 6 * 2
    a, b = g.stratified_sample(qs, 6, 1), g.stratified_sample(qs, 6, 1)
    assert [q["qid"] for q in a] == [q["qid"] for q in b]
    assert {q["qtype"] for q in a} == set(g.QTYPES)
    assert len(g.stratified_sample(qs, 0, 1)) == len(qs) == len(g.stratified_sample(qs, 999, 1))
    assert len({q["qid"] for q in g.stratified_sample(qs, 23, 5)}) == 23


# ── metrics ──────────────────────────────────────────────────────────────────────────────────

def test_published_averages_are_pooled_over_the_paper_denominators():
    """Table 2's Average is question-pooled over 715 with Unclear dropped from the denominator — not a
    mean of the columns. MemGPT is the one row with an Unclear (Temporal 20/161, Average 201/714)."""
    for system, row in g.PUBLISHED_TABLE2.items():
        ns = dict(g.PAPER_N_BY_TYPE)
        if system == "MemGPT":
            ns["temporal"] = 161
        correct = 0
        for t, n in ns.items():
            c = round(row[t] / 100 * n)
            assert round(100 * c / n, 2) == pytest.approx(row[t], abs=0.006), (system, t)
            correct += c
        assert round(100 * correct / sum(ns.values()), 2) == pytest.approx(row["average"], abs=0.006), system
        assert round(100 * statistics.mean(row[t] for t in g.QTYPES), 2) != row["average"], system
    assert sum(g.PAPER_N_BY_TYPE.values()) == g.PAPER_TOTAL
    assert sum(g.PAPER_ABSTENTION_N.values()) == g.PAPER_N_BY_TYPE["abstention"]


def _rec(domain, t, ok, verdict=None):
    return {"domain": domain, "category": t, "ok": ok, "verdict": verdict or ("Correct" if ok else "Incorrect")}


def test_official_metrics_paper_rule_drops_unclear_and_pools():
    recs = ([_rec("Finance", "temporal", True)] * 3 + [_rec("Finance", "temporal", False)]
            + [_rec("Healthcare", "abstention", False, "Unclear")] + [_rec("Healthcare", "abstention", True)])
    m = g.official_metrics(recs)
    assert m["n"] == 6 and m["scored"] == 5 and m["unclear"] == 1
    assert m["overall_pooled"] == 0.8, "4 correct over 5 scored (Unclear excluded)"
    assert m["overall_summarizer_rule"] == round(4 / 6, 4)
    assert m["by_type"]["temporal"]["accuracy"] == 0.75
    assert m["by_type"]["abstention"]["accuracy"] == 1.0
    assert m["by_type"]["abstention"]["summarizer_rule_accuracy"] == 0.5
    assert m["macro_over_types"] == round((0.75 + 1.0) / 2, 4)
    assert m["by_domain"]["Healthcare"]["accuracy"] == 1.0
    assert "paper_pool_estimate" not in m, "the 715 estimate needs the full 745 set"


def test_paper_pool_estimate_on_a_full_set():
    recs = []
    sizes = {"Finance": {"multi_hop": 48, "knowledge_update": 32, "temporal": 45, "user_implicit": 15,
                         "term_ambiguity": 45, "abstention": 29},
             "Technology": {"multi_hop": 41, "knowledge_update": 36, "temporal": 37, "user_implicit": 28,
                            "term_ambiguity": 43, "abstention": 28},
             "Healthcare": {"multi_hop": 48, "knowledge_update": 17, "temporal": 37, "user_implicit": 2,
                            "term_ambiguity": 7, "abstention": 34},
             "Manufacturing": {"multi_hop": 45, "knowledge_update": 22, "temporal": 43, "user_implicit": 4,
                               "term_ambiguity": 11, "abstention": 48}}
    for d, ts in sizes.items():
        for t, n in ts.items():
            recs.extend(_rec(d, t, True) for _ in range(n))
    m = g.official_metrics(recs)
    assert m["n"] == g.REPO_TOTAL and m["overall_pooled"] == 1.0
    assert m["paper_pool_estimate"]["average_715"] == 1.0
    assert m["paper_pool_estimate"]["abstention_109"] == 1.0


# ── run(): end-to-end with the BM25 arm and a fake reader/judge ──────────────────────────────

def test_run_bm25_end_to_end_resume_and_official_results(tmp_path, monkeypatch):
    root = _make_root(tmp_path)
    fake = _FakeLLM()
    monkeypatch.setattr(g, "_llm_chat", fake)
    monkeypatch.setenv("BENCH_WORKERS", "1")
    log = str(tmp_path / "items.jsonl")
    res_dir = str(tmp_path / "results")
    sc = g.run(root=root, n=0, agent_provider="ollama", model="m", arms=("bm25",), items_log=log,
               context_cache=str(tmp_path / "ctx.jsonl"), results_dir=res_dir)
    assert sc["status"] == "ran", sc
    a = sc["arms"]["bm25"]
    assert a["n"] == 24 and a["correct"] == 24 and a["overall_pooled"] == 1.0 and a["unclear"] == 0
    assert set(a["by_type"]) == set(g.QTYPES)
    assert len(fake.calls) == 48
    reader = [c for c in fake.calls if c[2] == g.AGENT_SYSTEM]
    assert all(c[4] == g.AGENT_MAX_TOKENS for c in reader)
    assert reader[0][3].startswith("Asking user: User_7\n\nQuestion:\n")
    assert "[1] [user=" in reader[0][3], "the reader sees official passages"
    tb = a["tokens"]
    assert tb["answer"]["in_tokens"] == 2400 and tb["judge"]["in_tokens"] == 1200
    assert tb["context_tokens_per_query"]["tokenizer"]
    assert sc["manifest"]["comparability"]["prompts_match_repo"] == {
        "hipporag_agent_system.txt": True, "hipporag_judge_system.txt": True}
    assert sc["manifest"]["comparability"]["judge_matches_official"] is False
    rows = [json.loads(x) for x in open(os.path.join(res_dir, "Finance", "bm25__temporal.jsonl"))]
    assert len(rows) == 2 and rows[0]["judge_answer"] == "Correct" and rows[0]["retrieved_docs"]
    # resume: a second run makes no reader/judge calls and gives the same numbers
    fake.calls.clear()
    sc2 = g.run(root=root, n=0, agent_provider="ollama", model="m", arms=("bm25",), items_log=log,
                context_cache=str(tmp_path / "ctx.jsonl"))
    assert fake.calls == [] and sc2["arms"]["bm25"]["correct"] == 24 and sc2["arms"]["bm25"]["resumed"] == 24
    assert sc2["manifest"]["context_cache"]["hits"] == 24
    # a changed judge refuses to resume the log
    sc3 = g.run(root=root, n=0, agent_provider="ollama", model="m", judge_model="other", arms=("bm25",),
                items_log=log)
    assert sc3["status"] == "skipped" and "different config" in sc3["reason"]


def test_run_qids_file_and_shard(tmp_path, monkeypatch):
    root = _make_root(tmp_path)
    monkeypatch.setattr(g, "_llm_chat", _FakeLLM())
    qf = tmp_path / "q.txt"
    qf.write_text("# dev\nFinance/temporal_1\nHealthcare/abstention_2\nFinance/multi_hop_1\n")
    sc = g.run(root=root, agent_provider="ollama", model="m", arms=("bm25",), qids_file=str(qf), shard="0/2")
    assert sc["config"]["n"] == 2 and sc["manifest"]["shard"] == "0/2"
    assert sc["manifest"]["benchmark"]["domains"] == ["Finance"]
    qf.write_text("Finance/nope_9\n")
    assert g.run(root=root, agent_provider="ollama", model="m", arms=("bm25",),
                 qids_file=str(qf))["status"] == "skipped"


def test_paid_provider_without_key_skips_before_any_work(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    sc = g.run(root=_make_root(tmp_path), agent_provider="openrouter")
    assert sc["status"] == "skipped" and "OPENROUTER_API_KEY" in sc["reason"]


# ── provider layer: system+user chat with the official budgets ───────────────────────────────

class _Resp:
    def __init__(self, body):
        self._b = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._b


def test_llm_chat_official_budget_for_gpt5_and_vendor_pin(monkeypatch):
    seen = {}

    def post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, json=json)
        return _Resp({"choices": [{"message": {"content": "Final: x"}}],
                      "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

    monkeypatch.setattr(P.requests, "post", post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.delenv("BENCH_OPENROUTER_PROVIDER", raising=False)
    assert P._llm_chat("openrouter", "openai/gpt-5", "SYS", "USER", max_tokens=512) == ("Final: x", 7, 3)
    body = seen["json"]
    assert body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert body["max_completion_tokens"] == 2048 and "temperature" not in body
    assert body["provider"] == {"order": ["OpenAI"], "allow_fallbacks": False}
    P._llm_chat("openrouter", "openai/gpt-5", "SYS", "USER", max_tokens=256)
    assert seen["json"]["max_completion_tokens"] == 2000
    P._llm_chat("openrouter", "openai/gpt-4o-mini", "SYS", "USER", max_tokens=256)
    assert seen["json"]["max_tokens"] == 256 and seen["json"]["temperature"] == 0.2


def test_llm_chat_error_body_raises_not_empty(monkeypatch):
    monkeypatch.setattr(P.requests, "post", lambda *a, **k: _Resp({"error": {"message": "nope"}}))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="error body"):
        P._llm_chat("openrouter", "openai/gpt-5", "S", "U", max_tokens=10)


def test_ollama_chat_disables_thinking_and_sets_budget(monkeypatch):
    seen = {}

    def post(url, json=None, timeout=None, **k):
        seen.update(url=url, json=json)
        return _Resp({"message": {"content": "Final: y"}, "prompt_eval_count": 4, "eval_count": 2})

    monkeypatch.setattr(P.requests, "post", post)
    assert P._llm_chat("ollama", "qwen3.5:9b-mlx-bf16", "S", "U", max_tokens=512) == ("Final: y", 4, 2)
    assert seen["url"].endswith("/api/chat") and seen["json"]["think"] is False
    assert seen["json"]["options"] == {"temperature": 0.2, "num_predict": 512}


def test_usd_prices_gpt5_from_openrouter_table():
    assert g._usd("openai/gpt-5", 1_000_000, 1_000_000) == pytest.approx(11.25)
    assert g._usd("qwen3.5:9b-mlx-bf16", 10**6, 10**6) == 0.0


@pytest.fixture
def llm(monkeypatch):
    f = _FakeLLM()
    monkeypatch.setattr(g, "_llm_chat", f)
    return f


def _run(fake, root, **kw):
    kw.setdefault("arms", ("kijito",))
    return g.run(root=root, agent_provider="ollama", model="m", _api_client=make_client(fake), **kw)


def _reader_users(llm):
    return [c[3] for c in llm.calls if c[2] == g.AGENT_SYSTEM]


def test_api_run_one_project_per_domain_and_the_reader_sees_the_exact_reply(tmp_path, llm):
    root = _make_root(tmp_path)
    fake = FakeKijito()
    sc = _run(fake, root, results_dir=str(tmp_path / "res"))
    assert sc["status"] == "ran", sc
    ka = sc["manifest"]["kijito_arm"]
    assert ka["mode"] == "api" and ka["server_commit"] == "c0ffee12" and ka["base_url"] == "https://kijito.test"
    assert ka["ctx_render"] == "mcp_text" and "MCP reply text" in ka["context_form"]
    assert ka["k"] == 10 and ka["blind"] == {**ka["blind"], "on": False, "scope": "account"}
    assert ka["project_per"] == "domain" and ka["scope_canary"] == "passed" and sc["api_leaks"] == {}
    for d in ("Finance", "Healthcare"):
        msgs = g.load_conversation_messages(g.conversation_path(root, d))
        stored = [m["content"] for m in fake.mem.values() if m["project"] == d]
        assert stored == [g.format_retrieved_message(m) for m in msgs]
    q_recalls = [a for n, a in fake.calls if n == "kijito_recall" and a.get("project") in ("Finance", "Healthcare")
                 and "probe" not in a["query"]]
    assert len(q_recalls) == 24
    assert all(a["limit"] == 10 and a["scope"] == "project" and a["query"].startswith("User_7 ") for a in q_recalls)
    users = _reader_users(llm)
    assert users and all("Retrieved passages:\nFound " in u and "⟦UNTRUSTED id=" in u for u in users)
    assert not any("\n[1] [user=" in u for u in users)
    # --results-dir keeps the official schema: retrieved_docs = the parsed bodies, plus the reply shown
    row = json.loads(open(tmp_path / "res" / "Finance" / "kijito__multi_hop.jsonl").readline())
    assert row["retrieved_docs"] and row["retrieved_docs"][0].startswith("[user=")
    assert row["mcp_reply"].startswith("Found ")


def test_api_ctx_contents_numbers_the_bodies_the_official_way(tmp_path, llm):
    root = _make_root(tmp_path, domains=("Finance",))
    sc = _run(FakeKijito(lean=True), root, api_ctx="contents")
    assert sc["status"] == "ran"
    u = _reader_users(llm)[0]
    assert "Retrieved passages:\n[1] [user=" in u and "⟦" not in u
    # the lean ' ⏎ ' glyph between the header and the body is shown as a line break again
    assert "msg_node=Msg_2]\nBudget moved to 40k & approved" in u


def test_remember_batch_calls_hold_at_most_60_items(tmp_path, llm):
    root = _make_root(tmp_path, domains=("Finance",))
    p = tmp_path / "gmb" / "data" / "final" / "Finance" / "synthetic_domain_channels_rolevariants_Finance.json"
    conv = {"Alpha": [{"msg_node": f"Msg_{i}", "content": f"budget note {i}", "author": "User_1", "role": "A",
                       "timestamp": f"2025-07-01T09:{i // 60:02d}:{i % 60:02d}"} for i in range(130)]}
    p.write_text(json.dumps(conv))
    fake = FakeKijito()
    assert _run(fake, root, n=1)["status"] == "ran"
    sizes = [len(a["memories"]) for n, a in fake.calls if n == "kijito_remember_batch" and a["project"] == "Finance"]
    assert sizes == [60, 60, 10]


def test_canary_refuses_when_project_scope_does_not_isolate(tmp_path, llm):
    sc = _run(FakeKijito(leak_projects=True), _make_root(tmp_path))
    assert sc["status"] == "refused" and "canary FAILED" in sc["reason"] and not llm.calls


def test_question_leak_marks_the_run_invalid(tmp_path, llm):
    sc = _run(FakeKijito(leak_question_only=True), _make_root(tmp_path))
    assert sc["status"] == "invalid" and "outside their project" in sc["invalid_reason"]
    assert sc["api_leaks"] and any("project=" in v for leak in sc["api_leaks"].values() for v in leak.values())


def test_failed_recall_is_a_retrieval_error_never_cached_and_the_ingest_is_reused(tmp_path, llm):
    root = _make_root(tmp_path, domains=("Finance",))
    cache = str(tmp_path / "ctx.jsonl")
    fake = FakeKijito()
    real_tool = fake._tool

    def broken(name, args):
        if name == "kijito_recall" and args["query"].startswith("User_7 temporal"):
            return fake._err("Error: something broke")
        return real_tool(name, args)
    fake._tool = broken
    sc = _run(fake, root, context_cache=cache)
    assert sc["status"] == "invalid" and sc["arms"]["kijito"]["retrieval_errors"] == 2
    fake._tool = real_tool
    writes = sum(1 for n, _ in fake.calls if n == "kijito_remember_batch")
    sc2 = _run(fake, root, context_cache=cache)   # the ingest record serves; only the 2 failed recalls re-run
    assert sc2["status"] == "ran" and sc2["manifest"]["kijito_arm"]["ingest_records"] == {"fresh": 0, "reused": 1}
    assert sum(1 for n, _ in fake.calls if n == "kijito_remember_batch") == writes + 2   # the canary pair only


def test_existing_project_is_refused(tmp_path, llm):
    fake = FakeKijito()
    fake.mem[999] = {"content": "benchmark namespace probe leftover", "project": "Finance", "scope": "project"}
    fake.next_id = 1000
    sc = _run(fake, _make_root(tmp_path, domains=("Finance",)))
    assert sc["status"] == "invalid" and sc["arms"]["kijito"]["retrieval_errors"] == 12


def test_cache_binds_the_server_and_the_resume_sentinel_binds_the_commit(tmp_path, llm):
    root = _make_root(tmp_path, domains=("Finance",))
    cache, log = str(tmp_path / "ctx.jsonl"), str(tmp_path / "items.jsonl")
    fake = FakeKijito()
    assert _run(fake, root, context_cache=cache, items_log=log)["status"] == "ran"
    n_calls = len(fake.calls)
    sc = _run(fake, root, context_cache=cache)          # every question hits: no tool call at all
    assert sc["status"] == "ran" and len(fake.calls) == n_calls
    assert "not run" in sc["manifest"]["kijito_arm"]["scope_canary"]
    fake2 = FakeKijito(commit="deadbeef")
    assert _run(fake2, root, context_cache=cache)["status"] == "ran"
    assert any(n == "kijito_remember_batch" and a["project"] == "Finance" for n, a in fake2.calls)
    sc = _run(FakeKijito(commit="deadbeef"), root, items_log=log)
    assert sc["status"] == "skipped" and "server-commit" in sc["reason"]


def test_only_the_kijito_cache_key_carries_the_server_tag(tmp_path, monkeypatch, llm):
    root = _make_root(tmp_path, domains=("Finance",))
    seen = []
    real = g.ContextCache.make_key
    monkeypatch.setattr(g.ContextCache, "make_key", staticmethod(lambda *a, **kw: (seen.append(kw.get("extra")),
                                                                                   real(*a, **kw))[1]))
    _run(FakeKijito(), root, retrieval_only=True, context_cache=str(tmp_path / "c.jsonl"))
    assert any(e and "|api=" in e for e in seen)
    seen.clear()
    g.run(root=root, agent_provider="ollama", model="m", arms=("bm25",), context_cache=str(tmp_path / "d.jsonl"),
          retrieval_only=True)
    assert seen and not any(e and "|api=" in e for e in seen)


def test_retrieval_only_spends_nothing_and_the_scored_run_reuses_it(tmp_path, llm):
    root = _make_root(tmp_path, domains=("Finance",))
    cache = str(tmp_path / "ctx.jsonl")
    fake = FakeKijito()
    sc = _run(fake, root, context_cache=cache, retrieval_only=True)
    assert sc["status"] == "retrieval-only" and sc["retrieved"] == {"kijito": 12} and not llm.calls
    n_calls = len(fake.calls)
    assert _run(fake, root, context_cache=cache)["status"] == "ran" and len(fake.calls) == n_calls


def test_blind_via_api_needs_the_account_enrolled(tmp_path, llm):
    root = _make_root(tmp_path, domains=("Finance",))
    sc = _run(FakeKijito(), root, blind=True)
    assert sc["status"] == "refused" and "fts-blind" in sc["reason"]
    sc = _run(FakeKijito(blind={"active": True, "index_exists": True}), root, blind=True)
    assert sc["status"] == "ran" and sc["manifest"]["kijito_arm"]["blind"]["on"] is True
    assert sc["manifest"]["kijito_arm"]["blind"]["status"]["active"] is True


def test_bad_ctx_missing_tool_and_missing_api_are_refused(tmp_path, llm):
    root = _make_root(tmp_path, domains=("Finance",))
    sc = g.run(root=root, agent_provider="ollama", model="m", arms=("kijito",))
    assert sc["status"] == "skipped" and "--via-api" in sc["reason"]
    assert _run(FakeKijito(), root, api_ctx="html")["status"] == "skipped"
    sc = _run(FakeKijito(tools=["kijito_recall", "kijito_get"]), root)
    assert sc["status"] == "skipped" and "kijito_remember_batch" in sc["reason"]


def test_full_account_stops_ingest_and_marks_every_remaining_question(tmp_path, llm):
    root = _make_root(tmp_path)                       # Finance + Healthcare, 3 messages each
    fake = FakeKijito(memory_limit=4)                 # canary (2) fits; a 3-message domain does not
    sc = _run(fake, root)
    assert sc["status"] == "invalid" and sc["arms"]["kijito"]["retrieval_errors"] == 24
    projects = {a["project"] for n, a in fake.calls if n == "kijito_remember_batch"}
    assert len([p for p in projects if "canary" not in p]) == 1      # the second domain was never written
    assert not llm.calls                              # nothing to answer
