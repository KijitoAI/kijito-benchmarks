"""LongMemEval harness (kijito_bench/longmemeval.py) against the in-memory fake MCP endpoint and a fake
reader/judge. No network, no LLM, no dataset download."""
import json

import pytest

from kijito_bench import client as kapi
from kijito_bench import longmemeval as lme
from kijito_bench.runner import ContextCache
from conftest import FakeKijito, make_client


def _qs():
    def q(qid, question, sessions, qtype="single-session-user"):
        return {"question_id": qid, "question_type": qtype, "question": question,
                "answer": "x", "question_date": "2024/01/01",
                "haystack_sessions": sessions, "haystack_dates": ["2023/05/01", "2023/06/01"]}
    return [
        q("q1", "what cat did i adopt", [[{"role": "user", "content": "i adopted a cat named biscuit"}],
                                        [{"role": "assistant", "content": "nice cat"}]]),
        q("q2", "where does my dog swim", [[{"role": "user", "content": "my dog rex likes to swim"}],
                                          [{"role": "user", "content": "the beach is near"}]]),
    ]


@pytest.fixture
def harness(monkeypatch):
    prompts = []
    monkeypatch.setattr(lme, "_load_split", lambda split, n, seed: (_qs(), "s.json"))

    def fake_llm(p, m, prompt, retries=2):
        prompts.append(prompt)
        return ("yes" if "Answer yes or no only" in prompt else "biscuit"), 5, 1
    monkeypatch.setattr(lme, "_llm", fake_llm)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(lme, "_md_contexts",
                        lambda questions, k, cache=None, digest=None, errors=None:
                        {q["question_id"]: ["md"] for q in questions})
    return prompts


def _run(fake, **kw):
    return lme.run(2, 10, "gpt-4o", 1, agent_provider="openai", official_eval=True,
                   _api_client=make_client(fake), **kw)


def _reader(prompts, question):
    return [p for p in prompts if question in p and "NOTES:" in p and "yes or no" not in p]


# ── prompts / judge ──────────────────────────────────────────────────────────────────────────────

def test_turn_notes_are_dated_and_skip_empty_turns():
    q = _qs()[0]
    q["haystack_sessions"][1].append({"role": "user", "content": "   "})
    assert lme._turn_notes(q) == ["[2023/05/01] user: i adopted a cat named biscuit",
                                  "[2023/06/01] assistant: nice cat"]


def test_official_judge_routes_by_type_and_abstention_id():
    assert lme._official_judge_prompt("temporal-reasoning", "q1", "Q", "G", "H").startswith(
        lme._OFFICIAL_JUDGE_TEMPORAL.split("{}")[0])
    assert "Rubric: G" in lme._official_judge_prompt("single-session-preference", "q1", "Q", "G", "H")
    p = lme._official_judge_prompt("temporal-reasoning", "q1_abs", "Q", "G", "H")
    assert p.startswith("I will give you an unanswerable question") and "Explanation: G" in p
    assert lme._official_judge_prompt("unknown-type", "q1", "Q", "G", "H") == \
        lme._OFFICIAL_JUDGE_DEFAULT.format("Q", "G", "H")


def test_official_judge_parse_is_yes_in_lowercase(monkeypatch):
    monkeypatch.setattr(lme, "_llm", lambda p, m, prompt: ("Yes.", 3, 1))
    assert lme._official_judge_counted("openai", "m", "multi-session", "q1", "Q", "G", "H") == (True, 3, 1, 1)
    monkeypatch.setattr(lme, "_llm", lambda p, m, prompt: ("No", 3, 1))
    assert lme._official_judge_counted("openai", "m", "multi-session", "q1", "Q", "G", "H")[0] is False


def test_stratified_sample_is_deterministic_and_covers_types():
    data = [{"question_id": f"{t}{i}", "question_type": t} for t in "abc" for i in range(4)]
    a, b = lme._stratified_sample(data, 3, 7), lme._stratified_sample(data, 3, 7)
    assert [x["question_id"] for x in a] == [x["question_id"] for x in b]
    assert {x["question_type"] for x in a} == {"a", "b", "c"}
    assert len(lme._stratified_sample(data, 0, 7)) == 12


# ── run() through the fake API ──────────────────────────────────────────────────────────────────

def test_api_run_reader_gets_the_exact_mcp_reply(harness):
    fake = FakeKijito()
    sc = _run(fake)
    assert sc["status"] == "ran", sc
    ka = sc["manifest"]["kijito_arm"]
    assert ka["mode"] == "api" and ka["server_commit"] == "c0ffee12" and ka["base_host"] == "kijito.test"
    assert ka["scope_canary"] == "passed"
    assert sc["api_leaks"] == {}
    recalls = [a for n, a in fake.calls if n == "kijito_recall" and a["query"] == "what cat did i adopt"]
    assert recalls == [{"query": "what cat did i adopt", "limit": 10, "scope": "project", "project": "q1"}]
    q1_notes = [m["content"] for m in fake.mem.values() if m["project"] == "q1"]
    assert q1_notes == lme._turn_notes(_qs()[0])
    reader = _reader(harness, "what cat did i adopt")
    assert reader and "⟦UNTRUSTED id=" in reader[0] and "score=0.900 scope=project" in reader[0]
    assert "- [2023/05/01] user:" not in reader[0]
    assert sc["arms"]["kijito"]["ctx_tokens_mean"] > 0
    t = sc["retrieval_timing"]["kijito"]["q1"]
    assert t["n_notes"] == 2 and {"ingest_s", "recall_s", "probe_s", "total_s"} <= set(t)


def test_reader_prompt_is_the_official_answer_prompt(harness):
    _run(FakeKijito(lean=True), no_append_md=True)
    reader = _reader(harness, "what cat did i adopt")[0]
    assert reader.startswith("You are a personal assistant answering a question about a user.")
    assert "Today's date: 2024/01/01\nNOTES:\nFound " in reader
    assert reader.endswith("QUESTION: what cat did i adopt\nANSWER:")


def test_api_ctx_contents_renders_the_bodies(harness):
    sc = _run(FakeKijito(), api_ctx="contents")
    assert sc["status"] == "ran"
    reader = _reader(harness, "what cat did i adopt")
    assert "- [2023/05/01] user: i adopted a cat named biscuit" in reader[0]
    assert "⟦UNTRUSTED" not in reader[0]


def test_without_an_api_the_run_is_skipped(harness):
    sc = lme.run(2, 10, "gpt-4o", 1, agent_provider="openai")
    assert sc["status"] == "skipped" and "--via-api" in sc["reason"]


def test_canary_refuses_when_project_scope_does_not_isolate(harness):
    sc = _run(FakeKijito(leak_projects=True))
    assert sc["status"] == "refused" and "canary FAILED" in sc["reason"]


def test_question_leak_marks_the_run_invalid(harness):
    sc = _run(FakeKijito(leak_question_only=True))
    assert sc["status"] == "invalid"
    assert set(sc["api_leaks"]) == {"q1", "q2"}
    assert any("project=" in v for v in sc["api_leaks"]["q1"].values())


def test_failed_call_is_a_retrieval_error_and_never_cached(harness, tmp_path):
    fake = FakeKijito()
    cache = str(tmp_path / "ctx.jsonl")
    c = make_client(fake)
    orig = c.recall

    def flaky(query, **kw):
        if query == "where does my dog swim":
            raise kapi.KijitoAPIError("boom")
        return orig(query, **kw)
    c.recall = flaky
    sc = lme.run(2, 10, "gpt-4o", 1, agent_provider="openai", _api_client=c, context_cache=cache)
    assert sc["status"] == "invalid" and sc["arms"]["kijito"]["retrieval_errors"] == 1
    recs = [json.loads(line) for line in open(cache)]
    replies = [r for r in recs if len(r["contexts"]) == 2]
    assert len(replies) == 1 and "biscuit" in replies[0]["contexts"][0]
    ingests = [json.loads(r["contexts"][0]) for r in recs if len(r["contexts"]) == 1]
    assert sorted(m["project"] for m in ingests) == ["q1", "q2"]


def test_full_account_stops_ingest_and_marks_the_rest_invalid(harness, monkeypatch):
    # the canary pair (2) + q1's two notes fit; q2's batch is refused, and nothing is written after it
    fake = FakeKijito(memory_limit=4)
    qs = _qs() + [dict(_qs()[1], question_id="q3")]
    monkeypatch.setattr(lme, "_load_split", lambda split, n, seed: (qs, "s.json"))
    sc = _run(fake, no_append_md=True)
    assert sc["status"] == "invalid" and sc["arms"]["kijito"]["retrieval_errors"] == 2
    writes = [a["project"] for n, a in fake.calls if n == "kijito_remember_batch"]
    assert writes[-1] == "q2" and "q3" not in writes
    reader_q = [p for p in harness if "NOTES:" in p and "yes or no" not in p]
    assert all("dog" not in p for p in reader_q)          # no reader call for the errored questions


def test_canary_refused_by_a_full_account_is_a_refusal(harness):
    sc = _run(FakeKijito(memory_limit=1))
    assert sc["status"] == "refused" and "MemoryCapReached" not in sc["reason"] and "cap" in sc["reason"]


def test_existing_project_is_refused(harness):
    fake = FakeKijito()
    make_client(fake).remember_batch(["benchmark namespace probe leftover"], "q1")
    sc = _run(fake)
    assert sc["status"] == "invalid"
    assert sc["arms"]["kijito"]["retrieval_errors"] == 1


def test_cache_key_binds_server_commit_and_cached_leaks_are_reasserted(harness, tmp_path):
    cache = str(tmp_path / "ctx.jsonl")
    fake = FakeKijito()
    assert _run(fake, context_cache=cache)["status"] == "ran"
    n_calls = len(fake.calls)
    sc2 = _run(fake, context_cache=cache)
    assert sc2["manifest"]["context_cache"]["hits"] == 2
    assert len(fake.calls) == n_calls
    assert sc2["manifest"]["kijito_arm"]["scope_canary"].startswith("not run")
    sc3 = _run(FakeKijito(commit="deadbeef"), context_cache=cache)
    assert sc3["manifest"]["context_cache"]["hits"] == 0
    cache2 = str(tmp_path / "ctx2.jsonl")
    leaky = FakeKijito(leak_question_only=True)
    _run(leaky, context_cache=cache2)
    sc4 = _run(leaky, context_cache=cache2)
    assert sc4["status"] == "invalid" and set(sc4["api_leaks"]) == {"q1", "q2"}


def test_cache_key_binds_the_server_tag_and_blind():
    c = object.__new__(ContextCache)
    base = lme._cache_key(c, "q1", "kijito", 10, "d", api_tag="h|commit=a|prefix=|recall_explain=off")
    assert base != lme._cache_key(c, "q1", "kijito", 10, "d", api_tag="h|commit=b|prefix=|recall_explain=off")
    assert base != lme._cache_key(c, "q1", "kijito", 10, "d", api_tag="h|commit=a|prefix=|recall_explain=on")
    assert base != lme._cache_key(c, "q1", "kijito", 10, "d", api_tag="h|commit=a|prefix=|recall_explain=off",
                                  blind=True)
    assert lme._cache_key(None, "q1", "kijito", 10, "d") is None


def test_resume_reuses_clean_records_and_refuses_a_different_server_commit(harness, tmp_path):
    log = str(tmp_path / "items.jsonl")
    assert _run(FakeKijito(), items_log=log)["status"] == "ran"
    n = len(harness)
    sc = _run(FakeKijito(), items_log=log)
    assert sc["status"] == "ran" and len(harness) == n          # resumed: no reader/judge call
    sc = _run(FakeKijito(commit="deadbeef"), items_log=log)
    assert sc["status"] == "skipped" and "DIFFERENT run config" in sc["reason"] and "/api(" in sc["reason"]


def test_items_log_records_the_per_question_fields(harness, tmp_path):
    log = str(tmp_path / "items.jsonl")
    _run(FakeKijito(), items_log=log, no_append_md=True)
    recs = [json.loads(x) for x in open(log)][1:]
    assert {r["qid"] for r in recs} == {"q1", "q2"}
    assert set(recs[0]) == {"qid", "arm", "category", "ok", "hyp_len", "in_tok", "out_tok", "error",
                            "answer_in_tok", "answer_out_tok", "judge_in_tok", "judge_out_tok", "hyp",
                            "elapsed_s"}
    assert recs[0]["ok"] is True and recs[0]["hyp"] == "biscuit"


def test_blind_requires_enrolment(harness):
    sc = _run(FakeKijito(), blind=True)
    assert sc["status"] == "refused" and "fts-blind" in sc["reason"]
    sc = _run(FakeKijito(blind={"active": True, "index_exists": True}), blind=True)
    assert sc["status"] == "ran" and sc["manifest"]["blind_index"] is True


def test_missing_tool_is_refused(harness):
    sc = _run(FakeKijito(tools=["kijito_recall"]))
    assert sc["status"] == "skipped" and "kijito_remember_batch" in sc["reason"]


def test_declared_recall_limit_is_recorded(harness):
    sc = _run(FakeKijito(recall_limit_max=100), no_append_md=True)
    assert sc["manifest"]["kijito_arm"]["recall_limit_max"] == 100


def test_no_append_md_runs_the_kijito_arm_alone(harness):
    sc = _run(FakeKijito(), no_append_md=True)
    assert sc["status"] == "ran" and set(sc["arms"]) == {"kijito"} and sc["stats"] is None
    assert sc["manifest"]["append_md_arm"].startswith("skipped")
    lme._print(sc)


def test_one_ingest_serves_a_lean_and_an_explain_read(harness, tmp_path):
    fake = FakeKijito(lean=True)
    cache = str(tmp_path / "ctx.jsonl")
    sa = _run(fake, context_cache=cache, items_log=str(tmp_path / "a.jsonl"), no_append_md=True)
    assert sa["status"] == "ran", sa
    assert sa["manifest"]["kijito_arm"]["ingest"] == {"fresh": 2, "reused": 0}
    lean_reader = _reader(harness, "what cat did i adopt")[-1]
    assert "- [3] 2023-05-01 ⟦n=abc123 [2023/05/01] user: i adopted a cat named biscuit n=abc123⟧" in lean_reader
    writes = sum(1 for n, _ in fake.calls if n == "kijito_remember_batch")
    sb = _run(fake, context_cache=cache, api_explain=True, items_log=str(tmp_path / "b.jsonl"),
              no_append_md=True)
    assert sb["status"] == "ran", sb
    ka = sb["manifest"]["kijito_arm"]
    assert ka["recall_explain"] is True and ka["ingest"] == {"fresh": 0, "reused": 2}
    new_writes = [a for n, a in fake.calls if n == "kijito_remember_batch"][writes:]
    assert all("canary" in m["content"] for a in new_writes for m in a["memories"])
    qrec = [a for n, a in fake.calls if n == "kijito_recall" and a["query"] == "what cat did i adopt"]
    assert [a.get("explain") for a in qrec] == [None, True]


def test_ingest_record_mismatch_is_a_retrieval_error(harness, tmp_path):
    fake = FakeKijito(lean=True)
    cache_path = str(tmp_path / "ctx.jsonl")
    assert _run(fake, context_cache=cache_path)["status"] == "ran"
    recs = [json.loads(line) for line in open(cache_path)]
    for r in recs:
        if len(r["contexts"]) == 1 and '"n_notes"' in r["contexts"][0]:
            m = json.loads(r["contexts"][0])
            m["n_notes"] += 1
            r["contexts"] = [json.dumps(m)]
    open(cache_path, "w").write("".join(json.dumps(r) + "\n" for r in recs))
    sc = _run(fake, context_cache=cache_path, api_explain=True)
    assert sc["status"] == "invalid" and sc["arms"]["kijito"]["retrieval_errors"] == 2


def test_cli_requires_a_key_file():
    with pytest.raises(SystemExit):
        lme.main(["--n", "1"])
