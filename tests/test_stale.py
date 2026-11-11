"""STALE harness (kijito_bench/stale.py): official prompts/parse/metrics, and the Kijito arm against the
in-memory fake MCP endpoint with a fake reader/judge. No network, no LLM, no dataset download."""
import hashlib
import json

import pytest

from kijito_bench import stale as S
from kijito_bench import llm as P
from kijito_bench.runner import ContextCache
from conftest import FakeKijito, make_client


# ── fixtures ─────────────────────────────────────────────────────────────────────────────────────

def _scenario(uid, typ, old_city, new_city, n_noise=2):
    sessions, stamps = [], []
    for i in range(n_noise):
        sessions.append([{"role": "user", "content": f"Can you suggest a pasta recipe number {i}?"},
                         {"role": "assistant", "content": f"Try a simple tomato pasta, variant {i}."}])
        stamps.append(f"2024-0{i + 1}-01 10:00" if i < 9 else f"2024-{i + 1}-01 10:00")
    sessions.append([{"role": "user", "content": f"I have been based in {old_city} for years."},
                     {"role": "assistant", "content": f"{old_city} is a lovely place to live."}])
    stamps.append("2024-05-01 09:00")
    sessions.append([{"role": "user", "content": f"I just settled into my new apartment in {new_city}."},
                     {"role": "assistant", "content": ""}])          # an empty turn is skipped
    stamps.append("2024-08-01 09:00")
    return {
        "uid": uid, "type": typ, "M_old": f"I have been based in {old_city} for years.",
        "M_new": f"I just settled into my new apartment in {new_city}.",
        "explanation": f"location is now {new_city}",
        "probing_queries": {"dim1_query": f"Does the user still live in {old_city}?",
                            "dim2_query": f"Since the user lives in {old_city}, what should they visit?",
                            "dim3_query": "Recommend weekend spots near my apartment."},
        "relevant_session_index": [n_noise, n_noise + 1],
        "timestamps": stamps, "haystack_session": sessions,
    }


def _dataset():
    return [_scenario("u1", "T1", "Seattle", "Austin"), _scenario("u2", "T2", "Portland", "Phoenix"),
            _scenario("u3", "T1", "Boston", "Denver"), _scenario("u4", "T2", "Miami", "Chicago")]


@pytest.fixture
def data_file(tmp_path):
    p = tmp_path / "stale_tiny.json"
    p.write_text(json.dumps(_dataset()))
    return str(p)


def _judge_json(p1=True, p2=False, p3=True, fence=False):
    body = json.dumps({f"dim{i}_eval": {"reasoning": f"r{i}", "pass": p} for i, p in ((1, p1), (2, p2), (3, p3))})
    return f"```json\n{body}\n```" if fence else body


class _FakeLLM:
    """Records every call; reader calls return a short answer, judge calls return the judge JSON."""
    def __init__(self, judge_text=None):
        self.calls = []
        self.judge_text = judge_text or _judge_json()

    def __call__(self, provider, model, prompt, retries=2, system=None, json_mode=False):
        self.calls.append({"provider": provider, "model": model, "prompt": prompt, "system": system,
                           "json_mode": json_mode})
        if system == S.JUDGE_SYSTEM:
            return self.judge_text, 1000, 50
        return "The user moved; they now live elsewhere.", 400, 20


@pytest.fixture
def llm(monkeypatch):
    f = _FakeLLM()
    monkeypatch.setattr(S, "_llm", f)
    monkeypatch.setattr(S, "_ollama_available", lambda model: True)
    return f


def _run(fake, data_file, **kw):
    kw.setdefault("n", 2)
    return S.run(k=3, model="m", data_path=data_file, _api_client=make_client(fake), **kw)


def _reader_prompts(llm):
    return [c for c in llm.calls if c["system"] != S.JUDGE_SYSTEM]


# ── official prompts + parse ─────────────────────────────────────────────────────────────────────

def test_judge_system_prompt_is_the_pinned_official_text():
    assert hashlib.sha256(S.JUDGE_SYSTEM.encode()).hexdigest() == S.JUDGE_SYSTEM_SHA256
    assert S.JUDGE_SYSTEM.startswith("\nYou are a strict and impartial expert evaluator")
    assert "by M_new.  \n" in S.JUDGE_SYSTEM            # the upstream trailing spaces survive


def test_reader_prompts_are_official_per_dim():
    assert S._READER_SYSTEM["dim1"] == S._READER_SYSTEM["dim2"] == (
        "You are a helpful assistant. Review the following conversation history with the user, "
        "then accurately answer the question.")
    assert S._READER_SYSTEM["dim3"].endswith("then respond to the user's latest query directly.")
    assert S._reader_user_prompt("H", "Q", "dim1") == "[Conversation History]\nH\n\n[Question]\nQ"
    assert S._reader_user_prompt("H", "Q", "dim3") == "[Conversation History]\nH\n\n[Latest Query]\nQ"


def test_judge_user_prompt_carries_ground_truth_and_all_three_responses():
    it = _scenario("u1", "T1", "Seattle", "Austin")
    p = S.judge_user_prompt(it, {"dim1": "A1", "dim2": "A2", "dim3": "A3"})
    assert p.startswith("\n[Ground Truth Context]\n- M_old: \"I have been based in Seattle")
    assert "- Hidden Logic: location is now Austin\n" in p
    for i in (1, 2, 3):
        assert f"Target Model Response {i}: A{i}\n" in p


def test_parse_judge_official_fence_and_plain():
    for fence in (True, False):
        res, fb = S.parse_judge(_judge_json(True, False, True, fence=fence))
        assert fb is False
        assert [S._judge_pass(res, d) for d in S._DIMS] == [True, False, True]


def test_parse_judge_fallback_is_counted_and_garbage_raises():
    res, fb = S.parse_judge("Here you go: " + _judge_json() + " thanks")
    assert fb is True and S._judge_pass(res, "dim1")
    with pytest.raises(ValueError):
        S.parse_judge("no json at all")
    with pytest.raises(ValueError):
        S.parse_judge(json.dumps({"dim1_eval": {"pass": True}}))


def test_string_false_is_not_a_pass():
    assert S._judge_pass({"dim1_eval": {"pass": "false"}}, "dim1") is False
    assert S._judge_pass({"dim1_eval": {"pass": "True"}}, "dim1") is True


# ── dataset shaping ──────────────────────────────────────────────────────────────────────────────

def test_stratified_sample_is_deterministic_and_covers_both_types():
    d = _dataset()
    a, b = S.stratified_sample(d, 2, 7), S.stratified_sample(d, 2, 7)
    assert [x["uid"] for x in a] == [x["uid"] for x in b]
    assert {x["type"] for x in a} == {"T1", "T2"}
    assert len(S.stratified_sample(d, 0, 7)) == 4


def test_session_notes_are_chronological_dated_and_skip_empty_turns():
    it = _scenario("u1", "T1", "Seattle", "Austin")
    it["haystack_session"].reverse(); it["timestamps"].reverse()
    sess = S.session_notes(it)
    assert sess[0][0].startswith("[2024-01-01 10:00] user: ")
    assert sess[-1] == ["[2024-08-01 09:00] user: I just settled into my new apartment in Austin."]


def test_load_dataset_validates_and_hashes(data_file, tmp_path):
    recs, path, sha = S.load_dataset(data_file)
    assert len(recs) == 4 and sha == hashlib.sha256(open(data_file, "rb").read()).hexdigest()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"uid": "x", "M_old": "a"}]))
    with pytest.raises(ValueError):
        S.load_dataset(str(bad))


# ── metrics + scoring ────────────────────────────────────────────────────────────────────────────

def test_overall_is_mean_of_six_settings_not_pooled():
    recs = [{"uid": "a", "type": "T1", "pass": {"dim1": True, "dim2": True, "dim3": True}},
            {"uid": "b", "type": "T1", "pass": {"dim1": True, "dim2": True, "dim3": True}},
            {"uid": "c", "type": "T1", "pass": {"dim1": True, "dim2": True, "dim3": True}},
            {"uid": "d", "type": "T2", "pass": {"dim1": False, "dim2": False, "dim3": False}},
            {"uid": "e", "type": "T2", "pass": None, "error": "boom", "error_stage": "judge"}]
    m = S.compute_metrics(recs)
    assert m["overall_mean_of_six"] == 50.0
    assert m["pooled_accuracy"] == 75.0
    assert m["settings"]["T1/SR"] == {"correct": 3, "total": 3, "accuracy": 100.0}
    assert m["by_type"]["T2"]["accuracy"] == 0.0
    assert m["errors"] == 1 and m["errors_by_stage"] == {"judge": 1} and m["n_scored"] == 4


def test_three_separate_reader_calls_and_one_judge_call(llm):
    it = _scenario("u1", "T1", "Seattle", "Austin")
    ctxs = {("u1", d): ["[2024-08-01 09:00] user: I just settled into my new apartment in Austin."]
            for d in S._DIMS}
    [r] = S.score_items([it], ctxs, {}, ("ollama", "reader-m"), ("ollama", "judge-m"))
    readers = [c for c in llm.calls if c["system"] != S.JUDGE_SYSTEM]
    judges = [c for c in llm.calls if c["system"] == S.JUDGE_SYSTEM]
    assert len(readers) == 3 and len(judges) == 1
    assert [c["system"] for c in readers] == [S._READER_SYSTEM[d] for d in S._DIMS]
    assert judges[0]["model"] == "judge-m" and judges[0]["json_mode"] is True
    assert r["pass"] == {"dim1": True, "dim2": False, "dim3": True}
    assert (r["ans_in"], r["ans_out"], r["judge_in"], r["judge_out"]) == (1200, 60, 1000, 50)


def test_retrieval_error_makes_no_llm_call(llm):
    it = _scenario("u1", "T1", "Seattle", "Austin")
    [r] = S.score_items([it], {}, {("u1", "dim2"): "recall: RuntimeError: x"}, ("ollama", "m"), ("ollama", "m"))
    assert llm.calls == [] and r["error_stage"] == "retrieval" and r["pass"] is None


def test_judge_failure_is_an_error_not_a_fail(llm):
    llm.judge_text = "I refuse to output JSON"
    it = _scenario("u1", "T1", "Seattle", "Austin")
    [r] = S.score_items([it], {("u1", d): ["x"] for d in S._DIMS}, {}, ("ollama", "m"), ("ollama", "m"))
    assert r["error_stage"] == "judge" and r["pass"] is None


def test_official_configuration_is_recognised_and_priced():
    c = S._comparability("openai/gpt-4o-mini", "google/gemini-3.1-flash-lite-preview")
    assert c["comparable_to_published_memory_rows"] is True
    assert not S._comparability("openai/gpt-4o", "google/gemini-3.1-flash-lite-preview")["reader_official"]
    assert S._cost("google/gemini-3.1-flash-lite-preview", 1_000_000, 1_000_000) == pytest.approx(1.75)
    assert S._cost("openai/gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)


def test_openrouter_body_carries_system_and_json_mode(monkeypatch):
    seen = {}

    class R:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "{}"}}], "usage": {}}

    def post(url, headers=None, json=None, timeout=None):
        seen.update(json)
        return R()
    monkeypatch.setattr(P.requests, "post", post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    P._openrouter_call("google/gemini-3.1-flash-lite-preview", "u", system="S", json_mode=True)
    assert seen["messages"] == [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]
    assert seen["response_format"] == {"type": "json_object"} and seen["temperature"] == 0
    assert "provider" not in seen
    P._openrouter_call("openai/gpt-4o-mini", "u", system="S")
    assert seen["provider"] == {"order": ["OpenAI"], "allow_fallbacks": False}


# ── the Kijito arm through the fake API ─────────────────────────────────────────────────────────

def test_api_run_one_project_per_scenario_and_the_reader_sees_the_exact_reply(data_file, llm):
    fake = FakeKijito()
    sc = _run(fake, data_file)
    assert sc["status"] == "ran", sc
    ka = sc["manifest"]["kijito_arm"]
    assert ka["mode"] == "api" and ka["server_commit"] == "c0ffee12"
    assert ka["base_host"] == "kijito.test" and ka["ctx_render"] == "mcp_text"
    assert ka["k"] == 3 and ka["blind"]["on"] is False and ka["project_per"] == "scenario"
    assert ka["scope_canary"] == "passed" and sc["api_leaks"] == {} and sc["config"]["via_api"] is True
    items = {it["uid"]: it for it in _dataset()}
    for uid in [r["uid"] for r in sc["items"]]:
        stored = [m["content"] for m in fake.mem.values() if m["project"] == uid]
        assert stored == S.scenario_notes_flat(items[uid])
        recalls = [a for n, a in fake.calls if n == "kijito_recall" and a.get("project") == uid
                   and "probe" not in a["query"]]
        assert recalls == [{"query": items[uid]["probing_queries"][f"{d}_query"], "limit": 3,
                            "scope": "project", "project": uid} for d in S._DIMS]
    hist = [c["prompt"] for c in _reader_prompts(llm) if "Seattle" in c["prompt"] or "Portland" in c["prompt"]]
    assert hist and all("⟦UNTRUSTED id=" in h and "score=0.900 scope=project" in h for h in hist)


def test_api_ctx_contents_renders_bodies_as_note_lines(data_file, llm):
    sc = _run(FakeKijito(lean=True), data_file, api_ctx="contents")
    assert sc["status"] == "ran"
    hist = [c["prompt"] for c in _reader_prompts(llm)
            if "[Question]\nDoes the user still live in Seattle?" in c["prompt"]]
    assert hist and "- [2024-05-01 09:00] user: I have been based in Seattle for years." in hist[0]
    assert "⟦" not in hist[0]


def test_chrono_order_needs_contents(data_file, llm):
    sc = _run(FakeKijito(), data_file, ctx_order="chrono")
    assert sc["status"] == "skipped" and "--api-ctx contents" in sc["reason"]
    assert _run(FakeKijito(), data_file, ctx_order="chrono", api_ctx="contents")["status"] == "ran"


def test_without_an_api_the_run_is_skipped(data_file, llm):
    sc = S.run(n=2, k=3, model="m", data_path=data_file)
    assert sc["status"] == "skipped" and "--via-api" in sc["reason"]


def test_remember_batch_calls_hold_at_most_60_items_in_order(tmp_path, llm):
    big = _scenario("big", "T1", "Seattle", "Austin", n_noise=40)   # 40 x 2 + 3 = 83 notes
    p = tmp_path / "big.json"
    p.write_text(json.dumps([big]))
    fake = FakeKijito()
    sc = S.run(n=1, k=3, model="m", data_path=str(p), _api_client=make_client(fake))
    assert sc["status"] == "ran"
    batches = [a for n, a in fake.calls if n == "kijito_remember_batch" and a["project"] == "big"]
    assert [len(b["memories"]) for b in batches] == [60, 23]
    assert [m["content"] for b in batches for m in b["memories"]] == S.scenario_notes_flat(big)


def test_canary_refuses_when_project_scope_does_not_isolate(data_file, llm):
    sc = _run(FakeKijito(leak_projects=True), data_file)
    assert sc["status"] == "refused" and "canary FAILED" in sc["reason"]
    assert not _reader_prompts(llm)


def test_probe_leak_marks_the_run_invalid(data_file, llm):
    sc = _run(FakeKijito(leak_question_only=True), data_file)
    assert sc["status"] == "invalid" and sc["api_leaks"]


def test_failed_recall_is_a_retrieval_error_and_never_cached(data_file, llm, tmp_path):
    cache = str(tmp_path / "ctx.jsonl")
    fake = FakeKijito()
    real_tool = fake._tool

    def broken(name, args):
        if name == "kijito_recall" and "Seattle" in args["query"]:
            return fake._err("Error: something broke")
        return real_tool(name, args)
    fake._tool = broken
    sc = _run(fake, data_file, context_cache=cache)
    assert sc["status"] == "invalid" and sc["metrics"]["errors_by_stage"].get("retrieval") == 1
    vals = [json.loads(line)["contexts"] for line in open(cache)]
    replies = [v for v in vals if len(v) == 2]
    assert sorted(json.loads(v[1])["dim"] for v in replies if json.loads(v[1])["uid"] == "u1") == ["dim3"]
    sc2 = _run(FakeKijito(), data_file, context_cache=cache)
    assert sc2["status"] == "ran"


def test_full_account_stops_ingest(data_file, llm):
    # canary (2) + the first scenario (7 notes) fit; the second scenario is refused, later ones not tried
    fake = FakeKijito(memory_limit=10)
    sc = _run(fake, data_file, n=4)
    errs = [r for r in sc["items"] if r["error"]]
    assert sc["status"] == "invalid" and len(errs) == 3
    assert all("Memory limit reached" in r["error"] for r in errs)
    projects = {a["project"] for n, a in fake.calls if n == "kijito_remember_batch"}
    assert len([p for p in projects if "canary" not in p]) == 2      # one stored + the refused one


def test_existing_project_is_refused(data_file, llm):
    fake = FakeKijito()
    fake.mem[999] = {"content": "benchmark namespace probe leftover", "project": "u1", "scope": "project"}
    fake.next_id = 1000
    sc = _run(fake, data_file, n=4)
    errs = [r for r in sc["items"] if r["error"]]
    assert sc["status"] == "invalid" and [r["uid"] for r in errs] == ["u1"]
    assert "already holds memories" in errs[0]["error"]


def test_cache_binds_the_server_and_cached_leaks_are_reasserted(data_file, llm, tmp_path):
    cache = str(tmp_path / "ctx.jsonl")
    fake = FakeKijito(leak_question_only=True)
    sc1 = _run(fake, data_file, context_cache=cache)
    assert sc1["status"] == "invalid"
    n_calls = len(fake.calls)
    sc2 = _run(fake, data_file, context_cache=cache)
    assert len(fake.calls) == n_calls
    assert sc2["status"] == "invalid" and sc2["api_leaks"] == sc1["api_leaks"]
    fake3 = FakeKijito(commit="deadbeef")
    sc3 = _run(fake3, data_file, context_cache=cache)
    assert sc3["status"] == "ran" and sc3["manifest"]["kijito_arm"]["ingest"] == {"fresh": 2, "reused": 0}


def test_cache_key_binds_dataset_k_server_and_blind(tmp_path):
    c = ContextCache(str(tmp_path / "c.jsonl"))
    base = S._cache_key(c, "u1", "dim1", "kijito", 10, "api", "d1", api_tag="h|commit=x")
    assert base != S._cache_key(c, "u1", "dim1", "kijito", 10, "api", "d2", api_tag="h|commit=x")
    assert base != S._cache_key(c, "u1", "dim1", "kijito", 5, "api", "d1", api_tag="h|commit=x")
    assert base != S._cache_key(c, "u1", "dim1", "kijito", 10, "api", "d1", api_tag="h|commit=y")
    S._BLIND["on"] = True
    try:
        assert base != S._cache_key(c, "u1", "dim1", "kijito", 10, "api", "d1", api_tag="h|commit=x")
    finally:
        S._BLIND["on"] = False


def test_resume_and_refusal(data_file, llm, tmp_path):
    log = str(tmp_path / "items.jsonl")
    sc = _run(FakeKijito(), data_file, n=4, items_log=log)
    assert sc["status"] == "ran" and sc["metrics"]["overall_mean_of_six"] == pytest.approx(66.7)
    n_calls = len(llm.calls)
    sc2 = _run(FakeKijito(), data_file, n=4, items_log=log)
    assert sc2["status"] == "ran" and len(llm.calls) == n_calls and sc2["cost"]["resumed_items"] == 4
    sc3 = _run(FakeKijito(), data_file, n=4, items_log=log, judge_model="other-judge")
    assert sc3["status"] == "skipped" and "DIFFERENT run config" in sc3["reason"]
    sc4 = _run(FakeKijito(commit="deadbeef"), data_file, n=4, items_log=log)
    assert sc4["status"] == "skipped" and "server-commit" in sc4["reason"]


def test_judge_error_marks_run_invalid_and_is_retried_on_resume(data_file, llm, tmp_path):
    log = str(tmp_path / "items.jsonl")
    cache = str(tmp_path / "ctx.jsonl")
    llm.judge_text = "not json"
    sc = _run(FakeKijito(), data_file, items_log=log, context_cache=cache)
    assert sc["status"] == "invalid" and sc["metrics"]["errors"] == 2
    llm.judge_text = _judge_json(True, True, True)
    sc2 = _run(FakeKijito(), data_file, items_log=log, context_cache=cache)
    assert sc2["status"] == "ran" and sc2["metrics"]["overall_mean_of_six"] == 100.0


def test_qids_file_and_shard(data_file, llm, tmp_path):
    q = tmp_path / "q.txt"
    q.write_text("# frozen\nu3\nu1\n")
    sc = _run(FakeKijito(), data_file, n=0, qids_file=str(q))
    assert [i["uid"] for i in sc["items"]] == ["u3", "u1"]
    q.write_text("nope\n")
    assert _run(FakeKijito(), data_file, n=0, qids_file=str(q))["status"] == "skipped"
    sh = _run(FakeKijito(), data_file, n=0, shard="1/2")
    assert [i["uid"] for i in sh["items"]] == ["u2", "u4"] and sh["manifest"]["shard"] == "1/2"


def test_items_log_carries_the_per_scenario_record(data_file, llm, tmp_path):
    log = str(tmp_path / "items.jsonl")
    _run(FakeKijito(), data_file, items_log=log)
    recs = [json.loads(x) for x in open(log)][1:]
    assert {"qid", "arm", "category", "ok", "pass", "answers", "reasoning", "ctx_tokens", "n_ctx",
            "judge_parse_fallback", "error_stage"} <= set(recs[0])
    assert recs[0]["pass"] == {"dim1": True, "dim2": False, "dim3": True} and recs[0]["ok"] is False


def test_blind_needs_the_account_enrolled(data_file, llm):
    sc = _run(FakeKijito(), data_file, blind=True)
    assert sc["status"] == "refused" and "fts-blind" in sc["reason"]
    sc = _run(FakeKijito(blind={"active": True, "index_exists": True}), data_file, blind=True)
    assert sc["status"] == "ran" and sc["manifest"]["kijito_arm"]["blind"]["on"] is True


def test_retrieval_only_fills_the_cache_and_spends_nothing(data_file, llm, tmp_path):
    cache = str(tmp_path / "ctx.jsonl")
    fake = FakeKijito()
    sc = _run(fake, data_file, context_cache=cache, retrieval_only=True)
    assert sc["status"] == "retrieval-only" and sc["retrieved_probes"] == 6 and not sc["retrieval_errors"]
    assert llm.calls == []
    n_calls = len(fake.calls)
    sc2 = _run(fake, data_file, context_cache=cache)
    assert sc2["status"] == "ran" and len(fake.calls) == n_calls and llm.calls


def test_paid_provider_without_key_skips_before_any_work(data_file):
    sc = S.run(n=1, k=3, model="openai/gpt-4o-mini", agent_provider="openrouter", data_path=data_file,
               _api_client=make_client(FakeKijito()))
    assert sc["status"] == "skipped" and "OPENROUTER_API_KEY" in sc["reason"]
