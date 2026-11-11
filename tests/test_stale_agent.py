"""STALE agent-curated arm (kijito_bench/stale_agent.py) against the in-memory fake MCP endpoint with fake
ingest-agent / reader / judge models. No network, no LLM, no dataset download."""
import json

import pytest

from kijito_bench import stale as S
from kijito_bench import stale_agent as A
from conftest import FakeKijito, make_client
from test_stale import _dataset, _judge_json


class _AgentLLM:
    """distill -> two dated memories per session; decide -> link every hinted pair; reader/judge as in
    test_stale."""
    def __init__(self):
        self.calls = []

    def __call__(self, provider, model, prompt, retries=2, system=None, json_mode=False):
        self.calls.append({"provider": provider, "model": model, "prompt": prompt, "system": system})
        if prompt.startswith("You keep long-term memory"):
            date = prompt.split("dated ", 1)[1].split(".", 1)[0]
            return f"- [{date[:10]}] fact one about this session\n- [{date[:10]}] fact two", 100, 10
        if prompt.startswith("You just saved these memories"):
            pairs = [line for line in prompt.splitlines() if line.startswith("- older [")]
            out = []
            for line in pairs:
                old = line.split("[", 1)[1].split("]", 1)[0]
                new = line.rsplit("[", 1)[1].split("]", 1)[0]
                out.append(f"{new} -> {old}")
            return "\n".join(out) or "NONE", 50, 5
        if system == S.JUDGE_SYSTEM:
            return _judge_json(), 1000, 50
        return "The user moved.", 400, 20


@pytest.fixture
def env(monkeypatch, tmp_path):
    data = _dataset()
    monkeypatch.setattr(S, "load_dataset", lambda data_path=None: (data, "stale.json", "sha"))
    llm = _AgentLLM()
    monkeypatch.setattr(A, "_llm", llm)
    monkeypatch.setattr(S, "_llm", llm)
    key = tmp_path / "key"
    key.write_text("kjt_test\n")
    key.chmod(0o600)
    state = {"fake": FakeKijito(hints=True)}
    monkeypatch.setattr(A, "KijitoMCPClient", lambda base, k: make_client(state["fake"]))
    return {"llm": llm, "key": str(key), "tmp": tmp_path, "state": state}


def _main(env, *extra):
    out = str(env["tmp"] / "out.json")
    A.main(["--api-key-file", env["key"], "--n", "2", "--project-prefix", "pre-",
            "--distill-cache", str(env["tmp"] / "distill.jsonl"), "--out", out, *extra])
    return out, json.load(open(out))


def test_curated_hints_on_writes_distilled_memories_and_links_hinted_pairs(env):
    out, sc = _main(env, "--mode", "curated", "--hints", "on")
    fake = env["state"]["fake"]
    assert sc["errors"] == {} and sc["metrics"]["n_scored"] == 2
    writes = [a for n, a in fake.calls if n == "kijito_remember_batch"]
    assert all(m["content"].startswith("[2024-") for a in writes for m in a["memories"])
    assert {a["project"] for a in writes} == {"pre-u1", "pre-u2"}
    assert fake.relations and all(r[2] == "supersedes" for r in fake.relations)
    st = sc["ingest_stats"]["u1"]
    assert st["raw"] == 0 and st["memories"] == 8 and st["relates"] == len(
        [r for r in fake.relations if fake.mem[r[0]]["project"] == "pre-u1"])
    ctx = [json.loads(x) for x in open(out + ".ctx.jsonl")]
    assert {(c["uid"], c["dim"]) for c in ctx} == {(u, d) for u in ("u1", "u2") for d in S._DIMS}
    recalls = [a for n, a in fake.calls if n == "kijito_recall" and "anything" not in a["query"]]
    assert all(a["limit"] == 10 and a["scope"] == "project" for a in recalls)
    assert sc["config"]["mode"] == "curated" and sc["server_commit"] == "c0ffee12"


def test_hints_off_stores_the_same_memories_and_links_nothing(env):
    _main(env, "--mode", "curated", "--hints", "on")
    on_writes = [m["content"] for n, a in env["state"]["fake"].calls if n == "kijito_remember_batch"
                 for m in a["memories"]]
    n_distill = sum(1 for c in env["llm"].calls if c["prompt"].startswith("You keep long-term memory"))
    env["state"]["fake"] = FakeKijito(hints=True)
    _main(env, "--mode", "curated", "--hints", "off")
    fake = env["state"]["fake"]
    off_writes = [m["content"] for n, a in fake.calls if n == "kijito_remember_batch" for m in a["memories"]]
    assert off_writes == on_writes and fake.relations == []
    # the distillations came from the shared cache: no new distill call
    assert sum(1 for c in env["llm"].calls if c["prompt"].startswith("You keep long-term memory")) == n_distill


def test_mode_both_writes_raw_turns_before_the_agents_memories(env):
    _, sc = _main(env, "--mode", "both", "--hints", "on")
    fake = env["state"]["fake"]
    first = [a for n, a in fake.calls if n == "kijito_remember_batch" and a["project"] == "pre-u1"][0]
    assert first["memories"][0]["content"].startswith("[2024-01-01 10:00] user: ")
    assert sc["ingest_stats"]["u1"]["raw"] == 7


def test_reuse_ctx_scores_saved_replies_with_another_reader(env):
    out, _ = _main(env, "--mode", "curated", "--hints", "on")
    env["state"]["fake"] = FakeKijito()
    env["llm"].calls.clear()
    out2 = str(env["tmp"] / "opus.json")
    A.main(["--api-key-file", env["key"], "--n", "0", "--project-prefix", "unused-", "--distill-cache",
            str(env["tmp"] / "none.jsonl"), "--reuse-ctx", out + ".ctx.jsonl", "--agent-provider", "anthropic",
            "--model", "claude-opus-5-5", "--out", out2])
    sc = json.load(open(out2))
    fake = env["state"]["fake"]
    assert [n for n, a in fake.calls] == []                      # no ingest, no recall
    readers = [c for c in env["llm"].calls if c["system"] != S.JUDGE_SYSTEM]
    judges = [c for c in env["llm"].calls if c["system"] == S.JUDGE_SYSTEM]
    assert readers and all(c["provider"] == "anthropic" and c["model"] == "claude-opus-5-5" for c in readers)
    assert judges and all(c["provider"] == "openrouter" and c["model"] == "google/gemini-3.1-flash-lite-preview"
                          for c in judges)
    assert sc["metrics"]["n_scored"] == 2


def test_existing_project_is_an_error(env):
    fake = env["state"]["fake"]
    make_client(fake).remember_batch(["anything at all"], "pre-u1")
    _, sc = _main(env, "--mode", "curated")
    assert set(sc["errors"]) == {f"u1:{d}" for d in S._DIMS}
    assert "already holds memories" in sc["errors"]["u1:dim1"]


def test_full_account_stops_and_marks_the_rest(env):
    env["state"]["fake"] = FakeKijito(hints=True, memory_limit=5)
    _, sc = _main(env, "--mode", "curated")
    assert set(sc["errors"]) == {f"{u}:{d}" for u in ("u1", "u2") for d in S._DIMS}
    assert all("Memory limit reached" in e for e in sc["errors"].values())
    projects = {a["project"] for n, a in env["state"]["fake"].calls if n == "kijito_remember_batch"}
    assert projects == {"pre-u1"}                                  # u2 was never written
