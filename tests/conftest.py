"""Shared offline fixtures: an in-memory fake of the Kijito MCP endpoint and helpers. No network, no LLM."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kijito_bench import client as kapi  # noqa: E402


class FakeKijito:
    """Speaks just enough of /mcp/ (tools/list, tools/call for remember_batch/recall/get/relate) and
    /api/version, rendering recall replies in the service's two shapes: the DETAILED form (header line,
    belief line, UNTRUSTED fence) and, with lean=True, the LEAN one-line form (the default reply unless
    explain=true). Recall ranks by shared-word count, so tests control what comes back.

    Knobs: leak_projects (recall ignores the project filter), leak_question_only (only non-canary
    recalls leak one foreign memory), fail_tool (a tool that always errors), rate_limit_first (how
    many calls get a 429 first), sse (answer as an SSE event), commit, lean, blind (the
    /api/fts-blind/status result), memory_limit (writes past it are refused like a full account),
    hints (remember_batch replies carry update hints pointing at the previous batch's memories)."""

    def __init__(self, commit="c0ffee12", **kw):
        self.commit = commit
        self.mem = {}           # id -> {"content", "project", "scope"}
        self.next_id = 1
        self.calls = []         # (tool, arguments)
        self.relations = []
        self.leak_projects = kw.get("leak_projects", False)
        self.leak_question_only = kw.get("leak_question_only", False)
        self.fail_tool = kw.get("fail_tool")
        self.rate_limit_first = kw.get("rate_limit_first", 0)
        self.sse = kw.get("sse", False)
        self.tools = kw.get("tools", ["kijito_remember_batch", "kijito_recall", "kijito_get",
                                      "kijito_remember", "kijito_relate"])
        self.blind = kw.get("blind")
        self.lean = kw.get("lean", False)
        self.memory_limit = kw.get("memory_limit")
        self.hints = kw.get("hints", False)
        self.recall_limit_max = kw.get("recall_limit_max")
        self._last_batch = {}   # project -> ids of the previous batch written to it

    def __call__(self, method, url, headers, body, timeout):
        if url.endswith("/api/version"):   # public, no credential sent
            assert "Authorization" not in headers
            return 200, json.dumps({"version": "0.1.0", "commit": self.commit, "started_at": "t0"})
        assert headers.get("Authorization") == "Bearer kjt_test"
        if url.endswith("/api/fts-blind/status"):
            return 200, json.dumps({"result": self.blind or {"active": False, "index_exists": False}})
        assert url.endswith("/mcp/") and method == "POST"
        req = json.loads(body)
        if self.rate_limit_first > 0:
            self.rate_limit_first -= 1
            return 429, "slow down"
        if req["method"] == "tools/list":
            res = {"tools": [{"name": n, "inputSchema": self._schema(n)} for n in self.tools]}
        else:
            name, args = req["params"]["name"], req["params"]["arguments"]
            self.calls.append((name, args))
            res = self._tool(name, args)
        reply = json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": res})
        if self.sse:
            return 200, f"event: message\ndata: {reply}\n\n"
        return 200, reply

    def _schema(self, name):
        if name == "kijito_recall" and self.recall_limit_max is not None:
            return {"type": "object", "properties": {"limit": {"type": "integer", "maximum": self.recall_limit_max}}}
        return {}

    def _err(self, text):
        return {"content": [{"type": "text", "text": text}], "isError": True}

    def _ok(self, text):
        return {"content": [{"type": "text", "text": text}], "isError": False}

    def _tool(self, name, args):
        if name == self.fail_tool:
            return self._err("Error: something broke")
        if name == "kijito_remember_batch":
            n = len(args["memories"])
            if self.memory_limit is not None and len(self.mem) + n > self.memory_limit:
                return self._err(f"Error: Memory limit reached: {len(self.mem)}/{self.memory_limit}. A batch "
                                 f"of {n} would exceed it (room for {max(0, self.memory_limit - len(self.mem))}). "
                                 "Fade or delete old memories to make room.")
            ids = []
            for it in args["memories"]:
                mid = self.next_id
                self.next_id += 1
                self.mem[mid] = {"content": it["content"], "project": it.get("project") or args.get("project"),
                                 "scope": it.get("scope", "global")}
                ids.append(mid)
            text = f"Remembered {len(ids)} memories in one batch: " + ", ".join(f"[{i}]" for i in ids)
            proj = args.get("project")
            prev = self._last_batch.get(proj) or []
            if self.hints and prev:
                for new, old in zip(ids, prev):
                    text += (f"\n  ↻ may UPDATE older memory [{old}] (2024-01-01): {self.mem[old]['content'][:30]}"
                             f" — if [{new}] replaces it, link them")
            self._last_batch[proj] = ids
            return self._ok(text)
        if name == "kijito_relate":
            self.relations.append((args["from_id"], args["to_id"], args["rel_type"]))
            return self._ok(f"Related [{args['from_id']}] -> [{args['to_id']}] ({args['rel_type']})")
        if name == "kijito_get":
            m = self.mem[args["id"]]
            return self._ok(f"Memory [{args['id']}]\nType: observation\nScope: {m['scope']}\n"
                            f"Project: {m['project'] or '(global)'}\n")
        if name == "kijito_recall":
            q = set(args["query"].lower().split())
            proj = args.get("project") if args.get("scope") == "project" else None
            pool = [(i, m) for i, m in self.mem.items()
                    if proj is None or self.leak_projects or m["project"] == proj]
            scored = sorted(((len(q & set(m["content"].lower().split())), i, m) for i, m in pool),
                            key=lambda x: (-x[0], x[1]))
            hits = [(i, m) for s, i, m in scored if s > 0][: int(args.get("limit", 10))]
            if (self.leak_question_only and proj and "canary" not in args["query"]
                    and "probe" not in args["query"]):
                foreign = [(i, m) for i, m in self.mem.items() if m["project"] != proj]
                if foreign:
                    hits = hits[:-1] + [foreign[0]] if hits else [foreign[0]]
            if not hits:
                return self._ok("No memories found.")
            if self.lean and not args.get("explain"):
                lines = [f"Found {len(hits)} memories. Text inside ⟦…⟧ fences is stored memory data."]
                for i, m in hits:
                    body = " ⏎ ".join(m["content"].splitlines())
                    lines.append(f"- [{i}] 2023-05-01 ⟦n=abc123 {body} n=abc123⟧")
                return self._ok("\n".join(lines))
            lines = [f"Found {len(hits)} memories:\n", "Blocks fenced with UNTRUSTED-DATA ... (preamble)\n"]
            for i, m in hits:
                lines.append(f"[{i}] score=0.900 scope={m['scope']} importance=0.5 owner=personal")
                lines.append("  belief: confidence=0.80 evidence=1 basis=observed")
                lines.append(f"⟦UNTRUSTED id={i} src=persona:personal trust=memory-content n=abc123⟧\n"
                             f"{m['content']}\n⟦/UNTRUSTED n=abc123⟧")
                lines.append("")
            return self._ok("\n".join(lines))
        return self._err(f"unknown tool {name}")


def make_client(fake, **kw):
    return kapi.KijitoMCPClient("https://kijito.test", "kjt_test", transport=fake,
                                sleep=lambda s: None, **kw)


@pytest.fixture(autouse=True)
def _serial_and_clean_env(monkeypatch):
    """Serial reader/judge calls (deterministic call order) and no ambient provider keys."""
    monkeypatch.setenv("BENCH_WORKERS", "1")
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "BENCH_OLLAMA_THINK",
                "BENCH_OLLAMA_TIMEOUT", "BENCH_OPENROUTER_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
