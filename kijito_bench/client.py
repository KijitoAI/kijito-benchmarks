"""A small client for Kijito's public agent interface: the MCP endpoint at <base>/mcp/ (streamable HTTP,
JSON-RPC `tools/call` with `Authorization: Bearer <key>`), plus the public GET /api/version.

Every benchmark arm in this repository ingests and recalls through it, so a number comes from the same
calls any agent makes: kijito_remember_batch to write, kijito_recall to read. Everything server-side
(storage, encryption, ranking, reply formatting) is whatever the service does for every user.

Transport rules (the harness's retrieval-error contract depends on them):
- RETRYABLE, with exponential backoff: connection refused/reset before a reply, HTTP 429/502/503/504,
  and a tool error whose text says rate-limited / at capacity / overloaded. These mean the call was
  not carried out, so resending cannot duplicate a write.
- NOT retried: a read timeout on a WRITE (the server may have stored the batch; a resend could
  duplicate it), any other tool error, a malformed reply. These raise KijitoAPIError, which the
  harness records as a retrieval error (never cached, run marked invalid).
- An account that has reached its memory cap refuses the write with a tool error; that raises
  MemoryCapReached (a KijitoAPIError), so the affected items are errors and the run is invalid. A
  refused write is never read back as an empty recall.
- Reads (recall/get/tools/list) are idempotent, so a read timeout is retried too.

The transport is injectable (`transport=`) so unit tests run with no network.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

# The edge in front of api.kijito.ai refuses urllib's default User-Agent; a browser-style UA keeps the
# client usable against the hosted service.
_UA = "Mozilla/5.0 (kijito-benchmarks harness)"
_RETRY_STATUS = {429, 502, 503, 504}
_RETRY_TEXT = ("rate limit", "rate_limited", "too many requests", "at capacity", "overloaded",
               "try again shortly", "temporarily unavailable")
# The text the service uses when an account is full (kijito_remember / kijito_remember_batch).
_CAP_TEXT = "memory limit reached"
_WRITE_TOOLS = {"kijito_remember", "kijito_remember_batch"}

# kijito_remember_batch limits: at most 60 items per call (the tool's schema), and one request must stay
# under the hosted edge's 1 MiB body limit. The byte budget leaves headroom for the JSON-RPC envelope.
BATCH_MAX_ITEMS = 60
BATCH_MAX_BYTES = 900_000
# The service refuses one memory whose content exceeds 65,536 bytes. An agent that hits that limit splits
# the text, so the harness does the same for the rare oversized turn (LongMemEval-S has one): consecutive
# parts, each keeping the turn's "[date] role: " stamp plus a "(part i/n)" marker. Notes under the limit
# are untouched.
CONTENT_MAX_BYTES = 65_536
PART_MAX_BYTES = 60_000
_STAMP_RE = re.compile(r"^(\[[^\]]*\] [^:\s]+: )")

_REMEMBERED_RE = re.compile(r"Remembered (\d+) memories in one batch: ((?:\[\d+\](?:, )?)*)")
_ID_RE = re.compile(r"\[(\d+)\]")
# kijito_recall's reply text comes in TWO forms: LEAN by default, DETAILED with explain=true.
#
# DETAILED: the header line `[id] score=... scope=... importance=...`, then (after the belief /
# warning lines) the fenced body `⟦UNTRUSTED id=<id> ... n=<nonce>⟧\n<body>\n⟦/UNTRUSTED n=<nonce>⟧`.
_HEADER_RE = re.compile(r"^\[(\d+)\] score=([-0-9.]+) scope=(\S+) importance=([-0-9.]+)(?: owner=(\S+))?",
                        re.M)
_FENCE_RE = re.compile(r"⟦UNTRUSTED id=(\d+) [^\n]*?n=([0-9a-f]+)⟧\n(.*?)\n⟦/UNTRUSTED n=\2⟧", re.S)
# LEAN: ONE line per memory, `- [id] YYYY-MM-DD{ · flag} ⟦n=<nonce> <body> n=<nonce>⟧{ [+N chars]}`,
# then optional INDENTED lines (advisories / warnings). A lean body holds no line break and no ⟦/⟧
# bracket, so the body is everything between the `⟦n=<nonce> ` open and the ` n=<nonce>⟧` close on that
# line. Nothing else in the reply starts a line with `- [`, so _LEAN_START counts entries independently
# of the full grammar (a start that does not fit _LEAN_RE is a parse failure, never a silently dropped
# memory).
_LEAN_START_RE = re.compile(r"^- \[(\d+)\] ", re.M)
_LEAN_RE = re.compile(r"^- \[(\d+)\](?P<meta>(?: [^⟦⟧\n]*?)?) ⟦n=(?P<n>\w{1,32}) (?P<body>[^⟦⟧\n]*) "
                      r"n=(?P=n)⟧(?: \[\+(?P<more>\d+) chars\])?[ \t]*$", re.M)
_LEAN_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
# Shown for every line break of a lean body. NOT reversed by parse_recall: the lean form also drops blank
# lines, collapses whitespace runs inside a line and rewrites ⟦/⟧ to [/], so `' ⏎ ' -> '\n'` is not an
# inverse (and a body that itself contains ' ⏎ ' would be corrupted).
LEAN_LINE_BREAK = " ⏎ "
_FOUND_RE = re.compile(r"^Found (\d+) memories[.:]", re.M)
_PROJECT_RE = re.compile(r"^Project:\s*(.*)$", re.M)


class KijitoAPIError(RuntimeError):
    """A call that failed for good (after any retries). Never treat it as an empty result."""


class MemoryCapReached(KijitoAPIError):
    """A write refused because the account holds as many memories as its plan allows."""


@dataclass
class RecallItem:
    """One recalled memory. `content` is the fenced body EXACTLY as the reply carries it (unfenced, not
    otherwise altered). In the LEAN form the body is already normalised for one-line display (see
    LEAN_LINE_BREAK); score and scope are not printed there, so they are None."""
    id: int
    score: float | None
    scope: str | None
    content: str
    form: str = "detailed"          # "detailed" (explain=true) or "lean" (the default reply)
    date: str | None = None         # lean only: the saved date (YYYY-MM-DD) from the entry header
    flags: tuple = ()               # lean only: the non-default header flags (`by x`, `pinned`, ...)
    more_chars: int = 0             # characters cut by a truncated render


@dataclass
class CallStats:
    calls: int = 0
    retries: int = 0
    by_tool: dict = field(default_factory=dict)


def _urllib_transport(method: str, url: str, headers: dict, body: bytes | None, timeout: float):
    """(status, body_text). Raises OSError/URLError for transport failures."""
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def _parse_rpc_body(text: str) -> dict:
    """A JSON-RPC response, whether the server answered application/json or one SSE `data:` event."""
    s = text.strip()
    if s.startswith("{"):
        return json.loads(s)
    for line in s.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                return json.loads(payload)
    raise KijitoAPIError(f"unparseable MCP reply: {s[:200]!r}")


class KijitoMCPClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 600.0, max_retries: int = 6,
                 backoff_s: float = 2.0, transport=None, sleep=time.sleep):
        self.base = base_url.rstrip("/")
        self._key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._transport = transport or _urllib_transport
        self._sleep = sleep
        self._next_id = 0
        self.stats = CallStats()

    # ── plumbing ──────────────────────────────────────────────────────────────────────────────
    def _headers(self, json_body: bool) -> dict:
        h = {"Authorization": f"Bearer {self._key}", "User-Agent": _UA,
             "Accept": "application/json, text/event-stream"}
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _rpc(self, method: str, params: dict, *, is_write: bool) -> dict:
        self._next_id += 1
        body = json.dumps({"jsonrpc": "2.0", "id": self._next_id, "method": method,
                           "params": params}).encode()
        attempt = 0
        while True:
            attempt += 1
            retryable_reason = None
            try:
                status, text = self._transport("POST", f"{self.base}/mcp/", self._headers(True), body,
                                               self.timeout)
            except (TimeoutError, OSError) as e:   # URLError is an OSError
                timed_out = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
                if timed_out and is_write:
                    raise KijitoAPIError(f"{method}: write timed out after {self.timeout}s — not "
                                         "resent (the server may have stored it; a resend could "
                                         "duplicate)") from e
                retryable_reason = f"transport: {type(e).__name__}: {e}"
            else:
                if status in _RETRY_STATUS:
                    retryable_reason = f"HTTP {status}: {text[:160]}"
                elif status != 200:
                    raise KijitoAPIError(f"{method}: HTTP {status}: {text[:300]}")
                else:
                    reply = _parse_rpc_body(text)
                    if "error" in reply:
                        msg = json.dumps(reply["error"])[:300]
                        if any(t in msg.lower() for t in _RETRY_TEXT):
                            retryable_reason = f"rpc error: {msg}"
                        else:
                            raise KijitoAPIError(f"{method}: JSON-RPC error {msg}")
                    else:
                        res = reply.get("result") or {}
                        if res.get("isError"):
                            etext = _text_of(res)
                            if _CAP_TEXT in etext.lower():
                                raise MemoryCapReached(
                                    f"{method}: tool error: {etext[:500]} — the account's memory cap is "
                                    "reached; this run needs more room than the account has (see the "
                                    "README: memory footprint per arm, and how to ask for a raised cap)")
                            if any(t in etext.lower() for t in _RETRY_TEXT):
                                retryable_reason = f"tool error: {etext[:200]}"
                            else:
                                raise KijitoAPIError(f"{method}: tool error: {etext[:500]}")
                        else:
                            return res
            if attempt > self.max_retries:
                raise KijitoAPIError(f"{method}: gave up after {attempt} attempts ({retryable_reason})")
            self.stats.retries += 1
            self._sleep(min(60.0, self.backoff_s * (2 ** (attempt - 1))))

    def call_tool(self, name: str, arguments: dict) -> str:
        """The text an agent reads from a tools/call (all text content parts, joined)."""
        self.stats.calls += 1
        self.stats.by_tool[name] = self.stats.by_tool.get(name, 0) + 1
        res = self._rpc("tools/call", {"name": name, "arguments": arguments},
                        is_write=name in _WRITE_TOOLS)
        return _text_of(res)

    def tools_list(self) -> list[dict]:
        return list(self._rpc("tools/list", {}, is_write=False).get("tools") or [])

    def version(self) -> dict:
        status, text = self._transport("GET", f"{self.base}/api/version", {"User-Agent": _UA}, None,
                                       60.0)
        if status != 200:
            raise KijitoAPIError(f"/api/version: HTTP {status}: {text[:200]}")
        return json.loads(text)

    # ── the agent-shaped operations the harness uses ─────────────────────────────────────────
    def remember_batch(self, notes: list[str], project: str, importance: float = 0.5) -> list[int]:
        """Write `notes` IN ORDER as project-scoped memories, split into tool-legal batches (<= 60
        items, <= BATCH_MAX_BYTES of arguments). Returns the new ids in note order; raises if any batch
        reply does not account for exactly its items."""
        ids: list[int] = []
        notes = [part for n in notes for part in split_oversized(n)]
        for chunk in _chunks(notes, project, importance):
            text = self.call_tool("kijito_remember_batch",
                                  {"memories": chunk, "project": project})
            got = parse_remembered_ids(text)
            if len(got) != len(chunk):
                raise KijitoAPIError(f"kijito_remember_batch stored {len(got)} of {len(chunk)} "
                                     f"items: {text[:300]}")
            ids.extend(got)
        return ids

    def recall(self, query: str, *, limit: int, project: str | None = None,
               scope: str | None = None, explain: bool = False) -> str:
        """kijito_recall's reply text. explain=True asks for the DETAILED form (score/scope/importance
        header, belief line, full UNTRUSTED fence, scoring breakdown); the default is the server's
        default (LEAN) form. `explain` is only sent when True, so a default call is the exact argument
        set an agent sends."""
        args = {"query": query, "limit": int(limit)}
        if scope:
            args["scope"] = scope
        if project is not None:
            args["project"] = project
        if explain:
            args["explain"] = True
        return self.call_tool("kijito_recall", args)

    def project_of(self, memory_id: int) -> str | None:
        """The project kijito_get reports for one memory ('' for none, None if unparseable)."""
        text = self.call_tool("kijito_get", {"id": int(memory_id)})
        m = _PROJECT_RE.search(text)
        if not m:
            return None
        v = m.group(1).strip()
        return "" if v in ("(global)", "(none)", "") else v


def _text_of(result: dict) -> str:
    parts = [c.get("text", "") for c in (result.get("content") or []) if c.get("type") == "text"]
    return "\n".join(parts)


def split_oversized(note: str, max_bytes: int = PART_MAX_BYTES) -> list[str]:
    """[note] when it fits the per-memory limit; else consecutive parts split on whitespace, each
    <= CONTENT_MAX_BYTES and carrying the note's stamp and a "(part i/n)" marker."""
    if len(note.encode()) <= CONTENT_MAX_BYTES:
        return [note]
    m = _STAMP_RE.match(note)
    stamp = m.group(1) if m else ""
    body = note[len(stamp):]
    pieces, cur, cur_b = [], [], 0
    for word in re.split(r"(\s+)", body):
        wb = len(word.encode())
        if cur and cur_b + wb > max_bytes:
            pieces.append("".join(cur).strip())
            cur, cur_b = [], 0
        while wb > max_bytes:  # one unbroken run longer than a part: cut it by characters
            head, word = word[: max_bytes // 4], word[max_bytes // 4:]
            pieces.append(head)
            wb = len(word.encode())
        cur.append(word)
        cur_b += wb
    if "".join(cur).strip():
        pieces.append("".join(cur).strip())
    n = len(pieces)
    return [f"{stamp}(part {i}/{n}) {p}" for i, p in enumerate(pieces, 1)]


def _chunks(notes, project, importance):
    chunk, size = [], 0
    for n in notes:
        item = {"content": n, "importance": importance, "scope": "project", "project": project}
        isz = len(json.dumps(item).encode())
        if chunk and (len(chunk) >= BATCH_MAX_ITEMS or size + isz > BATCH_MAX_BYTES):
            yield chunk
            chunk, size = [], 0
        chunk.append(item)
        size += isz
    if chunk:
        yield chunk


def parse_remembered_ids(text: str) -> list[int]:
    m = _REMEMBERED_RE.search(text or "")
    if not m:
        return []
    return [int(x) for x in _ID_RE.findall(m.group(2))]


class RecallParseError(KijitoAPIError):
    """A kijito_recall reply whose entries could not all be parsed (never read as fewer memories)."""


def parse_recall(text: str) -> list[RecallItem]:
    """The recalled items in reply order, from EITHER reply form (detected per entry):

    - detailed: id, score, scope and the fenced body (exact, unfenced).
    - lean:     id, saved date, header flags and the fenced body (exact, unfenced: what sits between
                `⟦n=<nonce> ` and ` n=<nonce>⟧`, including any ' ⏎ ' line-break glyphs); score and
                scope are None (the lean form does not print them).

    Raises RecallParseError when a lean entry start does not fit the lean grammar, when a detailed
    header has no matching fence, or when the reply's `Found N memories` count disagrees with the
    number of entries parsed — a reply the parser cannot fully account for is an error, never a
    shorter list."""
    text = text or ""
    fences = list(_FENCE_RE.finditer(text))
    bodies = {int(f.group(1)): f.group(3) for f in fences}
    # Every structural scan below runs on a copy whose DETAILED fence bodies are blanked (same length,
    # line breaks kept), so a stored body line that happens to read `- [12] …`, `[12] score=…` or
    # `Found 3 memories:` can never be taken for a server-written line. Lean bodies need no mask: they
    # are one line and never start one.
    masked = text
    if fences:
        chunks, last = [], 0
        for f in fences:
            a, b = f.start(3), f.end(3)
            chunks += [text[last:a], "".join(c if c == "\n" else " " for c in text[a:b])]
            last = b
        masked = "".join(chunks) + text[last:]
    found = []
    for h in _HEADER_RE.finditer(masked):
        found.append((h.start(), "detailed", h))
    lean_full = {m.start(): m for m in _LEAN_RE.finditer(masked)}
    for st in _LEAN_START_RE.finditer(masked):
        m = lean_full.get(st.start())
        if m is None:
            end = masked.find("\n", st.start())
            line = masked[st.start(): end if end >= 0 else len(masked)]
            raise RecallParseError(f"lean recall entry does not fit the lean grammar: {line[:200]!r}")
        found.append((st.start(), "lean", m))
    found.sort(key=lambda x: x[0])
    text = masked   # lean bodies are identical in both copies; detailed bodies come from `bodies`
    out = []
    for _, form, m in found:
        mid = int(m.group(1))
        if form == "detailed":
            if mid not in bodies:
                raise RecallParseError(f"detailed recall entry [{mid}] has no UNTRUSTED fence")
            out.append(RecallItem(id=mid, score=float(m.group(2)), scope=m.group(3),
                                  content=bodies[mid], form="detailed"))
        else:
            parts = [p.strip() for p in (m.group("meta") or "").split(" · ") if p.strip()]
            date = parts[0] if parts and _LEAN_DATE_RE.match(parts[0]) else None
            flags = tuple(parts[1:] if date else parts)
            out.append(RecallItem(id=mid, score=None, scope=None, content=m.group("body"),
                                  form="lean", date=date, flags=flags,
                                  more_chars=int(m.group("more") or 0)))
    fm = _FOUND_RE.search(text)
    if fm and int(fm.group(1)) != len(out):
        raise RecallParseError(f"reply says Found {fm.group(1)} memories but {len(out)} entries parsed")
    return out


def lean_unbreak(body: str) -> str:
    """A lean body with its line-break glyphs shown as real line breaks again — a READER-FACING
    approximation of the stored body (blank lines and intra-line whitespace runs are not restored;
    see LEAN_LINE_BREAK). Never use it where exact content matters."""
    return (body or "").replace(LEAN_LINE_BREAK, "\n")


def load_key(path: str) -> str:
    """Read an API key file (refusing a group/world-readable one)."""
    import os
    import stat
    st = os.stat(path)
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise KijitoAPIError(f"key file {path} is readable by group/others; chmod 600 it")
    key = open(path).read().strip()
    if not key:
        raise KijitoAPIError(f"key file {path} is empty")
    return key


def host_of(base_url: str) -> str:
    return urllib.parse.urlparse(base_url).netloc


# ── harness-side helpers shared by every arm (longmemeval, stale, stale_agent, groupmembench) ──────
# One implementation of the scope canary, the per-recall leak check, the keyword-index status read,
# the reader-facing body list and the trace writer, so the arms cannot drift apart.

class ScopeCanaryFailed(KijitoAPIError):
    """The scope canary found project-scoped recall crossing projects (or could not show isolation)."""


def scope_canary(api, prefix: str, *, exc=ScopeCanaryFailed, label: str = "bench") -> dict:
    """Scope-isolation canary at the start of a run: two fresh projects share one distinctive phrase.
    A project-scoped recall must return ONLY its own project's memory, and an account-wide recall must
    find BOTH (proof the phrase is findable, so the isolation result is not vacuous). Raises `exc` on
    any violation — a run whose units can see each other is not run."""
    import secrets
    import sys
    nonce = "".join(chr(97 + (b % 26)) for b in secrets.token_bytes(10))
    phrase = f"canary lighthouse marmalade {nonce}"
    pa, pb = f"{prefix}__canary-{nonce}-a", f"{prefix}__canary-{nonce}-b"
    ia = set(api.remember_batch([f"{phrase}: this note belongs to project A"], pa))
    ib = set(api.remember_batch([f"{phrase}: this note belongs to project B"], pb))
    ra = {it.id for it in parse_recall(api.recall(phrase, limit=10, scope="project", project=pa))}
    rb = {it.id for it in parse_recall(api.recall(phrase, limit=10, scope="project", project=pb))}
    rall = {it.id for it in parse_recall(api.recall(phrase, limit=20))}
    res = {"projects": [pa, pb], "a_ids": sorted(ia), "b_ids": sorted(ib), "a_recall": sorted(ra),
           "b_recall": sorted(rb), "all_recall_has_both": (ia | ib) <= rall}
    if ra != ia or rb != ib:
        raise exc(f"API scope canary FAILED: project-scoped recall crossed projects ({res})")
    if not res["all_recall_has_both"]:
        raise exc(f"API scope canary inconclusive: an account-wide recall did not find both "
                  f"canary memories, so isolation was not demonstrated ({res})")
    print(f"[{label}] API scope canary PASSED: project-scoped recall stayed inside its project",
          file=sys.stderr)
    return res


def leak_check(api, items, own_ids, project) -> dict:
    """{memory_id: why} for every recalled memory that is not this unit's. A memory the unit did not
    write is looked up with kijito_get: same project = server-created inside the project (reported,
    not a leak); any other project or a non-project scope = a LEAK."""
    leaks = {}
    for it in items:
        # The detailed form prints each item's scope; the lean form does not (scope=None), and then
        # the project check below carries the whole verdict (a non-project memory has no project).
        if it.scope is not None and it.scope != "project":
            leaks[it.id] = f"scope={it.scope}"
            continue
        if it.id in own_ids:
            continue
        p = api.project_of(it.id)
        if p != project:
            leaks[it.id] = f"project={p!r}"
    return leaks


def blind_status(api) -> dict:
    """GET /api/fts-blind/status with the run's key: whether the account is enrolled in the keyword
    index over encrypted content. Enrolment is an account setting made from a signed-in session, so the
    harness never enrols; it only checks. It covers every project of the key's account."""
    status, text = api._transport("GET", f"{api.base}/api/fts-blind/status", api._headers(False),
                                  None, 60.0)
    if status != 200:
        return {"error": f"HTTP {status}: {text[:200]}"}
    return json.loads(text).get("result") or {}


def reply_bodies(text: str) -> list[str]:
    """The memory bodies of a kijito_recall reply, in reply order, as a READER sees them when the
    harness renders contents instead of the reply (--api-ctx contents): a detailed body is exact; a
    lean body has its line-break glyphs shown as line breaks again (lean_unbreak — reader-facing)."""
    return [lean_unbreak(it.content) if it.form == "lean" else it.content for it in parse_recall(text)]


class Trace:
    """Append-only JSONL of per-unit API-arm records (timings, recalled ids, leaks, calls)."""
    def __init__(self, path):
        self.path = path

    def append(self, rec):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def api_arm_info(api, via_api: str, *, ctx_render: str, k: int, project_prefix: str, explain: bool,
                 blind: bool, project_per: str) -> dict:
    """The manifest block every arm records (the server's identity comes from /api/version, read once
    by the caller and stored on api.server_version_info)."""
    ver = getattr(api, "server_version_info", None) or {}
    return {"mode": "api", "base_url": (via_api or getattr(api, "base", "")).rstrip("/"),
            "base_host": host_of(api.base), "server_commit": ver.get("commit"),
            "server_version": ver.get("version"), "server_started_at": ver.get("started_at"),
            "ingest_tool": "kijito_remember_batch", "batch_max_items": BATCH_MAX_ITEMS,
            "recall_tool": "kijito_recall", "recall_scope": "project", "project_per": project_per,
            "project_prefix": project_prefix, "k": int(k), "recall_explain": bool(explain),
            "ctx_render": ctx_render, "recall_limit_max": getattr(api, "recall_limit_max", None),
            "context_form": ("the kijito_recall MCP reply text, verbatim" if ctx_render == "mcp_text"
                             else "the memory bodies parsed from the kijito_recall reply"),
            "blind": {"on": bool(blind), "scope": "account",
                      "note": "the keyword index over encrypted content is an ACCOUNT setting (enabled from "
                              "a signed-in session); it cannot be switched per project, so --blind means the "
                              "key's whole account is enrolled (checked with GET /api/fts-blind/status)"},
            "note": "the server's build is server_commit (from GET /api/version); this harness sets "
                    "nothing server-side."}


def recall_limit_max(tools) -> int | None:
    """The `maximum` kijito_recall's input schema declares for `limit` (from tools/list), or None when
    the schema does not declare one. Recorded in every manifest so a run states the ceiling it ran under."""
    for t in tools or []:
        if t.get("name") == "kijito_recall":
            lim = ((t.get("inputSchema") or {}).get("properties") or {}).get("limit") or {}
            v = lim.get("maximum")
            return int(v) if isinstance(v, (int, float)) else None
    return None


def connect(via_api: str, api_key_file: str, client=None):
    """(client, None) after reading /api/version + tools/list, or (None, reason) when the service is
    unusable or the key cannot call remember_batch/recall/get."""
    try:
        api = client or KijitoMCPClient(via_api, load_key(api_key_file))
        api.server_version_info = api.version()
        tools = api.tools_list()
        api.recall_limit_max = recall_limit_max(tools)
        tool_names = {t.get("name") for t in tools}
    except Exception as e:
        return None, f"--via-api {via_api or getattr(client, 'base', None)!r} unusable: {type(e).__name__}: {e}"
    need = {"kijito_remember_batch", "kijito_recall", "kijito_get"} - tool_names
    if need:
        return None, f"the service's key cannot call {sorted(need)}"
    return api, None
