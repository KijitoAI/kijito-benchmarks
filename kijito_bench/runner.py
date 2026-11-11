"""Runner scaffolding shared by the benchmark harnesses: checkpointing, concurrency, provenance and
paired statistics, so a long run is resumable, parallel and traceable.

- `ItemLog` is an append-only per-item checkpoint: on restart a harness reloads it and skips the items
  already scored, and every per-item outcome is kept (the input to paired statistics).
- `paired_stats` turns two aligned arms into a McNemar exact test + a bootstrap CI on the accuracy
  difference.
- `run_manifest` stamps the harness git commit (+ dirty flag), machine, UTC time, seed and dataset
  hashes, so a number can be traced to the exact harness code and data that produced it.
- `pmap` is a bounded, order-preserving thread map that parallelizes the reader/judge calls without
  letting one bad item kill the run.
- `ContextCache` persists retrieved contexts so a scoring pass can be repeated without new retrieval.

Standard library only.
"""
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import threading
from datetime import datetime, timezone

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _git(args, cwd=_PKG_DIR):
    """Best-effort `git` invocation in this harness's checkout — returns stripped stdout, or "" if git is
    absent / not a repo / slow."""
    try:
        return subprocess.run(["git"] + list(args), capture_output=True, text=True,
                              timeout=10, cwd=cwd).stdout.strip()
    except Exception:
        return ""


def _repo_root():
    """The harness checkout's top-level directory, or None when it is not a git checkout."""
    return _git(["rev-parse", "--show-toplevel"]) or None


def _sha256_file(path):
    """sha256 of a file, streamed so a large dataset file doesn't load into RAM. None on any error
    (missing file, unreadable) — provenance is best-effort, it must never abort a benchmark."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def run_manifest(seed=None, dataset_paths=None):
    """Provenance for a scorecard: the harness git commit + a dirty flag (uncommitted changes => the
    number isn't reproducible from a clean checkout), the machine, a UTC timestamp, the seed, and the
    sha256 of each dataset file passed in `dataset_paths`. The server's identity is recorded separately
    by each arm (GET /api/version)."""
    commit = _git(["rev-parse", "HEAD"])
    dirty = bool(_git(["status", "--porcelain"]))   # any staged/unstaged/untracked change => dirty
    dataset_sha = {p: _sha256_file(p) for p in (dataset_paths or []) if p and os.path.isfile(p)}
    return {
        "git_commit": commit or None,
        "git_dirty": dirty,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dataset_sha256": dataset_sha,
    }


# ── ItemLog: append-only per-item checkpoint / resume primitive ────────────────────────────────

class ItemLog:
    """Append-only JSONL store of per-item scoring records — the checkpoint that makes a run resumable.

    Each `append` writes ONE self-contained JSON line and flushes+fsyncs it, so a run killed mid-flight
    leaves a valid prefix: `ItemLog.load(path)` re-reads it into {(qid, arm): record} and the harness
    skips those pairs. Record shape: {qid, arm, category, ok, hyp_len, in_tok, out_tok, error}.
    Thread-safe (a lock guards the append) so it composes with `pmap`.
    """
    _FIELDS = ("qid", "arm", "category", "ok", "hyp_len", "in_tok", "out_tok", "error")

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    def append(self, qid, arm, category=None, ok=False, hyp_len=0, in_tok=0, out_tok=0, error=None,
               **extra):
        # `extra` lets one harness log additional per-item fields without widening _FIELDS (which
        # other harnesses index records by). Readers use .get with a default.
        rec = {"qid": qid, "arm": arm, "category": category, "ok": bool(ok), "hyp_len": hyp_len,
               "in_tok": in_tok, "out_tok": out_tok, "error": error, **extra}
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            # Open-per-append + flush + fsync: the line is durable before we move on, so a resume after a
            # hard kill never sees a torn record mid-run (only possibly a torn LAST line, which load skips).
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        return rec

    @staticmethod
    def load(path):
        """{(qid, arm): record} from a JSONL log; {} if it doesn't exist. Last write wins for a repeated
        key. A trailing torn line (crash mid-write) is skipped rather than fatal."""
        out = {}
        if not os.path.exists(path):
            return out
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue   # torn final line from an interrupted write — ignore, keep the valid prefix
                out[(rec.get("qid"), rec.get("arm"))] = rec
        return out


# ── pmap: bounded, order-preserving, fault-isolating parallel map ───────────────────────────────

def pmap(fn, items, workers=8):
    """Map `fn` over `items` with a bounded thread pool, returning results in INPUT order.

    A per-item exception is captured and returned in that item's slot (as the Exception object) instead
    of aborting the whole map — one bad item never kills a long run. Worker count comes from
    BENCH_WORKERS if set (else `workers`); 1 forces a serial, thread-free fallback. Used for the
    reader/judge calls only; retrieval stays serial.
    """
    items = list(items)
    env = os.environ.get("BENCH_WORKERS")
    if env:
        try:
            workers = max(1, int(env))
        except ValueError:
            pass

    def _safe(it):
        try:
            return fn(it)
        except Exception as e:   # noqa: BLE001 — deliberate: propagate as a per-item result, never raise
            return e

    if workers <= 1 or len(items) <= 1:
        return [_safe(it) for it in items]

    results = [None] * len(items)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_safe, it): i for i, it in enumerate(items)}
        for fut in concurrent.futures.as_completed(futs):
            results[futs[fut]] = fut.result()   # _safe never raises, so .result() can't either
    return results


# ── paired_stats: McNemar exact + bootstrap CI on the accuracy difference ────────────────────────

def paired_stats(records_a, records_b, resamples=10000, seed=20260629, ci=0.95):
    """Compare two arms scored on the SAME items (aligned per-item booleans) — a paired test, because the
    arms answer identical questions so their outcomes are correlated (an unpaired test throws that away).

    Returns {n, b, c, mcnemar_p, delta, ci95}:
    - b = items A got right and B wrong; c = items A wrong and B right (the discordant pairs).
    - mcnemar_p = the EXACT two-sided McNemar p-value: under H0 each discordant pair is a fair coin, so the
      count on one side is Binomial(b+c, 0.5); p = min(1, 2 * P[X <= min(b, c)]). Computed directly with
      math.comb — no scipy.
    - delta = acc(A) - acc(B); ci95 = a seeded percentile bootstrap CI on that paired difference.
    """
    a = [bool(x) for x in records_a]
    b = [bool(x) for x in records_b]
    if len(a) != len(b):
        raise ValueError("paired_stats requires aligned equal-length arms")
    n = len(a)

    nb = sum(1 for x, y in zip(a, b) if x and not y)   # A right, B wrong
    nc = sum(1 for x, y in zip(a, b) if y and not x)   # A wrong, B right
    nd = nb + nc
    if nd == 0:
        p = 1.0                                          # no discordant pairs => no evidence of a difference
    else:
        tail = sum(math.comb(nd, i) for i in range(min(nb, nc) + 1))
        p = min(1.0, 2.0 * tail / (2 ** nd))

    acc_a = sum(a) / n if n else 0.0
    acc_b = sum(b) / n if n else 0.0
    delta = acc_a - acc_b

    # Percentile bootstrap on the paired per-item difference d_i = a_i - b_i: resampling d and averaging is
    # exactly resampling the item indices and taking acc(A) - acc(B) on the resample. Seeded => reproducible.
    d = [int(x) - int(y) for x, y in zip(a, b)]
    if n:
        rng = random.Random(seed)
        boot = sorted(sum(rng.choices(d, k=n)) / n for _ in range(resamples))
        lo = boot[int((1 - ci) / 2 * resamples)]
        hi = boot[min(resamples - 1, int((1 + ci) / 2 * resamples))]
        ci95 = [round(lo, 4), round(hi, 4)]
    else:
        ci95 = [0.0, 0.0]

    return {"n": n, "b": nb, "c": nc, "mcnemar_p": round(p, 6),
            "delta": round(delta, 4), "ci95": ci95}


# ── ContextCache: persist retrieved contexts so scoring can be repeated without new retrieval ──────

class ContextCache:
    """A JSONL-persisted cache of retrieved contexts, keyed by a caller-supplied composite key, behind
    every harness's opt-in `--context-cache` flag.

    Ingest + retrieval is most of a run's wall clock; reader and judge passes do not change what was
    retrieved. Paying retrieval ONCE and replaying the cached contexts lets a scoring pass (another reader
    model, a re-judge) run over exactly the same retrieved text.

    The key (built via `make_key`) pins every input that changes what gets retrieved: the dataset, the
    question id, the ARM, the retrieval budget `k`, the dataset digest and, for the Kijito arm, the
    server (host + /api/version commit + project namespace + reply form). A changed configuration simply
    misses and recomputes. The reader/judge model is deliberately NOT in the key — the retrieved context
    is model-independent. When a cache is active a harness discloses its path + hit/miss counts in the
    scorecard manifest, so a scorecard always states whether retrieval was live or replayed.

    Persistence is append-only JSONL ({"key", "contexts"}); a fresh run reloads the whole file. Torn
    final lines from an interrupted write are skipped, not fatal. Thread-safe.
    """
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._mem = {}
        self.hits = 0
        self.misses = 0
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue   # torn final line from an interrupted write — keep the valid prefix
                    if "key" in rec and "contexts" in rec:
                        self._mem[rec["key"]] = rec["contexts"]

    @staticmethod
    def make_key(dataset, qid, arm, k, digest, extra=None):
        """Composite sha256 key. Every argument that changes the RETRIEVED context belongs here; the model
        deliberately does NOT (see the class docstring). `\\x1f`-joined so field boundaries can't collide."""
        parts = [str(dataset), str(qid), str(arm), str(k), str(digest), str(extra or "")]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def get(self, key):
        """Cached contexts (a fresh list copy) for `key`, or None on a miss. Increments the hit/miss counts
        the scorecard manifest discloses — so exactly one get() per (question, arm) tallies the run."""
        with self._lock:
            v = self._mem.get(key)
            if v is None:
                self.misses += 1
                return None
            self.hits += 1
            return list(v)

    def peek(self, key):
        """Like get() but NOT counted in hits/misses — for auxiliary records (e.g. an arm's ingest
        record) that must not distort the per-(question, arm) tally the manifest shows."""
        with self._lock:
            v = self._mem.get(key)
            return None if v is None else list(v)

    def put(self, key, contexts):
        """Persist `contexts` under `key` (durably flushed+fsynced, so a killed run keeps what it computed)."""
        contexts = list(contexts)
        line = json.dumps({"key": key, "contexts": contexts}, ensure_ascii=False)
        with self._lock:
            self._mem[key] = contexts
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    def stats(self):
        return {"hits": self.hits, "misses": self.misses}
