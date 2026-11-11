"""The markdown-notes control arm used by the LongMemEval harness (append_md): the same turn notes the
Kijito arm stores, ranked by Okapi BM25 per question. No model, fully deterministic.

(GroupMemBench's BM25 control is the benchmark's own BM25 baseline and lives in groupmembench.py.)
"""
import math
import re
from collections import Counter
from dataclasses import dataclass

# A small, conventional English stoplist. Kept deliberately short — over-aggressive stopping would
# unfairly weaken the keyword baseline.
_STOP = frozenset("""
a an the of to in on for and or is are was were be been being it its this that these those as at by
with from into about over under than then so such not no nor but if while which who whom whose what
when where why how can could should would may might will shall do does did done has have had having
i you he she they we me him her them us our your their my his its
""".split())


@dataclass
class _Fact:
    id: int
    text: str


def _tok(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, minus a short stoplist."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def bm25_rank(items, query_text: str, limit: int, k1: float = 1.5, b: float = 0.75
              ) -> tuple[list[int], dict[int, float]]:
    """Okapi BM25 — IDF-weighted, length-normalised. Deterministic (corpus-fixed IDF; id tiebreak).
    `items` are objects with `.id` and `.text`; returns (ranked ids[:limit], {id: score})."""
    docs = [_tok(it.text) for it in items]
    N = len(docs)
    avgdl = (sum(len(d) for d in docs) / N) if N else 0.0
    df: Counter = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    # Okapi IDF (the standard non-negative form).
    idf = {t: math.log(1.0 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    q = _tok(query_text)
    scored: list[tuple[float, int]] = []
    for it, d in zip(items, docs):
        if not d:
            continue
        tf = Counter(d)
        dl = len(d)
        s = 0.0
        for t in q:
            f = tf.get(t, 0)
            if f:
                s += idf.get(t, 0.0) * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / avgdl))
        if s > 0.0:
            scored.append((s, it.id))
    scored.sort(key=lambda x: (-x[0], x[1]))
    ranking = [iid for _, iid in scored][:limit]
    scores = {iid: sc for sc, iid in scored}
    return ranking, scores
