# Results

One directory per published run: `results/<benchmark>/<reader>/`.

| File | Contents |
|---|---|
| `items.jsonl` | One record per question (per scenario for STALE), in the harness's own items format (see "Output files" in the top-level README): the question id and type, the reader's answer, the judge's verdict (`ok`), token counts. |
| `summary.json` | Totals recomputed from `items.jsonl`, plus the run's public facts: reader and judge models, dataset revision and sha256, split file, `k`, recall reply form, keyword-search setting, the Kijito build that answered, the run date, and notes on anything retried. |

`stale-agent-curated/distill-cache.jsonl.gz` is the ingest agent's per-session distillations, shared by
both STALE agent-curated arms (see "Rerunning with the published distillations" in the top-level README).

## How these runs were made

Each run drove the Kijito production build named in its `summary.json` through the MCP endpoint only
(`kijito_remember_batch` to write, `kijito_recall` to read), on our own machine rather than on the hosted
service, so ingest was not paced by a hosted account's write rate limit. The service code was the build
deployed on kijito.ai at the time.

The runs were made with the private harness this repository was extracted from. Parity tests compare
the two on the same inputs and confirm the same MCP calls and arguments, the same reader and judge
prompts, the same per-question records and the same metrics for every published arm. The private
harness's run manifests are not published, because they describe that harness's own process; the
facts a rerun needs are in `summary.json`.

## Retries

The harness appends to its items log, so a question asked again appears more than once there. Here,
each question keeps its last record that has no error; `summary.json` lists every question that was
asked more than once (`questions_asked_more_than_once`) and its `note` says why. In every case the
earlier attempts ended in an error; no question was asked again after it had produced an answer.

## Checking a total

```bash
python - <<'EOF'
import json
recs = [json.loads(l) for l in open("results/longmemeval-s-heldout/gpt-4o/items.jsonl")]
print(sum(r["ok"] in (True, "True") for r in recs), "/", len(recs))
EOF
```
