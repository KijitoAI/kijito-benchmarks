# kijito-benchmarks

Harness code for running three public long-term-memory benchmarks against [Kijito](https://kijito.ai):
LongMemEval, STALE and GroupMemBench.

The harness talks to Kijito only through its public MCP endpoint (`https://api.kijito.ai/mcp/`, with a
Bearer API key), the same interface any agent uses: `kijito_remember_batch` to write, `kijito_recall` to
read, `kijito_get` to check which project a recalled memory belongs to, and (for one arm) `kijito_relate`.
It does not contain Kijito code, does not change anything on the server, and does not depend on any
server setting. Anyone with a Kijito account can rerun the arms below; a full run needs more memories than a
free account holds, so see "Account size" for what fits and how to get a raised cap.

Results are not part of this repository's code. Published result files go in [`results/`](results/).

## Contents

| Path | What it is |
|---|---|
| `kijito_bench/client.py` | MCP client: batching, retries, reply parsing, scope canary, leak check |
| `kijito_bench/llm.py` | Reader and judge calls (OpenRouter, OpenAI, Anthropic, local Ollama), token prices |
| `kijito_bench/runner.py` | Per-item checkpoint log, context cache, parallel scoring, paired statistics, run manifest |
| `kijito_bench/baselines.py` | BM25 over the same notes (the LongMemEval `append_md` control arm) |
| `kijito_bench/longmemeval.py` | LongMemEval |
| `kijito_bench/stale.py` | STALE, raw-ingest arm |
| `kijito_bench/stale_agent.py` | STALE, agent-curated arm |
| `kijito_bench/groupmembench.py` | GroupMemBench (Kijito arm + the benchmark's own BM25 control) |
| `splits/` | The frozen LongMemEval-S dev and held-out question lists |
| `tests/` | Offline tests (fake MCP server, fake models) |
| `results/` | Published result files (added separately) |

## Setup

### 1. A Kijito account and API key

1. Sign up at <https://kijito.ai> and verify your email address. An unverified account can hold only a
   small number of memories.
2. On the **Account** page, create an API key with read and write access to memories
   (`memory.read` and `memory.write`). The key starts with `kjt_` and is shown once. The REST
   equivalent is documented at <https://kijito.ai/docs/api>.
3. Save the key in a file that only you can read:

   ```bash
   mkdir -p ~/.kijito && printf '%s\n' 'kjt_...' > ~/.kijito/bench.key && chmod 600 ~/.kijito/bench.key
   ```

   The harness refuses a key file that is readable by group or others.

Use an account (or a set of fresh project prefixes) dedicated to benchmarking. Every run writes
benchmark data into your account: one project per question (LongMemEval), per scenario (STALE) or per
domain (GroupMemBench), named `<prefix><id>`, plus two small projects for the scope check described
below. A project that already holds memories is refused, so every new ingest needs a fresh
`--api-project-prefix` (`--project-prefix` for the agent arm).

### 2. Model provider keys

The reader and judge models are called directly by the harness. Set the keys for the providers you use:

```bash
export OPENROUTER_API_KEY=...     # the protocol readers and judges (GPT-4o, GPT-4o-mini, GPT-5, Gemini)
export ANTHROPIC_API_KEY=...      # only for the disclosed Claude Opus 5.5 reader arms
export OPENAI_API_KEY=...         # only if you choose --agent-provider openai
```

OpenRouter requests for `openai/...` and `anthropic/...` models are pinned to that vendor with fallbacks
off, so the judge is the named model and never a substitute.

### 3. Install

Python 3.10 or newer.

```bash
git clone <this repository> && cd kijito-benchmarks
python3 -m venv .venv && . .venv/bin/activate
pip install -e .            # requests, huggingface_hub, tiktoken
pip install pytest && pytest -q   # optional: offline tests, no network, no keys
```

Datasets:

- **LongMemEval** and **STALE** download from Hugging Face on first use, at pinned revisions
  (LongMemEval `xiaowu0162/longmemeval-cleaned` @ `98d7416c`, STALE `STALEproj/STALE` @ `617c51dc`).
  The manifest of every run records the sha256 of the file it read.
- **GroupMemBench** is read from a clone of the benchmark repository at the release the harness
  reproduces:

  ```bash
  git clone https://github.com/UCSB-NLP-Chang/GroupMemBench ~/.cache/groupmembench
  git -C ~/.cache/groupmembench checkout e2682e01ff490acfe4fac2940159dce60307dfc9
  ```

  (or pass `--data-root`, or set `GROUPMEMBENCH_ROOT`).

## Account size

Each memory the harness writes counts toward your account's memory cap. During the beta a free account
holds up to **10,000 memories**; the Account page shows your plan's cap. A full run of any arm needs far
more than that:

| Arm | Memories written | Full run |
|---|---|---|
| LongMemEval-S | one per chat turn, about 494 per question | held-out split (350 q): 173,050; dev split (150 q): 73,689 |
| STALE, raw ingest | one per chat turn, 586 to 600 per scenario | all 400 scenarios: 237,358 |
| STALE, agent-curated | the ingest agent's memories, about 400 per scenario (`--mode curated`) | about 160,000 per arm; the hints-on and hints-off arms each need their own projects |
| GroupMemBench | one per message, 30,000 per domain; every question of a domain is asked against the whole domain | all 4 domains: 120,000; one domain: 30,000 |

Every run also writes two small memories for the scope check.

**Runs that fit a free account (10,000 memories):**

```bash
# LongMemEval-S: 15 questions of the dev split (7,382 memories)
python -m kijito_bench.longmemeval --split s --k 50 --official-eval \
  --agent-provider openrouter --model openai/gpt-4o-2024-08-06 \
  --qids-file splits/lme-s-dev.qids --shard 0/10 --no-append-md \
  --api-key-file ~/.kijito/bench.key --api-project-prefix lme-sample1- \
  --out out/lme-sample.json --context-cache out/lme-sample.ctx.jsonl

# STALE raw ingest: 12 scenarios, stratified over both conflict types (7,106 memories)
python -m kijito_bench.stale --n 12 --k 10 \
  --agent-provider openrouter --model openai/gpt-4o-mini \
  --judge-provider openrouter --judge-model google/gemini-3.1-flash-lite-preview \
  --api-key-file ~/.kijito/bench.key --api-project-prefix stale-sample1- \
  --out out/stale-sample.json --context-cache out/stale-sample.ctx.jsonl

# STALE agent-curated: 10 scenarios per arm (about 4,000 memories each; both arms fit together)
python -m kijito_bench.stale_agent --n 10 --mode curated --hints on \
  --api-key-file ~/.kijito/bench.key --project-prefix sa-on-sample1- \
  --distill-cache out/distill.jsonl --out out/sa-on-sample.json
python -m kijito_bench.stale_agent --n 10 --mode curated --hints off \
  --api-key-file ~/.kijito/bench.key --project-prefix sa-off-sample1- \
  --distill-cache out/distill.jsonl --out out/sa-off-sample.json
```

GroupMemBench cannot be sampled into a free account: even one question needs its whole 30,000-message
domain. `--arms bm25` runs the BM25 control without Kijito.

**A full rerun needs a raised cap.** To rerun a full arm for research, email hello@kijito.ai to raise
your account's cap (and its write rate). When the cap is reached, the service refuses the write; the harness stops writing,
records every remaining question or scenario as a retrieval error, and marks the run `invalid` (or
refuses to start, if the account is already full). It never scores a refused write as an empty recall.

Writes are also rate-limited per account. When the service answers "rate limit", the client waits and
retries; ingest time on a hosted account is set by that limit. Shards that share one account share the
limit.

## Running the published arms

Common to every arm:

- `--via-api` defaults to `https://api.kijito.ai`. `--api-key-file` is required.
- Large runs are split into shards (`--shard I/N`), one process per shard, each with its own `--out`,
  `--items-log` and `--context-cache`; the per-question records of all shards together are the run.
  A shard loop, for example:

  ```bash
  N=6; for i in $(seq 0 $((N-1))); do
    python -m kijito_bench.longmemeval ... --shard "$i/$N" --out "out/x/shard$i.json" \
      --items-log "out/x/shard$i.items.jsonl" --context-cache "out/x/ctx-shard$i.jsonl" \
      > "out/x/shard$i.log" 2>&1 &
  done; wait
  ```

- A killed run resumes from its items log: rerun the same command. Records with an error are retried;
  clean records are reused. The log refuses to resume under a different configuration (models, k, data,
  harness code, server commit, project prefix, reply form).

### LongMemEval-S (held-out split, protocol reader and judge)

Reader and judge `gpt-4o-2024-08-06`, the official per-type judge prompts, k = 50 (the number of memories
the harness asks `kijito_recall` for), the reader sees the `kijito_recall` reply text as returned.

```bash
python -m kijito_bench.longmemeval --split s --k 50 --official-eval \
  --agent-provider openrouter --model openai/gpt-4o-2024-08-06 \
  --qids-file splits/lme-s-holdout.qids --shard 0/6 --no-append-md \
  --api-key-file ~/.kijito/bench.key --api-project-prefix holdout-run1- --api-ctx mcp_text \
  --out out/holdout-gpt4o/shard0.json --items-log out/holdout-gpt4o/shard0.items.jsonl \
  --context-cache out/holdout-ctx/ctx-shard0.jsonl
```

**Disclosed Claude Opus 5.5 reader arm.** Same retrievals, a different reader; the judge stays
`gpt-4o-2024-08-06`. Run it after the GPT-4o arm with the same `--api-project-prefix` and the same
`--context-cache` files, so every retrieval is replayed from the cache and nothing is written or recalled
again:

```bash
python -m kijito_bench.longmemeval --split s --k 50 --official-eval \
  --agent-provider anthropic --model claude-opus-5-5 \
  --judge-provider openrouter --judge-model openai/gpt-4o-2024-08-06 \
  --qids-file splits/lme-s-holdout.qids --shard 0/6 --no-append-md \
  --api-key-file ~/.kijito/bench.key --api-project-prefix holdout-run1- --api-ctx mcp_text \
  --out out/holdout-opus/shard0.json --items-log out/holdout-opus/shard0.items.jsonl \
  --context-cache out/holdout-ctx/ctx-shard0.jsonl
```

The dev split (`splits/lme-s-dev.qids`) is the one used for tuning; the held-out split is scored once.
See [`splits/README.md`](splits/README.md).

### STALE, raw ingest (all 400 scenarios, protocol reader and judge)

Reader `gpt-4o-mini`, judge `gemini-3.1-flash-lite-preview` (the paper's memory-system configuration),
k = 10, one project per scenario, every turn written as one memory.

```bash
python -m kijito_bench.stale --n 0 --k 10 \
  --agent-provider openrouter --model openai/gpt-4o-mini \
  --judge-provider openrouter --judge-model google/gemini-3.1-flash-lite-preview \
  --shard 0/6 --api-key-file ~/.kijito/bench.key --api-project-prefix stale-run1- \
  --out out/stale/shard0.json --items-log out/stale/shard0.items.jsonl \
  --context-cache out/stale/ctx-shard0.jsonl
```

### STALE, agent-curated (disclosed arm, all 400 scenarios, hints on and off)

An ingest agent (`gpt-4o-mini`) reads each session and saves dated memories with
`kijito_remember_batch`. With `--hints on`, when the write reply flags older memories that a new one may
update, the agent decides which pairs are real replacements and links them with
`kijito_relate(new, old, "supersedes")`. With `--hints off` it ignores the hints. Distillations are cached
per session in `--distill-cache` and shared, so both arms store identical memories and differ only in
whether the agent acts on the hints. Probes are scored exactly as in the raw arm.

```bash
python -m kijito_bench.stale_agent --n 0 --shard 0/8 --mode curated --hints on \
  --api-key-file ~/.kijito/bench.key --project-prefix sa-on-run1- \
  --distill-cache out/distill.jsonl --out out/sa-on/shard0.json
python -m kijito_bench.stale_agent --n 0 --shard 0/8 --mode curated --hints off \
  --api-key-file ~/.kijito/bench.key --project-prefix sa-off-run1- \
  --distill-cache out/distill.jsonl --out out/sa-off/shard0.json
```

Shards of one arm can share the distill cache file. This arm has no resume: a shard that dies is rerun
with a fresh `--project-prefix` (the distill cache keeps the finished distillations).

**Rerunning with the published distillations.** The distillations our runs used are published as
[`results/stale-agent-curated/distill-cache.jsonl.gz`](results/stale-agent-curated/distill-cache.jsonl.gz):
all 20,000 sessions of the 400 scenarios, one line per session (`uid`, `si` = session index, `mems` =
the distiller's lines, kept exactly as the runs stored them, including the occasional preamble line;
300 sessions have none). Starting from it makes both arms store the same memories our runs stored and
skips the distillation cost:

```bash
gunzip -c results/stale-agent-curated/distill-cache.jsonl.gz > out/distill.jsonl
```

Hint decisions are still made live by the ingest agent, so a rerun of the hints-on arm can differ
slightly from ours.

**Disclosed Claude Opus 5.5 reader arm.** Re-scores the saved recall replies of a finished run
(`<out>.ctx.jsonl`) with another reader; nothing is written or recalled (the run only reads
`/api/version`), and the judge is unchanged:

```bash
python -m kijito_bench.stale_agent --n 0 --shard 0/8 --project-prefix unused- --distill-cache /dev/null \
  --reuse-ctx out/sa-on/shard0.json.ctx.jsonl --agent-provider anthropic --model claude-opus-5-5 \
  --api-key-file ~/.kijito/bench.key --out out/sa-on-opus/shard0.json
```

`--judge-provider` / `--judge-model` change the judge the same way.

### GroupMemBench (all 745 questions, protocol reader and judge)

Reader and judge `gpt-5`, k = 10, the official reader and judge prompts and output budgets, one project
per domain, every message written as one memory in the benchmark's own passage format. One process per
domain:

```bash
for dom in Finance Technology Healthcare Manufacturing; do
  python -m kijito_bench.groupmembench --domains "$dom" --arms kijito,bm25 \
    --agent-provider openrouter --model openai/gpt-5 --judge-provider openrouter --judge-model openai/gpt-5 \
    --api-key-file ~/.kijito/bench.key --api-project-prefix gmb-run1- \
    --context-cache "out/gmb/ctx-$dom.jsonl" --results-dir out/gmb/results \
    --out "out/gmb/$dom.json" --items-log "out/gmb/$dom.items.jsonl" > "out/gmb/$dom.log" 2>&1 &
done; wait
```

**With keyword search on.** Kijito's keyword search over encrypted content is off by default and is an
account setting: sign in at <https://kijito.ai/account>, turn on **Keyword search**, and wait for the
existing memories to be indexed. Then run the same command with `--blind` added (use a separate account,
or delete the earlier projects, so the two arms do not share memories). The harness checks the setting
with `GET /api/fts-blind/status`, records the answer in the manifest, and refuses to run if the account
is not enrolled.

## Cost and time

Measured on the published runs unless marked as an estimate (prices as listed by the providers in October 2026; the scorecard's
`cost` block reports exact token counts so cost can be recomputed). Wall times were measured against a
local copy of the service with the shard counts shown above; on a hosted account, ingest time is bounded
by the account's write rate limit.

| Arm | Reader + judge cost | Other model cost | Time |
|---|---|---|---|
| LongMemEval-S held-out (350 q), GPT-4o | about $12 | none | ingest + recall about 150 s per question; about 2.5 h with 6 shards |
| LongMemEval-S held-out, Opus 5.5 reader | about $31 (7.2M input, 0.1M output tokens at $4 / $20 per million) | none | minutes (no ingest) |
| LongMemEval-S dev (150 q), GPT-4o | about $5 | none | about 1 h with 6 shards |
| STALE raw (400 scenarios) | about $0.90 | none | about 250 s per scenario; about 4.5 h with 6 shards |
| STALE agent-curated, per arm | about $1 | distillation about $12 once, shared by both arms; hint decisions about $3 (hints on) | about 500 to 700 s per scenario; about 8 h with 8 shards |
| STALE agent-curated, Opus 5.5 reader | estimated $10 to $30 (output-heavy; at most 1,024 output tokens per call) | none | minutes (no ingest) |
| GroupMemBench (745 q), GPT-5 | about $13 per scoring pass | none | ingest about 1.5 h per 30,000-message domain (domains in parallel) |

The `usd` fields come from the harness's built-in list-price table (`kijito_bench/llm.py`); token counts
are exact, so cost can be recomputed if prices change.

## Output files

Every arm writes, for each shard:

| File | Contents |
|---|---|
| `<out>` (JSON scorecard) | `status` (`ran`, or `invalid` when any retrieval or scoring error or any cross-project leak occurred; an invalid run must not be published), `config`, `metrics` / `arms`, `cost`, `manifest`, `api_leaks`, `retrieval_timing` |
| `<out>.items.jsonl` or `--items-log` | One JSON record per question (per scenario for STALE), described below. For LongMemEval, STALE raw and GroupMemBench the first line is a configuration fingerprint used for resume |
| `--context-cache` | Each retrieval: the `kijito_recall` reply text exactly as returned, plus the project, the recalled ids and the leak verdict; also one record per finished ingest. Replayed by later runs with the same server, prefix and settings |
| `<out>.api-trace.jsonl` | Per question / scenario / domain: timings, recalled ids, leak verdicts, number of calls and retries |
| `<out>.ctx.jsonl` (agent arm) | The recall reply for every (scenario, probe); input to `--reuse-ctx` |
| `--distill-cache` (agent arm) | The ingest agent's memories per (scenario, session) |
| `--results-dir` (GroupMemBench) | The benchmark's own output schema, `results/<Domain>/<arm>__<qtype>.jsonl`, readable by its `task_synthesis/summarize_typed_eval.py`; the Kijito rows add `mcp_reply`. Not published here: those rows repeat the benchmark's question text and retrieved passages, and GroupMemBench publishes no license |

The `manifest` records the harness git commit and dirty flag, the dataset file sha256, the question list
and its sha256, the shard, and the server's identity as reported by `GET /api/version` (`server_commit`,
`server_version`), the base URL, the project prefix, the recall reply form and the number of MCP calls
and retries.

Per-question records:

- **LongMemEval**: `qid`, `arm`, `category` (question type), `ok`, `hyp` (the reader's answer), `hyp_len`,
  `answer_in_tok` / `answer_out_tok`, `judge_in_tok` / `judge_out_tok`, `in_tok` / `out_tok`, `error`,
  `elapsed_s`.
- **STALE (both arms)**: `qid` (scenario uid), `category` (T1 / T2), `pass` (per probe: `dim1`, `dim2`,
  `dim3`), `ok` (all three passed), `answers`, `reasoning` (the judge's, per probe), `ctx_tokens` and
  `n_ctx` per probe, token counts, `judge_parse_fallback`, `error`, `error_stage`.
- **GroupMemBench**: `qid` (`Domain/id`), `category` (question type), `domain`, `agent_answer`,
  `agent_reasoning`, `judge_answer`, `judge_reasoning`, `verdict` (Correct / Incorrect / Unclear), `ok`,
  token counts, `ctx_tokens`, `n_ctx`.

## How answers are scored

- **LongMemEval** (`--official-eval`): one judge call per question with the official per-type yes/no
  prompt from LongMemEval's `src/evaluation/evaluate_qa.py` (temporal questions forgive off-by-one day
  counts, knowledge-update questions accept an updated answer next to stale information, preference
  questions are graded against a rubric, question ids containing `_abs` use the abstention prompt); a
  question is correct when the judge's reply contains "yes". `accuracy` is correct / all questions;
  `task_averaged_accuracy` is the mean of the per-type accuracies; `abstention_accuracy` covers the `_abs`
  questions. A question whose retrieval failed is counted as a miss and makes the run invalid.
- **STALE**: three reader calls per scenario (the official prompts; one per probe), then one judge call
  with the official all-in-one judge prompt in JSON mode, grading the three answers against the
  scenario's old state, new state and explanation. Accuracy is computed per (conflict type, probe): six
  settings. `overall_mean_of_six` is the mean of the six (the paper's Overall); `pooled_accuracy` pools
  all probes. A judge reply that cannot be parsed is an error, not a fail.
- **GroupMemBench**: one reader call (official system prompt; the answer is the `Final:` line) and one
  judge call per question. `overall_pooled` follows the paper's rule: verdicts are read from the judge's
  `Final:` line, `Unclear` verdicts are excluded from the denominator, questions are pooled across
  domains. The repository summarizer's stricter rule is reported as `overall_summarizer_rule`. When all
  745 questions are scored, `paper_pool_estimate` reweights to the paper's 715-question pool.

## Integrity checks built into every run

- **Scope canary.** Before the first live retrieval, two fresh projects share one distinctive phrase; a
  project-scoped recall must return only its own project's memory, and an account-wide recall must find
  both. If not, the run is refused.
- **Leak check.** Every recalled memory that the question did not write is looked up with `kijito_get`;
  a memory from another project marks the run invalid.
- **Fresh projects.** A project that already holds memories is refused, so a rerun never doubles a
  haystack.
- **Errors are not scores.** Failed calls are retried only when the service says the call was not
  carried out (rate limit, overload, 429/5xx); a timed-out write is never resent. Any other failure is a
  retrieval error: never cached, never answered, and the run is marked invalid until a resume retries it.

## Datasets and licenses

The harness code is licensed under Apache-2.0 (see [`LICENSE`](LICENSE)). The datasets are not included and keep their
own terms:

| Dataset | Source | License |
|---|---|---|
| LongMemEval | <https://github.com/xiaowu0162/LongMemEval>, Hugging Face `xiaowu0162/longmemeval-cleaned` | MIT |
| STALE | <https://github.com/icedreamc/STALE>, Hugging Face `STALEproj/STALE` | CC BY 4.0 (its distractor sessions come from LongMemEval, MIT) |
| GroupMemBench | <https://github.com/UCSB-NLP-Chang/GroupMemBench> | No license file is published in the repository (checked at commit `e2682e0`); use it under the authors' terms |

One derived file is included: `results/stale-agent-curated/distill-cache.jsonl.gz` is an adaptation of
STALE (CC BY 4.0, by the STALE authors). It holds `gpt-4o-mini` summaries of STALE's chat sessions, made
by this repository's ingest agent; the summaries are not STALE's own text. It is shared under CC BY 4.0.

## Tests

```bash
pytest -q
```

The tests run offline against an in-memory fake of the MCP endpoint and fake models. They cover the client
(batching, retries, reply parsing, the cap refusal), every arm's ingest and recall calls, prompts, judge
parsing, metrics, resume and caching, and a scan of every file in the repository for secrets and
machine-specific paths.
