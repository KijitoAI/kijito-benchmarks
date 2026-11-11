# LongMemEval-S question splits

`longmemeval_s_cleaned.json` (Hugging Face `xiaowu0162/longmemeval-cleaned`, snapshot
`98d7416c24c778c2fee6e6f3006e7a073259d48f`, file sha256
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`) has 500 questions, split into:

| File | Questions | Use |
|---|---|---|
| `lme-s-dev.qids` | 150 | development and tuning |
| `lme-s-holdout.qids` | 350 | held out: scored once at the end, not used for tuning |

The split is stratified by `question_type` over base question ids (a question and its `_abs` abstention
twin are always in the same split), about 30% dev and 70% held out, with `random.Random(20261024)`, frozen on 2026-10-24.

One question id per line; lines starting with `#` are comments. Pass a file with `--qids-file`; the
question order follows the file, and `--shard I/N` takes every N-th question starting at I.

The question ids and their order are identical to the files used for the published runs. Only the
comment header was reworded, so the whole-file sha256 that a run's manifest records
(`manifest.qids_file.sha256_12`) differs from the published scorecards:

| File | Published runs (`sha256_12`) | This repository (`sha256_12`) | sha256_12 of the id list (one id per line, `\n`-terminated) |
|---|---|---|---|
| `lme-s-dev.qids` | `7ca8629088fc` | `a0186c00f533` | `f42f2a49aad9` |
| `lme-s-holdout.qids` | `9bbaa9bcae6d` | `dbfdd783d230` | `7227fcc9ad85` |
