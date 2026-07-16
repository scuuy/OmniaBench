# OmniaBench evaluation framework

This directory contains the model runner, four-route orchestrator, rubric/checklist evaluator, and score aggregation utilities used by OmniaBench.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r evaluation/requirements.txt
cp evaluation/configs/profiles.example.json evaluation/configs/profiles.json
cp evaluation/.env.example evaluation/.env
```

`profiles.json` only stores *environment variable names* (e.g. `OMNIABENCH_AGENT_API_KEY`), never secrets — the actual keys and base URLs go in `evaluation/.env`. `run_eval.py` loads `.env` automatically at startup via `python-dotenv`. You only need to fill in the variables referenced by the profile you actually use:

```bash
# --- Agent model (the model under test) ---
OMNIABENCH_AGENT_API_KEY=sk-...
OMNIABENCH_AGENT_BASE_URL=https://api.example.com/v1

# --- User-simulator model (plays the human side of the conversation) ---
OMNIABENCH_USER_API_KEY=sk-...
OMNIABENCH_USER_BASE_URL=https://api.example.com/v1

# --- Rubric judge model (scores agent transcripts against the rubric) ---
OMNIABENCH_JUDGE_API_KEY=sk-...
OMNIABENCH_JUDGE_BASE_URL=https://api.example.com/v1
```

Credentials must never be committed — `.env` is already covered by `.gitignore`.

## Data layout

The 644-task challenge subset is hosted on [Hugging Face](https://huggingface.co/datasets/scuuy666/OmniaBench)
(it's not committed to this repo, since the files are tens of MB each). You don't need to download it
yourself: the first time `orchestrate_eval.py` runs, it checks whether each route's data file is present
under `evaluation/data/routes/`, and if any are missing it calls `huggingface_hub.snapshot_download()` to
pull the whole dataset repo into `evaluation/data/` automatically, landing the four files at:

```text
evaluation/data/routes/route1.json
evaluation/data/routes/route2.json
evaluation/data/routes/route3.json
evaluation/data/routes/route4.json
```

This requires the `huggingface_hub` package (already in `requirements.txt`) and outbound network access.
If you'd rather manage the data yourself:

- `--no-download`: disable the automatic download entirely (e.g. offline/air-gapped environments). Any
  route still missing a data file will be reported as `missing_data_file` in the run summary instead of
  being downloaded.
- `--data-override route_id=/path/to/file.json`: point a specific route at a local file instead (see
  Advanced usage below). Routes with an override never trigger a download, since they already resolve.
- Manual download, if you want the files without running an eval first:

  ```bash
  python -c "
  from huggingface_hub import snapshot_download
  snapshot_download(repo_id='scuuy666/OmniaBench', repo_type='dataset', local_dir='evaluation/data')
  "
  ```

Route 1 also requires a filesystem sandbox bundle, which *is* committed to this repo (it's small — a few hundred KB) at `evaluation/runtime_assets/fs_bundle/`. No extra download step is needed for it.

## Run

Run all four routes:

```bash
python evaluation/scripts/orchestrate_eval.py \
  --profile openai_compatible \
  --routes route1 route2 route3 route4 \
  --pass-k 1 \
  --max-task-workers 8
```

Run a subset of routes, optionally restricted to a global_id range per route:

```bash
python evaluation/scripts/orchestrate_eval.py \
  --profile openai_compatible \
  --routes route2 route3 \
  --route-global-id-range route2=1-10 \
  --route-global-id-range route3=all
```

Test one or more specific tasks by `global_id` directly, without needing to know
which route owns which id range — the route is auto-detected from
`configs/routes.json`:

```bash
python evaluation/scripts/orchestrate_eval.py \
  --profile openai_compatible \
  --global-id 1000023 3000005
```

To score an existing result file without rerunning the agent:

```bash
python evaluation/scripts/run_eval.py \
  --execution-mode eval_only \
  --task-items-path evaluation/data/routes/route1.json \
  --result-file-path /path/to/result.json
```

Outputs are written under `evaluation/results/` and are intentionally ignored by Git.

## Advanced usage

- `--resume` (passed after `--`, e.g. `... -- --resume`): automatically picks up the latest matching result file in `results/<profile>-*/` and skips tasks that already completed, retrying only failed/`INFRA_ERROR` runs. Add `--resume-keep-failed` to keep those failed runs as-is instead of retrying them.
- `--incremental-dir <path>`: put the incremental shard files on a faster disk instead of next to the output directory. Useful for long runs where I/O contention matters.
- `--num-shards <n>` (default 16): number of incremental shard files runs are checkpointed into. Higher shard counts reduce lock contention under high `--max-task-workers`.
- `--pass-k <k>`: run each task `k` times and compute pass@k in addition to pass@1.
- `--lang-filter cn|en|all`: restrict tasks to a language subset, intersected with any `global_id` range filter.
- `--data-override route_id=/path/to/file.json`: point a specific route at a local data file instead of the path in `configs/routes.json`. Repeatable.
- `--no-download`: disable the automatic Hugging Face download when a route's data file is missing (see Data layout above).

## Interpreting results

Each run produces a directory `evaluation/results/<profile>-<run_tag>/` containing:

```text
<profile>-<run_tag>/
├── route1/                 # per-route raw + aggregated result JSON
├── route2/
├── route3/
├── route4/
├── incremental_shards/     # checkpoint shards written during the run (safe to delete after completion)
├── state_diffs/            # merged environment state diffs across all routes
└── combined-<profile>-<user_model>-<run_tag>.runs.jsonl   # all routes' .runs.jsonl merged
```

A top-level `evaluation/results/route_summary-<profile>-<run_tag>.json` aggregates all routes. You can also regenerate the per-route table from any aggregated result file with:

```bash
python evaluation/scripts/route_scores.py --result-file-path /path/to/route1_result.json
```

Key fields in the summary:

| Field | Meaning |
|---|---|
| `task_count` | Number of tasks evaluated in the route |
| `scored_count` | Number of tasks that produced a usable score (may be less than `task_count` if some runs errored) |
| `pass_at_1` | Fraction of tasks whose sample-1 combined score equals exactly 1.0 |
| `score` | Mean combined score across all scored tasks |
| `score_source` | Which scorer produced the combined score, in priority order: `verifier` (route 3, exact-match/checker-based) → `rubric` (routes 1/2/4, LLM judge against a rubric) → `total_reward` (fallback) |
| `pass_at_k` / `k` | Only present when `--pass-k` > 1; fraction of tasks with at least one passing sample out of `k` |

`route_scores.py` also prints an equal-weighted overall row (arithmetic mean of `pass_at_1`/`score` across the four routes, not weighted by each route's task count).

