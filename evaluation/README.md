# OmniaBench evaluation framework

This directory contains the model runner, four-route orchestrator, rubric/checklist evaluator, and score aggregation utilities used by OmniaBench.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r evaluation/requirements.txt
cp evaluation/configs/profiles.example.json evaluation/configs/profiles.json
```

Set the API keys and compatible API base URLs referenced by `profiles.json` as environment variables. Credentials must never be committed.

## Data layout

The public dataset will be linked separately. After downloading it, place the four route files at:

```text
evaluation/data/routes/route1.json
evaluation/data/routes/route2.json
evaluation/data/routes/route3.json
evaluation/data/routes/route4.json
```

Route 1 also requires the released filesystem sandbox bundle at `evaluation/runtime_assets/fs_bundle/`.

## Run

```bash
python evaluation/scripts/orchestrate_eval.py \
  --profile openai_compatible \
  --routes route1 route2 route3 route4 \
  --pass-k 1 \
  --max-task-workers 8
```

To score an existing result file without rerunning the agent:

```bash
python evaluation/scripts/run_eval.py \
  --execution-mode eval_only \
  --task-items-path evaluation/data/routes/route1.json \
  --result-file-path /path/to/result.json
```

Outputs are written under `evaluation/results/` and are intentionally ignored by Git.

