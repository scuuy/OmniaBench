---
pretty_name: OmniaBench
license: other
license_name: omniabench-research-only
license_link: https://github.com/scuuy/OmniaBench/blob/main/LICENSE
task_categories:
- text-generation
- other
language:
- en
- zh
tags:
- agent
- benchmark
- tool-use
- llm-evaluation
size_categories:
- 1K<n<10K
---

# Dataset Card for OmniaBench

- **Paper:** OmniaBench: Benchmarking General AI Agents Across Diverse Scenarios (coming soon)
- **Homepage:** https://scuuy.github.io/OmniaBench/
- **Repository:** https://github.com/scuuy/OmniaBench
- **Point of Contact:** see the paper / GitHub repository

## Dataset Summary

OmniaBench is a broad, diagnostic benchmark for evaluating general AI agents. Scenario knowledge is
distilled from app stores, product documents, industry resources, and web retrieval into a hierarchical
taxonomy spanning ToC, ToB, and ToE with 90 level-1 and 354 level-2 domains. On top of this taxonomy,
executable environments are built and tasks are synthesized through four complementary construction
routes, then scored along a ten-dimensional capability taxonomy and eight atomic difficulty factors.

This repository hosts the **644-task challenging subset**, released for cost-efficient,
contamination-resistant leaderboard evaluation, as four route files (`route1.json`–`route4.json`).
It is the companion dataset to the [OmniaBench GitHub repository](https://github.com/scuuy/OmniaBench),
which provides the evaluation harness that consumes these files directly.

## Dataset Structure

Each route file is a JSON array of task items. The four routes correspond to four different task
construction methods and are evaluated with different scorers:

| Route file | Route name | Tasks | Turn mode | Scorer |
|---|---|---|---|---|
| `route1.json` | DAG (main body) | 354 | multi-turn | rubric |
| `route2.json` | Solver | 60 | single-turn | rubric |
| `route3.json` | Program | 30 | single-turn | verifier (executable check code) |
| `route4.json` | DAG-S | 200 | single-turn | rubric |

### Data Fields

Fields are route-specific since each construction route captures different information. Common fields
across most routes:

- `global_id` / `task_id` / `env_id`: task and environment identifiers
- `task`: the natural-language task instruction given to the agent
- `init_config`: initial state configuration for the task's environment
- `rubrics`: the weighted rubric checklist (or, for route3, see `verifier_code` below) used to score the agent's trajectory

Route-specific fields:

- **route1 / route2**: `env_item` (full environment definition: tools, class code, constraints, introduction), `persona` / `persona_card_md` / `persona_idx` (user-simulator persona for the multi-turn conversation), `env_class_name`, `lang`
- **route1**: also includes `fs_inputs` / `fs_runtime_bootstrap` for tasks requiring a filesystem sandbox (see `runtime_assets/fs_bundle/` in the GitHub repo)
- **route3**: `candidate_tools`, `env_class_code`, `constraints_rules`, `environment_introduction`, `ground_truth_answer`, `verifier_code` (executable Python check function used for binary pass/fail scoring instead of rubric judging)
- **route4**: `candidate_tools`, `env_class_code`, `env_class_name`, `persona` / `persona_card_md` / `persona_idx`, `lang`

Some internal bookkeeping fields used only during dataset construction (e.g. domain taxonomy labels,
construction-pipeline debug metadata) have been stripped from this release and are not required by the
evaluation harness.

### Data Splits

This repository contains a single, fixed 644-task challenging subset (not split into train/val/test):
354 + 60 + 30 + 200 tasks across the four routes above. This split is used as-is for leaderboard
evaluation; see the paper for details on how it was selected from the full 1,431-task collection.

## Usage

See [evaluation/README.md](https://github.com/scuuy/OmniaBench/blob/main/evaluation/README.md) in the
GitHub repository for how to download this dataset and run model evaluation against it with the provided
evaluation harness.

## Licensing Information

The dataset license is still being finalized and will be published alongside the arXiv paper. Until then,
treat this dataset as research-use only; do not redistribute. Check back here or the
[GitHub repository](https://github.com/scuuy/OmniaBench) for the finalized terms.

## Citation

```bibtex
Coming soon.
```
