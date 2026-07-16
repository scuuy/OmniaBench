<div align="center">

# OmniaBench

**Benchmarking General AI Agents Across Diverse Scenarios**

[![Project Page](https://img.shields.io/badge/🌐-Project%20Page-1a73e8)](https://scuuy.github.io/OmniaBench/)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-ffcc4d)](https://huggingface.co/datasets/scuuy666/OmniaBench)
[![Citation](https://img.shields.io/badge/📚-Citation-6f42c1)](#citation)

</div>

<p align="center">
  <img src="website/public/figures/overview.webp" width="820" alt="Conceptual overview of OmniaBench: data sources, taxonomy, executable environments, and evaluation protocol">
</p>

OmniaBench is a broad, diagnostic benchmark for evaluating general AI agents. Scenario knowledge is
distilled from app stores, product documents, industry resources, and web retrieval into a hierarchical
taxonomy spanning **ToC, ToB, and ToE** with **90 level-1 and 354 level-2 domains**. On top of this
taxonomy we build executable environments and synthesize tasks through four complementary construction
routes, then score every trajectory along a **ten-dimensional capability taxonomy** and **eight atomic
difficulty factors**.

The full collection contains **1,431 tasks**; a **644-task challenging subset** is released for cost-efficient,
contamination-resistant leaderboard evaluation. Even frontier models find it hard: Claude-Sonnet-5 and
GPT-5.6-Sol top the leaderboard with Overall Pass@1 of only **58.54%** and **57.14%**, respectively.

## Highlights

- **Broad scenario coverage** — 90 level-1 / 354 level-2 domains grounded in real app stores, PRDs, and industry taxonomies across consumer, business, and employee-facing settings.
- **Four complementary construction routes** — DAG (multi-turn, stateful tool-chain execution), DAG-S (single-turn tasks derived from DAG via query refinement), Solver (selection / scheduling / allocation / optimization), and Program (procedural reasoning with branching, iteration, and execution debugging).
- **Diagnostic evaluation, not just pass/fail** — a ten-dimensional capability taxonomy (Task Understanding, Information Gathering, Planning & Decision Making, State Management, Tool Use, Code & Programmatic Operations, Data Analysis, Office & Document Handling, Interactive Collaboration, Reliability & Safety) and eight atomic difficulty factors enable fine-grained analysis beyond a single aggregate score.
- **Rubric- and code-based scoring** — DAG / DAG-S / Solver tasks are judged against weighted rubric checklists; Program tasks use binary `VerifyCode` verification.
- **Reproducible evaluation harness** — a model-agnostic runner, four-route orchestrator, and scoring pipeline are provided under [`evaluation/`](evaluation/), portable across OpenAI-compatible and Anthropic-native providers.

## Leaderboard (challenging subset, Pass@1)

| Rank | Model | Access | DAG | Solver | Program | DAG-S | Overall |
|---|---|---|---|---|---|---|---|
| 1 | Claude-Sonnet-5 | Closed | 57.34 | 9.97 | 56.67 | 63.33 | **58.54** |
| 2 | GPT-5.6-Sol | Closed | 55.37 | 7.52 | 65.00 | 50.00 | 57.14 |
| 3 | GLM-5.2 | Open | 54.80 | 6.83 | 26.67 | 60.00 | 56.83 |
| 4 | GPT-5.5 | Closed | 54.80 | 7.27 | 38.33 | 60.00 | 56.52 |
| 5 | DeepSeek-V4-Pro | Open | 52.54 | 6.23 | 36.67 | 53.33 | 54.50 |
| 6 | Claude-Opus-4.7 | Closed | 53.39 | 7.60 | 43.33 | 63.33 | 54.19 |
| 7 | Kimi-K2.6 | Open | 49.72 | 7.09 | 45.00 | 63.33 | 52.33 |
| 8 | Qwen3.7-Max | Closed | 48.59 | 6.49 | 51.67 | 66.67 | 49.69 |

The paper and its arXiv link will be added after the official release. See the
[project page](https://scuuy.github.io/OmniaBench/) for the interactive leaderboard and analyses.

## Data construction

<p align="center">
  <img src="website/public/figures/pipeline.webp" width="820" alt="OmniaBench data construction pipeline: taxonomy, environment synthesis, and four task construction routes">
</p>

Domain knowledge collected by a web agent is calibrated through human refinement, then translated into
instantiable Python environments with entities, states, tools, and initialization configs. Tasks are
synthesized on top of these environments through four routes:

| Route | Description | Challenging subset |
|---|---|---|
| **DAG** *(anchor)* | Multi-turn, stateful interaction over sampled tool-dependency chains | 354 |
| **DAG-S** *(derived)* | Single-turn tasks obtained by query-refining DAG tasks | 200 |
| **Solver** | Selection, scheduling, allocation, and optimization scenarios | 60 |
| **Program** | Procedural reasoning with branching, iteration, and task-program synthesis | 30 |

The challenging subset (644 tasks) is a fixed, curated slice of the full 1,431-task collection, selected to
reduce evaluation cost and mitigate contamination risk after public release while preserving domain coverage.

## Repository layout

```text
evaluation/  Reproducible evaluation runners, route orchestration, and scoring
website/     Next.js project page deployed to GitHub Pages
```

See [evaluation/README.md](evaluation/README.md) for evaluation setup. The 644-task route files are hosted
on [Hugging Face](https://huggingface.co/datasets/scuuy666/OmniaBench) (not committed to this repo); the
filesystem sandbox assets needed by Route 1 are committed under `evaluation/runtime_assets/fs_bundle/`.

## Getting started

```bash
git clone https://github.com/scuuy/OmniaBench.git
cd OmniaBench

python -m venv .venv && source .venv/bin/activate
pip install -r evaluation/requirements.txt
cp evaluation/configs/profiles.example.json evaluation/configs/profiles.json
cp evaluation/.env.example evaluation/.env   # fill in your API keys, never commit this file

python evaluation/scripts/orchestrate_eval.py \
  --profile openai_compatible \
  --routes route1 route2 route3 route4 \
  --pass-k 1 --max-task-workers 8
```

Full setup, data download, and advanced usage are documented in [evaluation/README.md](evaluation/README.md).

## Website development

```bash
cd website
npm ci
npm run dev
```

Pushes to `main` automatically build and deploy the static site through GitHub Pages.

## Team

OmniaBench is developed by the Huawei Cloud Post-Training Team and PKU DCAI Team, in collaboration with
Renmin University of China, Beijing Institute of Technology, and Tsinghua University. The full author list
and acknowledgments will be added with the arXiv release.

## Citation

```bibtex
Coming soon.
```

## License

The source code in `evaluation/` and `website/` is released under the
[Apache License 2.0](LICENSE).

Benchmark figures, institutional marks, and third-party logos are
not covered by the Apache-2.0 code license. Their respective copyrights and
trademarks remain with their owners. The OmniaBench dataset will be released
separately with its own license and terms of use.
