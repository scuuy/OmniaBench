# OmniaBench

**Benchmarking General AI Agents across Diverse Tasks and Environments**

OmniaBench is a broad, diagnostic benchmark for general AI agents. The challenging split contains 644 tasks across four construction routes (354 / 60 / 30 / 200), drawn from a full collection of 1,431 tasks.

[Project page](https://scuuy.github.io/OmniaBench/) · [Paper](./omniabench.pdf) · Dataset (coming soon)

## Repository layout

```text
evaluation/  Reproducible evaluation runners, route orchestration, and scoring
website/     Next.js project page deployed to GitHub Pages
```

See [evaluation/README.md](evaluation/README.md) for evaluation setup. The dataset and filesystem assets are deliberately not committed yet; the project page keeps the dataset link as a release placeholder until the Hugging Face upload is ready.

## Website development

```bash
cd website
npm ci
npm run dev
```

Pushes to `main` automatically build and deploy the static site through GitHub Pages.

