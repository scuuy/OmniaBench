"use client";

import { useEffect, useRef, useState } from "react";
import radarDataSource from "./data/radar_22models_micro.json";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const asset = (path: string) => `${basePath}${path}`;

const links = {
  paper: asset("/omniabench.pdf"), // Replace with the public arXiv URL when available.
  github: "https://github.com/scuuy/OmniaBench",
  dataset: "#release", // Replace with the public Hugging Face dataset URL.
};

const providerLogos: Record<string, string> = {
  OpenAI: asset("/logos/openai.png"),
  Anthropic: asset("/logos/claude.png"),
  Kimi: asset("/logos/kimi.png"),
  Qwen: asset("/logos/qwen.png"),
  GLM: asset("/logos/glm.png"),
  DeepSeek: asset("/logos/deepseek.png"),
  Gemini: asset("/logos/gemini.png"),
  Doubao: asset("/logos/doubao.png"),
};

type RadarData = {
  caps: string[];
  models: string[];
  capability: Record<string, Record<string, number>>;
  split: Record<string, { tob: number; toc: number; toe: number }>;
  leaderboard: Record<string, {
    R1: number;
    R2: number;
    R3: number;
    R4: number;
    Overall: number;
    passed: number;
    total: number;
    UsrTurns: number;
    ToolSteps: number;
    turns_steps_source: string;
    is_new: boolean;
  }>;
};

type RadarView = "capability" | "split" | "routes";

const radarData = radarDataSource as RadarData;
const paperRadarModels = ["GPT-5.5", "Claude-Opus-4.7", "GLM-5.2", "DeepSeek-V4-Pro", "Qwen3.6-35B-A3B"];
const radarPalette = [
  "#d86f55", "#318b88", "#7d9c43", "#5878b8", "#c46f9b", "#9a6cc1", "#d9953f", "#498cba", "#8392a6", "#5f9b72",
  "#b36d5e", "#4f76a5", "#af8543", "#4d9b9a", "#8e70a6", "#c15f72", "#6e914d", "#8c7356", "#497da0", "#9b6286",
  "#3f8f6d", "#b06d36",
];
const paperRadarColors: Record<string, string> = {
  "GPT-5.5": "#F0876A",
  "Claude-Opus-4.7": "#3FB6A6",
  "GLM-5.2": "#9AC75B",
  "DeepSeek-V4-Pro": "#8189D4",
  "Qwen3.6-35B-A3B": "#E389B6",
};

function providerForModel(model: string) {
  if (model.startsWith("GPT")) return "OpenAI";
  if (model.startsWith("Claude")) return "Anthropic";
  if (model.startsWith("Kimi")) return "Kimi";
  if (model.startsWith("Qwen")) return "Qwen";
  if (model.startsWith("GLM")) return "GLM";
  if (model.startsWith("DeepSeek")) return "DeepSeek";
  if (model.startsWith("Gemini")) return "Gemini";
  if (model.startsWith("Doubao")) return "Doubao";
  return "OpenAI";
}

function colorForModel(model: string) {
  if (paperRadarColors[model]) return paperRadarColors[model];
  const index = radarData.models.indexOf(model);
  return radarPalette[Math.max(0, index) % radarPalette.length];
}

function accessForModel(model: string) {
  if (model.startsWith("GPT") || model.startsWith("Claude") || model.startsWith("Gemini") || model.startsWith("Doubao")) return "Proprietary";
  if (model === "Qwen3.7-Max") return "Proprietary";
  return "Open";
}

function effortForModel(model: string) {
  if (model.startsWith("GPT")) return "high";
  if (model.startsWith("Claude") || model.startsWith("GLM") || model.startsWith("DeepSeek")) return "max";
  return undefined;
}

const leaderboard = radarData.models.slice(0, 8).map((model, index) => {
  const scores = radarData.leaderboard[model];
  return {
    rank: index + 1,
    provider: providerForModel(model),
    model,
    access: accessForModel(model),
    effort: effortForModel(model),
    dag: scores.R1.toFixed(2),
    solver: scores.R2.toFixed(2),
    program: scores.R3.toFixed(2),
    dags: scores.R4.toFixed(2),
    overall: scores.Overall.toFixed(2),
  };
});

const authors = [
  "Chengyu Shen*", "Yujie Fu*", "Gangtao Xin*", "Yanheng Hou", "Wenlong Fei",
  "Guojie Zhu", "Jiawei Li", "Hongcheng Gao", "Runming He", "Zhen Hao Wong",
  "Meiyi Qiang", "Hao Liang", "Zhao Cao", "Hao Jiang†", "Chong Chen‡", "Wentao Zhang‡",
];

const sections = [
  { id: "leaderboard", label: "Leaderboard", short: "Results" },
  { id: "overview", label: "Overview", short: "Intro" },
  { id: "method", label: "Data Construction", short: "Method" },
  { id: "diagnosis", label: "Capability Diagnosis", short: "Analysis" },
  { id: "findings", label: "Key Findings", short: "Findings" },
  { id: "release", label: "Open Release", short: "Release" },
];

function BrandIcon({ src, label }: { src: string; label: string }) {
  return (
    <span className="brand-icon" aria-hidden="true">
      <span className="brand-fallback">{label.slice(0, 1)}</span>
      <img src={src} alt="" onError={(event) => {
        event.currentTarget.style.display = "none";
        const fallback = event.currentTarget.previousElementSibling as HTMLElement | null;
        if (fallback) fallback.style.display = "grid";
      }} />
    </span>
  );
}

function PodiumIcon() {
  return <span className="podium-icon" aria-hidden="true"><i /><i /><i /></span>;
}

function HuggingFaceIcon() {
  return (
    <span className="huggingface-icon" aria-hidden="true">
      <img src={asset("/logos/huggingface.png")} alt="" />
    </span>
  );
}

function SectionNavigation({ active }: { active: string }) {
  return (
    <>
      <aside className="section-rail" aria-label="Page sections">
        <div className="rail-line" />
        {sections.map((section, index) => (
          <a key={section.id} href={`#${section.id}`} className={active === section.id ? "active" : ""} aria-current={active === section.id ? "location" : undefined}>
            <span className="rail-dot" /><small>{String(index + 1).padStart(2, "0")}</small><b>{section.short}</b>
          </a>
        ))}
      </aside>
      <nav className="mobile-section-nav" aria-label="Quick section navigation">
        <div>
          {sections.map((section) => (
            <a key={section.id} href={`#${section.id}`} className={active === section.id ? "active" : ""} aria-current={active === section.id ? "location" : undefined}>{section.short}</a>
          ))}
        </div>
      </nav>
    </>
  );
}

function RadarExplorer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedModels, setSelectedModels] = useState(paperRadarModels);
  const [view, setView] = useState<RadarView>("capability");

  const viewConfig = view === "capability"
    ? {
        title: "Ten capability dimensions",
        labels: radarData.caps,
        shortLabels: ["Task Understanding", "Info Gathering", "Planning & Decision", "Tool Use", "State Management", "Office & Docs", "Data Analysis", "Code & Program", "Interactive Collab.", "Reliability & Safety"],
        values: (model: string) => radarData.caps.map((capability) => radarData.capability[model][capability]),
        minScore: 38,
        maxScore: 61,
        gridLevels: [40, 45, 50, 55, 60],
      }
    : view === "split"
      ? {
          title: "Scenario split performance",
          labels: ["ToB", "ToC", "ToE"],
          shortLabels: ["ToB", "ToC", "ToE"],
          values: (model: string) => [radarData.split[model].tob, radarData.split[model].toc, radarData.split[model].toe],
          minScore: 0,
          maxScore: 70,
          gridLevels: [20, 40, 60, 70],
        }
      : {
          title: "Route-level Pass@1",
          labels: ["DAG", "Solver", "Program", "DAG-S", "Overall"],
          shortLabels: ["DAG", "Solver", "Program", "DAG-S", "Overall"],
          values: (model: string) => {
            const scores = radarData.leaderboard[model];
            return [scores.R1, scores.R2, scores.R3, scores.R4, scores.Overall];
          },
          minScore: 0,
          maxScore: 70,
          gridLevels: [20, 40, 60, 70],
        };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const container = canvas.parentElement;
    if (!container) return;

    function draw() {
      if (!canvas || !container) return;
      const width = Math.max(300, container.clientWidth);
      const height = width < 560 ? Math.max(410, width * 1.08) : Math.min(650, Math.max(500, width * .78));
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      const context = canvas.getContext("2d");
      if (!context) return;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);

      const labels = viewConfig.shortLabels;
      const count = labels.length;
      const centerX = width / 2;
      const centerY = height / 2 + (count > 5 ? 10 : 6);
      const radius = Math.min(width * (count > 5 ? .31 : .33), height * (count > 5 ? .36 : .38));
      const { minScore, maxScore, gridLevels } = viewConfig;
      const scoreToDistance = (score: number) => radius * Math.max(0, Math.min(1, (score - minScore) / (maxScore - minScore)));
      const angleFor = (index: number) => -Math.PI / 2 + index * Math.PI * 2 / count;
      const pointFor = (index: number, distance: number) => ({
        x: centerX + Math.cos(angleFor(index)) * distance,
        y: centerY + Math.sin(angleFor(index)) * distance,
      });

      context.lineJoin = "round";
      gridLevels.forEach((ring) => {
        context.beginPath();
        labels.forEach((_, index) => {
          const point = pointFor(index, scoreToDistance(ring));
          if (index === 0) context.moveTo(point.x, point.y); else context.lineTo(point.x, point.y);
        });
        context.closePath();
        context.strokeStyle = ring === gridLevels[gridLevels.length - 1] ? "#adbfcd" : "#d7e2ea";
        context.lineWidth = ring === gridLevels[gridLevels.length - 1] ? 1.2 : 1;
        context.stroke();
        const labelPoint = pointFor(0, scoreToDistance(ring));
        context.fillStyle = "#8a9ba9";
        context.font = "10px SFMono-Regular, Consolas, monospace";
        context.textAlign = "left";
        context.fillText(String(ring), labelPoint.x + 5, labelPoint.y + 3);
      });

      labels.forEach((label, index) => {
        const outer = pointFor(index, radius);
        context.beginPath();
        context.moveTo(centerX, centerY);
        context.lineTo(outer.x, outer.y);
        context.strokeStyle = "#d1dde6";
        context.lineWidth = 1;
        context.stroke();

        const labelPoint = pointFor(index, radius + (count > 5 ? 27 : 33));
        const cosine = Math.cos(angleFor(index));
        context.textAlign = Math.abs(cosine) < .2 ? "center" : cosine > 0 ? "left" : "right";
        context.textBaseline = "middle";
        context.fillStyle = "#425b70";
        context.font = `${width < 500 ? 10 : 12}px Inter, -apple-system, sans-serif`;
        context.fillText(label, labelPoint.x, labelPoint.y);
      });

      selectedModels.forEach((model) => {
        const values = viewConfig.values(model);
        const color = colorForModel(model);
        context.beginPath();
        values.forEach((value, index) => {
          const point = pointFor(index, scoreToDistance(value));
          if (index === 0) context.moveTo(point.x, point.y); else context.lineTo(point.x, point.y);
        });
        context.closePath();
        context.save();
        context.globalAlpha = .09;
        context.fillStyle = color;
        context.fill();
        context.restore();
        context.strokeStyle = color;
        context.lineWidth = 2.2;
        context.stroke();

        values.forEach((value, index) => {
          const point = pointFor(index, scoreToDistance(value));
          context.beginPath();
          context.arc(point.x, point.y, 2.8, 0, Math.PI * 2);
          context.fillStyle = color;
          context.fill();
        });
      });
    }

    draw();
    const resizeObserver = new ResizeObserver(draw);
    resizeObserver.observe(container);
    return () => resizeObserver.disconnect();
  }, [selectedModels, view, viewConfig]);

  function toggleModel(model: string) {
    setSelectedModels((current) => {
      if (current.includes(model)) return current.length === 1 ? current : current.filter((item) => item !== model);
      if (current.length >= 6) return current;
      return [...current, model];
    });
  }

  return (
    <section className="radar-explorer" aria-labelledby="radar-title">
      <div className="radar-header">
        <div>
          <span className="radar-kicker">Interactive analysis</span>
          <h3 id="radar-title">Compare capability profiles</h3>
          <p>Choose up to six models. The default selection matches the five models shown in the paper.</p>
        </div>
        <div className="radar-tabs" aria-label="Radar data view">
          {([
            ["capability", "Capabilities"],
            ["split", "Scenario splits"],
            ["routes", "Routes"],
          ] as [RadarView, string][]).map(([id, label]) => (
            <button key={id} type="button" className={view === id ? "active" : ""} onClick={() => setView(id)}>{label}</button>
          ))}
        </div>
      </div>

      <div className="radar-layout">
        <div className="radar-chart">
          <div className="radar-chart-title"><span>{viewConfig.title}</span><small>Pass@1 · 644 challenging tasks</small></div>
          <canvas ref={canvasRef} role="img" aria-label={`${viewConfig.title} radar chart comparing ${selectedModels.join(", ")}`} />
          <div className="radar-legend" aria-label="Selected model colors">
            {selectedModels.map((model) => <span key={model}><i style={{ backgroundColor: colorForModel(model) }} />{model}</span>)}
          </div>
        </div>

        <div className="model-picker">
          <div className="model-picker-head"><div><b>Models</b><small>{selectedModels.length} / 6 selected</small></div><button type="button" onClick={() => setSelectedModels(paperRadarModels)}>Paper selection</button></div>
          <div className="model-options">
            {radarData.models.map((model) => {
              const provider = providerForModel(model);
              const checked = selectedModels.includes(model);
              const disabled = !checked && selectedModels.length >= 6;
              return (
                <label key={model} className={`${checked ? "checked" : ""} ${disabled ? "disabled" : ""}`}>
                  <input type="checkbox" checked={checked} disabled={disabled} onChange={() => toggleModel(model)} />
                  <i className="model-color" style={{ backgroundColor: colorForModel(model) }} />
                  <BrandIcon src={providerLogos[provider]} label={provider} />
                  <span><b>{model}</b><small>{radarData.leaderboard[model].Overall.toFixed(2)} Overall</small></span>
                </label>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const [activeSection, setActiveSection] = useState("leaderboard");

  useEffect(() => {
    const elements = sections.map(({ id }) => document.getElementById(id)).filter(Boolean) as HTMLElement[];
    const revealElements = Array.from(document.querySelectorAll<HTMLElement>("[data-reveal]"));

    const sectionObserver = new IntersectionObserver((entries) => {
      const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio);
      if (visible[0]?.target.id) setActiveSection(visible[0].target.id);
    }, { rootMargin: "-22% 0px -58% 0px", threshold: [0, 0.12, 0.35] });

    const revealObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08 });

    elements.forEach((element) => sectionObserver.observe(element));
    revealElements.forEach((element) => revealObserver.observe(element));
    return () => { sectionObserver.disconnect(); revealObserver.disconnect(); };
  }, []);

  return (
    <main id="top">
      <div className="ambient ambient-one" aria-hidden="true" />
      <div className="ambient ambient-two" aria-hidden="true" />

      <nav className="site-nav">
        <div className="wide-container nav-inner">
          <a className="brand" href="#top">OmniaBench</a>
          <div className="nav-links">
            <a href="#leaderboard">Leaderboard</a>
            <a href="#overview">Overview</a>
            <a href="#method">Method</a>
            <a href="#diagnosis">Analysis</a>
          </div>
          <a className="nav-cta" href={links.github}><BrandIcon src="https://cdn.simpleicons.org/github/17324D" label="GitHub" /> GitHub Repo <span aria-hidden="true">↗</span></a>
        </div>
      </nav>

      <SectionNavigation active={activeSection} />

      <header className="paper-hero container">
        <div className="venue-label">GENERAL AGENT BENCHMARK · 2026</div>
        <h1>OmniaBench</h1>
        <p className="paper-title">Benchmarking General AI Agents Across Diverse Scenarios</p>
        <div className="author-list" aria-label="Authors">
          {authors.map((author, index) => <span key={author}>{author}{index < authors.length - 1 ? "," : ""}</span>)}
        </div>
        <p className="affiliations">
          Peking University · Renmin University of China · Tsinghua University · Beijing Institute of Technology<br />
          Huawei Cloud Post-Training Team · Zhongguancun Academy
        </p>
        <div className="affiliation-logos" aria-label="University affiliations">
          <a href="https://www.pku.edu.cn/" target="_blank" rel="noreferrer">
            <span className="affiliation-mark"><img className="pku-mark" src={asset("/affiliations/pku.jpg")} alt="Peking University logo" /></span>
            <span>Peking University</span>
          </a>
          <a href="https://www.ruc.edu.cn/" target="_blank" rel="noreferrer">
            <span className="affiliation-mark"><img className="ruc-mark" src={asset("/affiliations/ruc.png")} alt="Renmin University of China logo" /></span>
            <span>Renmin University</span>
          </a>
          <a href="https://www.tsinghua.edu.cn/" target="_blank" rel="noreferrer">
            <span className="affiliation-mark"><img className="tsinghua-mark" src={asset("/affiliations/tsinghua.jpg")} alt="Tsinghua University logo" /></span>
            <span>Tsinghua University</span>
          </a>
          <a href="https://www.bit.edu.cn/" target="_blank" rel="noreferrer">
            <span className="affiliation-mark"><img className="bit-mark" src={asset("/affiliations/bit.jpg")} alt="Beijing Institute of Technology logo" /></span>
            <span>Beijing Institute of Technology</span>
          </a>
        </div>
        <p className="author-note">* Equal contribution &nbsp; † Project lead &nbsp; ‡ Corresponding authors</p>
        <p className="contact-line"><span>Contact</span><a href="mailto:scuuy05@gmail.com">scuuy05@gmail.com</a></p>

        <div className="resource-links">
          <a className="resource primary" href={links.paper} target="_blank"><BrandIcon src="https://cdn.simpleicons.org/arxiv/B31B1B" label="arXiv" /><span><b>Paper</b><small>arXiv</small></span></a>
          <a className="resource" href={links.github}><BrandIcon src="https://cdn.simpleicons.org/github/17324D" label="GitHub" /><span><b>Code</b><small>GitHub</small></span></a>
          <a className="resource" href={links.dataset}><HuggingFaceIcon /><span><b>Dataset</b><small>Hugging Face</small></span></a>
          <a className="resource" href="#leaderboard"><PodiumIcon /><span><b>Leaderboard</b><small>View results</small></span></a>
        </div>
      </header>

      <section className="metrics container" aria-label="Benchmark statistics" data-reveal>
        <div className="metric-primary"><strong>644</strong><span>challenging set · leaderboard</span></div>
        <div><strong>1,431</strong><span>tasks in the full set</span></div>
        <div><strong>354</strong><span>DAG tasks</span></div>
        <div><strong>60</strong><span>Solver tasks</span></div>
        <div><strong>30</strong><span>Program tasks</span></div>
        <div><strong>200</strong><span>DAG-S tasks</span></div>
      </section>

      <section className="section leaderboard-section" id="leaderboard" data-reveal>
        <div className="container">
          <div className="section-label">Evaluation results</div>
          <div className="section-heading leaderboard-heading">
            <div><h2>Leaderboard</h2><p className="section-subtitle">Current frontier performance on the 644-task challenging set.</p></div>
            <p>Overall is Pass@1 across all 644 tasks (micro-average). Similar final scores can reflect very different execution profiles.</p>
          </div>

          <div className="table-wrap">
            <table>
              <thead><tr><th>Rank</th><th>Model</th><th>Access</th><th>Effort</th><th>DAG</th><th>Solver</th><th>Program</th><th>DAG-S</th><th>Overall</th></tr></thead>
              <tbody>
                {leaderboard.map(row => (
                  <tr key={row.model}>
                    <td><span className={`rank rank-${row.rank}`}>{row.rank}</span></td>
                    <td className="model-name"><BrandIcon src={providerLogos[row.provider]} label={row.provider} /><span><b>{row.model}</b><small>{row.provider}</small></span></td>
                    <td><span className={`access ${row.access === "Open" ? "open" : ""}`}>{row.access}</span></td>
                    <td>{row.effort ? <span className="effort">{row.effort}</span> : <span className="effort-empty">—</span>}</td>
                    <td>{row.dag}</td><td>{row.solver}</td><td>{row.program}</td><td>{row.dags}</td><td className="overall-score">{row.overall}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="leaderboard-footer"><p>644 tasks: 354 DAG · 60 Solver · 30 Program · 200 DAG-S.</p><a href={links.paper} target="_blank">View all 22 models <span>↗</span></a></div>
          <RadarExplorer />
        </div>
      </section>

      <section className="section container" id="overview" data-reveal>
        <div className="section-label">Overview</div>
        <div className="abstract-grid">
          <h2>A broad, executable test of general agent capability.</h2>
          <div className="abstract-copy">
            <p>Large language models are evolving from text generators into general agents that understand requests, invoke external tools, and complete complex tasks through interaction. Existing benchmarks, however, often focus on limited scenarios, tool ecosystems, or interaction formats.</p>
            <p>OmniaBench evaluates agents across diverse application settings with explicit state spaces. Its taxonomy spans consumer, business, and employee settings, while executable environments require planning, tool use, state maintenance, and adaptation to feedback.</p>
          </div>
        </div>

        <figure className="paper-figure overview-figure">
          <img src={asset("/figures/overview.webp")} alt="OmniaBench data sources, environments, interaction trajectories, and evaluation" />
          <figcaption><b>Figure 1.</b> From real application knowledge to executable trajectories and verifiable outcomes.</figcaption>
        </figure>

        <div className="taxonomy-row">
          <article><span>ToB</span><strong>38</strong><h3>Business-facing domains</h3><p>186 fine-grained scenarios covering professional services, industry workflows, and enterprise operations.</p></article>
          <article><span>ToC</span><strong>22</strong><h3>Consumer-facing domains</h3><p>101 scenarios derived from real app ecosystems and everyday user needs.</p></article>
          <article><span>ToE</span><strong>30</strong><h3>Employee-oriented domains</h3><p>67 scenarios grounded in industry-general knowledge work and workplace tasks.</p></article>
        </div>
      </section>

      <section className="section section-tint" id="method" data-reveal>
        <div className="container">
          <div className="section-label">Data construction</div>
          <div className="section-heading">
            <h2>Four complementary routes</h2>
            <p>The 644-task challenging set contains 354 DAG, 200 DAG-S, 60 Solver, and 30 Program tasks, all sharing executable environments and structured evaluation.</p>
          </div>
          <div className="route-list">
            <article><div className="route-index">01</div><div><h3>DAG</h3><p>Multi-turn stateful interaction and tool-chain execution.</p></div><strong>354 tasks</strong></article>
            <article><div className="route-index">02</div><div><h3>DAG-S</h3><p>Single-turn tasks derived through query refinement.</p></div><strong>200 tasks</strong></article>
            <article><div className="route-index">03</div><div><h3>Solver</h3><p>Selection, scheduling, allocation, and optimization.</p></div><strong>60 tasks</strong></article>
            <article><div className="route-index">04</div><div><h3>Program</h3><p>Procedural reasoning with branching, iteration, and debugging.</p></div><strong>30 tasks</strong></article>
          </div>
          <figure className="large-figure"><img src={asset("/figures/pipeline.webp")} alt="Four-route OmniaBench environment and task construction pipeline" /><figcaption><b>Figure 2.</b> The multi-route construction pipeline, from taxonomy seeds and executable environments to task curation.</figcaption></figure>
        </div>
      </section>

      <section className="section container" id="diagnosis" data-reveal>
        <div className="section-label">Capability diagnosis</div>
        <div className="section-heading">
          <h2>Beyond a single success rate</h2>
          <p>Ten capability dimensions reveal where agents succeed, where they fail, and why similar overall scores can conceal different practical strengths.</p>
        </div>
        <div className="analysis-grid">
          <figure><img src={asset("/figures/capabilities.webp")} alt="Capability profiles and per-task score distributions" /><figcaption>Capability profiles across ten execution dimensions.</figcaption></figure>
          <figure><img src={asset("/figures/scenarios.webp")} alt="Model performance across scenario splits" /><figcaption>Pass@1 across ToB, ToC, and ToE scenario splits.</figcaption></figure>
        </div>
        <div className="capability-tags" aria-label="Ten capability dimensions">
          {["Task understanding", "Information gathering", "Planning & decision", "State management", "Tool use", "Code & programming", "Data analysis", "Office & documents", "Interaction & collaboration", "Reliability & safety"].map((item, i) => <span key={item}><b>{String(i + 1).padStart(2, "0")}</b>{item}</span>)}
        </div>
      </section>

      <section className="section findings-section" id="findings" data-reveal>
        <div className="container">
          <div className="section-label">Key findings</div>
          <div className="findings-list">
            <article><span>01</span><div><h3>Frontier models solve only about half of the benchmark.</h3><p>Claude-Sonnet-5 reaches 58.54 Overall Pass@1, showing substantial headroom even for the strongest evaluated systems.</p></div><strong>58.54%</strong></article>
            <article><span>02</span><div><h3>Reasoning—not tool syntax—is the primary bottleneck.</h3><p>Planning, decomposition, constraint maintenance, reflection, and adaptive correction account for most observed failures.</p></div><strong>53.8%</strong></article>
            <article><span>03</span><div><h3>Stronger models complete tasks with fewer tool steps.</h3><p>Longer trajectories often reflect redundant exploration or repeated replanning rather than more thorough execution.</p></div><strong>↓ steps</strong></article>
            <article><span>04</span><div><h3>Aggregate rankings hide scenario-specific strengths.</h3><p>Performance shifts substantially across business, consumer, employee, and fine-grained application domains.</p></div><strong>90 domains</strong></article>
          </div>
          <figure className="large-figure findings-figure"><img src={asset("/figures/analysis.webp")} alt="Tool efficiency, multi-run reliability, user simulation robustness, and error analysis" /><figcaption>Action efficiency, repeated-run reliability, robustness across user simulators, and the distribution of agent errors.</figcaption></figure>
        </div>
      </section>

      <section className="release-section" id="release" data-reveal>
        <div className="container release-inner">
          <div><div className="section-label">Open release</div><h2>Code, data, and evaluation tools</h2></div>
          <div><p>The benchmark environments, task sets, evaluation harness, and leaderboard are being prepared for public release.</p><a href="mailto:scuuy05@gmail.com">Contact the team <span>↗</span></a></div>
        </div>
      </section>

      <footer>
        <div className="container footer-inner"><div><a className="brand footer-brand" href="#top">OmniaBench</a><p>Benchmarking General AI Agents Across Diverse Scenarios</p></div><div className="footer-links"><a href={links.paper} target="_blank">Paper</a><a href="#leaderboard">Leaderboard</a><a href="mailto:scuuy05@gmail.com">Contact</a></div></div>
        <div className="container footer-bottom"><span>Huawei Cloud Post-Training Team</span><span>© 2026 OmniaBench</span></div>
      </footer>
    </main>
  );
}
