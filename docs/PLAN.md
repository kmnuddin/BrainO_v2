# BrainO 2.0 — Platform Plan

Status: **draft for discussion** · Last updated: 2026-10-08

## 1. Vision

BrainO is an open-source **EEG and fMRI analysis platform** with:

- **Universal data import** — every common EEG/MEG and MRI/fMRI format, normalised to BIDS.
- **Validated analysis** — preprocessing, statistics, machine learning and connectivity built on
  established open-source neuroscience libraries.
- **AI agents** — an analysis agent that plans and runs pipelines through real, executed tools and
  explains the results, with the researcher approving key decisions.
- **Interactive 3D visualisation** — a Unity-powered brain viewer embedded in the application for
  surfaces, sensors, statistical maps and networks.

### Decisions taken

| Topic | Decision |
|---|---|
| Project model | **Open source**, developed by a single developer (contributors welcome later) |
| Users | Research first, with a path to clinical use (design for PHI handling and audit from day one) |
| AI | **Own local AI built on open-weight Qwen models** — no Claude or other hosted LLM API. Runs on the user's machine; fine-tuned for BrainO's tools |
| Compute | Local engine on the user's machine or lab server; heavy jobs can be offloaded to cloud/HPC |
| Main application | React + TypeScript web UI, packaged as a desktop app (Electron); can also be served to a browser later |
| 3D viewer | Unity 6 LTS, **visualisation only**, built for WebGL and embedded as a panel in the main UI |
| VR/AR | **Out of scope** |
| First milestone | EEG pipeline + agent, with results shown in 3D |

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Desktop app (Electron)                                           │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ Main UI — React + TypeScript                                 │ │
│ │  data browser · pipeline builder · EEG trace browser ·       │ │
│ │  2D plots · stats tables · agent chat + approvals · reports  │ │
│ │                                                              │ │
│ │   ┌────────────────────────────────────────────┐             │ │
│ │   │ 3D viewer panel — Unity 6 WebGL build      │             │ │
│ │   │ brain surfaces · sensors · stat maps ·     │◄─ JS bridge │ │
│ │   │ connectomes · volumes                      │  (commands, │ │
│ │   └───────────────────▲────────────────────────┘  selections)│ │
│ └───────────────────────┼──────────────────────────────────────┘ │
└──────────▲──────────────┼────────────────────────────────────────┘
           │ REST +       │ HTTP (binary meshes/arrays,
           │ WebSocket    │ fetched directly by the viewer)
┌──────────┴──────────────┴────────────────────────────────────────┐
│ Engine server (Python, FastAPI) — runs as a local sidecar         │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ Agent      │→ │ Tool registry│→ │ Analysis modules          │   │
│  │ (local     │  │ typed tools, │  │ io · eeg · fmri · stats · │   │
│  │  Qwen LLM) │  │ sandboxed    │  │ ml · connectivity         │   │
│  └─────▲──────┘  └──────────────┘  └──────────────────────────┘   │
│        │ OpenAI-compatible API (localhost)                        │
│  ┌─────┴──────────────────────────┐                               │
│  │ LLM server: Ollama / llama.cpp │  + knowledge base (RAG over   │
│  │ / vLLM serving a Qwen model    │    MNE/nilearn docs, methods) │
│  └────────────────────────────────┘                               │
│  Job runner (local processes; optional cloud/HPC backend)         │
│  Provenance store · Scene service (meshes/arrays for the viewer)  │
└──────────▲────────────────────────────────────────────────────────┘
           │
┌──────────┴────────────────────────────────────────────────────────┐
│ Data: BIDS dataset + derivatives/ (BIDS-Derivatives layout)       │
└───────────────────────────────────────────────────────────────────┘
```

Principles:

1. **Each layer does one job.** Python does the science, the React app does the workflow and the
   2D views, and Unity only renders 3D. Unity contains no analysis code, file I/O or business logic.
2. **BIDS everywhere.** Every import is converted to, or indexed as, BIDS. Every analysis output is
   written as a BIDS derivative.
3. **The agent never invents a number.** Every value shown comes from an executed tool. The agent's
   job is to plan, choose parameters, interpret results and explain them.
4. **Reproducible by construction.** Each analysis run exports a standalone script plus a provenance
   record (inputs, hashes, parameters, library versions).
5. **The engine works without the agent.** All tools can also be used from the UI and a Python API,
   so the platform is useful without AI and can be tested on its own.

### Why Electron rather than Tauri
Tauri renders with the operating system's webview, which differs between platforms (WebView2 on
Windows, WebKit on macOS, WebKitGTK on Linux), and WebGL support and performance vary between them.
Electron ships its own Chromium, so the embedded Unity WebGL viewer behaves the same everywhere. We
can revisit this if installer size becomes a problem.

## 3. Data formats

| Modality | Formats | Library |
|---|---|---|
| EEG/MEG | EDF/EDF+, BDF, BrainVision, EEGLAB .set, FIF, Neuroscan CNT, EGI MFF, GDF, XDF (LSL), Curry, Nihon Kohden, Persyst, BIDS-EEG | MNE-Python, MNE-BIDS, pyxdf |
| MRI/fMRI | DICOM, NIfTI-1/2, PAR/REC, Analyze, AFNI BRIK/HEAD, MGH/MGZ, BIDS | dcm2niix, nibabel |
| Surfaces | FreeSurfer surf/annot/label, GIFTI, CIFTI-2 | nibabel, nilearn |
| Derived | Connectivity matrices (CSV/MAT/NPY), source estimates, atlases | numpy, scipy, MNE |

The import step is one pipeline: **detect → read → validate → de-identify → convert to BIDS → index.**
It produces a QC summary that both the UI and the agent read.

## 4. Analysis capability map

**EEG** (milestone 1 in bold)
- **Filtering, re-referencing, resampling, bad-channel detection (PyPREP), ICA artifact removal
  (ICLabel via mne-icalabel), epoching, autoreject**
- **ERPs, power spectra (Welch/multitaper), time–frequency analysis (Morlet), band power**
- **Sensor-level statistics: t-tests/ANOVA, cluster-based permutation tests, FDR/FWER correction**
- Source localisation (MNE/dSPM/eLORETA with fsaverage or an individual MRI)
- Connectivity: coherence, PLV, wPLI, Granger/PDC (mne-connectivity), graph metrics
- ML decoding: CSP+LDA, Riemannian methods (pyriemann), temporal generalisation, deep learning
  (braindecode)
- Microstates, ERP component detection, sleep staging (YASA)

**fMRI** (phase 3)
- Preprocessing with fMRIPrep (container, optional), confound handling
- First- and second-level GLM (nilearn), contrasts, thresholding
- Resting-state functional connectivity, ICA/dual regression, seed-based maps
- MVPA/searchlight, ROI analysis on the existing BrainO atlases

**Multimodal** (phase 5): EEG-informed fMRI, joint source and connectivity views.

## 5. Agent design

### 5.1 Structure
- One **orchestrator agent** running on a **local, open-weight Qwen model** with native tool
  calling. Nothing is sent to an external AI service.
- Specialist sub-agents (QC reviewer, statistics reviewer) are added only when an evaluation shows
  they help.

### 5.2 BrainO's own AI: what "building it" means
Training a language model from scratch is not realistic for one developer. "Our own AI" is built
in four layers on top of Qwen:

1. **Base model** — an open-weight Qwen instruct model with tool calling, served locally through an
   OpenAI-compatible API (Ollama or llama.cpp for desktops, vLLM for lab servers). The engine talks
   only to that API, so any compatible model can be swapped in.
2. **Model tiers by hardware** (4-bit quantised, approximate):

   | Tier | Example Qwen size | GPU memory | Use |
   |---|---|---|---|
   | Laptop | ~4–8B dense | 6–8 GB (or CPU, slowly) | Explanations, simple pipelines |
   | Workstation (default) | ~30B MoE or ~32B dense | 20–24 GB | Full agent |
   | Lab server | Largest Qwen that fits | 48–80 GB+ | Best quality, multi-user |

   **Development hardware available:**
   - *Office workstation, RTX 2080 Ti (11 GB):* runs the laptop tier (~8B, 4-bit) fully on the GPU
     for day-to-day development. A ~30B mixture-of-experts model also runs, with experts partly
     offloaded to CPU RAM by llama.cpp; it is slower but fine for testing the workstation tier.
   - *University of Memphis HPC, V100 GPUs:* fine-tuning (QLoRA) and large evaluation runs, submitted
     as batch jobs. Model serving for evals uses llama.cpp, which supports this GPU generation;
     check vLLM's current GPU support before relying on it there.
   - Both GPUs are pre-Ampere: **no bfloat16 and no FlashAttention 2**. Training uses fp16 mixed
     precision with loss scaling; inference uses quantised GGUF models.

3. **Knowledge (RAG)** — a local search index over MNE-Python, nilearn and BIDS documentation,
   BrainO's own tool docs, and methods guidelines (e.g. COBIDAS / COBIDAS-MEEG). The agent looks
   these up instead of relying on memory.
4. **Fine-tuning (LoRA/QLoRA)** — teach the model BrainO's tools and analysis conventions using
   tool-call examples: synthetic analysis sessions generated from validated pipeline templates on
   public datasets, then checked by running them. This specialises the model; it does not replace it.

Before fine-tuning anything, the agent is measured with the base model + RAG on the eval set
(M1.7). We fine-tune only where the evaluation shows gaps.

### 5.3 Making a local model reliable
Local models are weaker planners than the largest hosted models, so reliability comes from the
system design, not the model alone:

- **Constrained output:** tool calls are forced to match their JSON schema (grammar-constrained
  decoding in llama.cpp, structured outputs in vLLM). Invalid calls are impossible, not just unlikely.
- **Validated pipeline templates:** for common analyses (ERP comparison, spectral comparison,
  resting-state connectivity) the agent fills in a tested template rather than inventing a
  pipeline from scratch. Free-form planning is reserved for unusual requests.
- **Plan check:** before running, the agent's plan is checked by rules (required steps present,
  order valid, multiple-comparison correction chosen) and shown to the user.
- **Small, explicit tool results:** tools return short structured summaries so the context stays
  within what a local model handles well.

### 5.4 Tool contract
Each tool is a typed Python function (Pydantic input and output schemas) that:
- runs in the job runner, not inside the model,
- returns a compact structured summary to the agent and writes full outputs to `derivatives/`,
- logs a provenance entry,
- declares a risk level: `read`, `compute`, or `decision`. **`decision` tools (dropping
  channels or subjects, choosing a statistical model, correcting for multiple comparisons) require
  user approval in the UI.**

Initial EEG tool set:

| Tool | Purpose |
|---|---|
| `dataset.inspect` | Formats, subjects, sessions, channels, sampling rate, events, durations |
| `eeg.qc_report` | Line noise, flat or noisy channels, artifact estimate |
| `eeg.preprocess` | Filter, notch, reference, resample (configurable) |
| `eeg.detect_bad_channels` | PyPREP; proposes channels to drop (`decision`) |
| `eeg.ica_clean` | ICA + ICLabel; proposes components to remove (`decision`) |
| `eeg.epoch` | Epoch by events, baseline, reject |
| `eeg.erp` | Condition averages, peak and latency measures |
| `eeg.spectral` | PSD and band power |
| `eeg.tfr` | Time–frequency decomposition |
| `stats.cluster_permutation` | Sensor/time/frequency cluster tests |
| `stats.mass_univariate` | t-test/ANOVA with FDR/FWER correction |
| `viz.show` | Show a result in the UI: a 2D plot, or a 3D scene in the viewer |
| `report.generate` | HTML/PDF report and re-runnable Python script |

### 5.5 Safety and data governance
- **No data leaves the machine.** The LLM runs locally, so there is no external AI service to
  trust. The agent still only receives metadata, QC summaries and statistics (not raw signals), to
  keep its context small and its reasoning focused.
- De-identification on import: EDF patient fields, DICOM PHI tags, and file names.
- Statistical guardrails built into the tools: warn about circular analysis, ML train/test leakage
  (e.g. ICA or scaling fitted on all data), missing multiple-comparison correction and
  underpowered designs.
- Full audit log of agent actions and user approvals (needed for any future clinical path).

## 6. User interface

### 6.1 Main UI (React + TypeScript)
- **Data browser:** datasets, subjects and sessions, import wizard, QC status.
- **EEG trace browser:** scrolling multichannel view with annotations and bad-channel marking
  (WebGL-accelerated canvas, so long high-density recordings stay smooth).
- **2D plots:** ERPs, spectra, time–frequency maps, topomaps, stats results (Plotly or similar).
- **Pipeline view:** the steps of a run, their parameters and status; re-run from any step.
- **Agent panel:** chat, the agent's plan, approval prompts for `decision` tools, links to outputs.
- **Reports:** generated HTML/PDF, and export of the re-runnable script.

### 6.2 3D viewer (Unity 6, WebGL)
Shows only what the engine sends. It does no analysis or file I/O.

Built from scratch in Unity 6 (URP). Planned views:
- Cortical surfaces (fsaverage, individual FreeSurfer) with per-vertex data
- Head model with EEG sensor layout; animated topographies over time
- Volume slices and glass-brain view for fMRI maps
- Picking: clicking a sensor, vertex or region tells the main UI, which can open the related plot

**Communication**
- *Main UI → viewer* (JS bridge): load a scene, change threshold, colour map or time point,
  highlight, capture a screenshot.
- *Viewer → main UI* (JS bridge): selection events, camera state, ready/error.
- *Viewer ↔ engine* (HTTP): large binary payloads (meshes, per-vertex and per-timepoint arrays)
  are fetched directly from the engine's scene service, so they do not pass through JavaScript.

**Viewer architecture**
- No analysis, file parsing or menus in Unity: it only receives scenes and commands.
- Dependency injection with VContainer and a small message bus driven by the JS bridge.
- Data arrives as compact binary arrays (vertices, faces, per-vertex/per-timepoint values) from the
  engine's scene service, so the viewer has no dependency on neuroimaging file formats.
- Atlas region coordinates and connectome views (nodes and edges with threshold bands) are rebuilt
  on the same scene format.

## 7. Repository layout (proposed)

```
BrainO/
├── engine/                 # Python package `braino`
│   ├── braino/io/          # readers, de-identification, BIDS conversion
│   ├── braino/eeg/         # preprocessing, ERP, spectral, TFR
│   ├── braino/fmri/        # (phase 3)
│   ├── braino/stats/
│   ├── braino/ml/
│   ├── braino/provenance/
│   ├── braino/agent/       # orchestrator, tool registry, templates, plan checks
│   ├── braino/llm/         # client for the local OpenAI-compatible LLM server, RAG index
│   ├── braino/scene/       # mesh/array export for the 3D viewer
│   ├── braino/server/      # FastAPI REST + WebSocket
│   ├── braino/cli.py       # command-line interface
│   └── tests/
├── training/               # dataset generation, LoRA fine-tuning scripts, agent evals
├── app/                    # React + TypeScript UI and Electron shell
├── viewer/                 # Unity 6 project (WebGL build), added in M1.6
├── docs/
└── examples/               # public sample datasets + notebooks
```

## 8. Roadmap

### Order of work for a single developer
Four technology stacks (Python, local LLM, React/Electron, Unity) is a lot for one person, so work
is ordered so that **each step produces something people can use** and the riskiest science comes
first:

1. **Engine as a Python library + CLI** — useful to researchers from day one, through scripts and
   notebooks, and the base for everything else.
2. **Agent in the CLI/notebook** — proves the local-AI idea before investing in UI.
3. **Main UI** — makes it usable without code.
4. **3D viewer** — the new Unity viewer, once the data it shows exists.

### Milestone 1 — EEG pipeline + agent

Durations assume one developer working full-time; part-time roughly doubles them.

| Step | Deliverable | Done when | Approx. time |
|---|---|---|---|
| M1.0 Foundation | Repo layout, Python package, CI (ruff, mypy, pytest), contribution docs | CI green; `pip install -e engine` works | 2–4 weeks |
| M1.1 Import | EEG readers → BIDS, de-identification, `dataset.inspect`, QC report, CLI | All listed EEG formats load from public test files | 1.5–2 months |
| M1.2 Pipeline | Preprocess, bad channels, ICA, epoching, ERP, spectral, TFR, cluster stats, provenance, script export | Reproduces a published ERP effect on a public dataset | 2.5–3 months |
| **Release v0.1** | Engine on PyPI with docs and example notebooks | People outside the project can use it | — |
| M1.3 Local AI | LLM client, tool registry, templates, plan checks, RAG index, approvals in the CLI | "Compare conditions A vs B" runs end to end from the CLI on the workstation model | 2–3 months |
| M1.4 Evaluation | Agent eval set (tasks + expected outputs); first LoRA fine-tune only if evals show gaps | Eval pass rate tracked; base vs fine-tuned compared | 1–1.5 months, then ongoing |
| **Release v0.2** | Engine + local agent | — | — |
| M1.5 Engine server + main UI | FastAPI server, Electron/React app: data browser, trace browser, plots, agent panel with approvals | The M1.3 workflow runs entirely from the app | 3–4 months |
| M1.6 3D viewer | New Unity 6 WebGL viewer, embedding, JS bridge, scene service; sensor topographies over time | Results appear in the embedded 3D panel; clicking a sensor opens its ERP | 2–3 months |
| **Release v1.0** | Desktop app | — | — |

**Total for milestone 1: about 13–18 months full-time.**

### Milestone 1 build order
The detailed order of work within milestone 1. Near-term stages are specified in more detail than
later ones; later stages will be refined as they come closer.

**Progress (2026-10-08):** Stage A done (items 1–4), the Qwen spike done, item 5 done except XDF.
Next: item 6.

**Stage A — Engine plumbing (before any EEG code, ~2–3 weeks).** Every later tool depends on
these, and they are cheapest to settle while only one tool exists.

1. **Run context in the tool contract.** Tools receive a `RunContext` (BIDS root,
   `derivatives/braino/` output folder, run ID, provenance logger) in addition to their arguments.
2. **Provenance store.** Each tool call records input file hashes, parameters, library versions,
   outputs and timing as JSON next to the derivatives. Script export and the agent's audit log
   are both built from this record.
3. **Propose/apply for `decision` tools.** A decision tool returns a proposal (e.g. channels to
   drop); a separate apply step commits it. The CLI, the UI and the agent then all handle approval
   the same way.
4. **Test-data setup.** Small public sample files for each format, downloaded with `pooch` and
   cached in CI; network-dependent tests are marked. Scientific dependencies (mne, mne-bids, numpy)
   go in an `[eeg]` extra so the core install stays light.

*Optional spike (2–3 days):* serve a small Qwen model with llama.cpp on the RTX 2080 Ti and have
it call `system.info` through `registry.schemas()`. This tests local tool calling end to end
months before M1.3, while the tool contract is still cheap to change.
*Result (2026-10-07, `spikes/llm_tool_call.py`):* Qwen3.5-9B Q4_K_M on llama.cpp (CUDA, `--jinja`,
temperature 0) passed 9/9 cases: tool choice, arguments, chained calls, recovering from tool
errors, a decision tool whose proposal was then approved, and no invented numbers. About 80
tokens/s using 6.4 GB of VRAM. The tool contract needed no changes.

**Stage B — M1.1 Import.**

5. **Format detection and readers:** EDF/BDF, BrainVision, FIF and EEGLAB first; then CNT, MFF,
   GDF, Curry, Nihon Kohden and Persyst; XDF last (custom code on top of pyxdf).
   *Done for all MNE-backed formats (`braino.io`); XDF remains.*
6. **Normalised recordings:** channel types, montages, and events/annotations. Every format stores
   events differently, so this is expected to be the hardest part of import.
7. **De-identification:** EDF patient fields, measurement info and file names, with tests that
   check the personal data is gone.
8. **BIDS conversion** with mne-bids. Subject, session and task labels come from CLI options or a
   mapping file; the result is checked with bids-validator.
9. **`dataset.inspect` and `eeg.qc_report` tools**, with CLI commands `braino import`,
   `braino inspect` and `braino qc`.

**Stage C — M1.2 Pipeline → v0.1.**

10. **Choose the validation dataset first** (ERP CORE is a candidate), so the pipeline is built
    against the published effect it must reproduce.
11. **Tools, in dependency order:** `eeg.preprocess` → `eeg.detect_bad_channels` (decision) →
    `eeg.ica_clean` (decision) → `eeg.epoch` with autoreject → `eeg.erp`, `eeg.spectral`,
    `eeg.tfr` → `stats.mass_univariate`, `stats.cluster_permutation`.
12. **Pipeline spec and runner:** steps declared in YAML/JSON, results cached, re-run from any
    step. These specs later become the agent's validated templates.
13. **Script export** from the provenance record.
14. **Docs site, example notebooks and a PyPI release workflow** → v0.1.

**Stage D — M1.3–M1.4 Local agent → v0.2.**

15. **Eval harness first** (tasks with expected outputs), so every agent change is scored.
16. **LLM client** for the OpenAI-compatible API, with schema-constrained tool calls.
17. **Agent loop in the CLI**, with approval prompts for decision tools.
18. **Templates and plan checks**, built on the Stage C pipeline specs.
19. **RAG index** over the MNE, BIDS and BrainO documentation.
20. **Baseline evaluation**, then LoRA fine-tuning on the HPC only where the evals show gaps.

**Stage E — M1.5 Server and UI.**

21. **FastAPI server** exposing the tool registry (already schema-driven, so this layer stays thin).
22. **Job runner** with progress reported over WebSocket.
23. **Electron/React app**, in this order: data browser → plots → agent panel with approvals →
    trace browser (the hardest view, so it comes last).

**Stage F — M1.6 Unity viewer → v1.0.**

24. **Scene service** in the engine for binary meshes and arrays.
25. **Unity viewer:** WebGL build, JS bridge, sensor topographies over time.
26. **Picking:** clicking a sensor in the 3D view opens its ERP plot in the main UI.

### Later phases
- **Phase 2 — EEG depth:** source localisation on cortical surfaces, connectivity, ML decoding,
  BrainO connectome view driven by computed connectivity.
- **Phase 3 — fMRI:** DICOM/NIfTI import, fMRIPrep integration, GLM, resting-state connectivity,
  surface and volume rendering in the viewer.
- **Phase 4 — Collaboration and scale:** group studies, HPC job backend, browser access to a lab
  server, shared projects.
- **Phase 5 — Multimodal and extensibility:** EEG-fMRI fusion, plugin SDK for third-party tools.

## 9. Resources (open source, single developer)

| Item | Cost |
|---|---|
| All software: MNE, nilearn, scikit-learn, PyTorch, FastAPI, React, Electron, Ollama/llama.cpp/vLLM, Qwen weights | Free (open source / open weights) |
| Unity 6 | Free (Unity Personal; no revenue) |
| GitHub, CI for public repos, PyPI, Read the Docs | Free |
| Development GPU | Already available: RTX 2080 Ti (office) |
| Fine-tuning and large evaluations | Already available: University of Memphis HPC (V100), free |
| Code-signing certificates for installers (macOS $99/yr; Windows varies) | Optional; unsigned builds work but show security warnings |
| Test datasets | Free (OpenNeuro, MNE sample data) |

The main resource is **developer time** (section 8). Grants for open-source research software
(e.g. NIH BRAIN Initiative informatics, NSF CSSI, Chan Zuckerberg Initiative EOSS) can fund it, and
a paper in the Journal of Open Source Software (JOSS) helps adoption and citation.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Scope is very large for one developer | Library-first order with usable releases at each step; build depth before breadth; invite contributors after v0.1 |
| Local model plans or reasons poorly | Constrained tool calls, validated templates, rule-based plan checks, approvals, eval suite; fine-tune where evals show gaps |
| Users without a capable GPU | Tiered models; the engine and UI work fully without the agent |
| Agent produces wrong science | Executed tools only, guardrails, approvals, eval suite, re-runnable scripts |
| WebGL memory and performance limits for large surfaces/volumes | Server-side decimation and level-of-detail meshes, streaming time series in chunks, 2D slice rendering in the main UI as a fallback |
| Unity WebGL build size and load time | Keep the viewer small, compressed builds, load the viewer once and keep it alive |
| Licensing | See open question 1. Model weights are downloaded by the user, not bundled; check each Qwen model's licence (most recent Qwen releases are Apache 2.0, some earlier sizes used a custom Qwen licence). FSL/FreeSurfer stay optional external tools, never bundled. |
| PHI and clinical regulation | De-identify on import, everything local, audit log; clinical use needs a separate regulatory track (e.g. FDA SaMD / EU MDR) |

## 11. Open questions

1. **Licence:** keep GPLv3 or move to a permissive licence (MIT/BSD-3, common in the MNE/nilearn
   ecosystem)? All code is new, so this is your decision alone. Note that GPL code depending on
   the proprietary Unity engine is a known grey area; a permissive licence (at least for
   `viewer/`) avoids it.
2. Real-time EEG streaming (LSL) for neurofeedback/BCI: in scope, and when?
3. Which in-house datasets should drive milestone 1 validation, in addition to public ones?
4. Packaging: one installer bundling the Python runtime and LLM server, or a pip/conda engine plus
   the desktop app?
