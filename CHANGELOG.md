# Changelog

All notable changes to BrainO. Versions follow [semantic versioning](https://semver.org); before
1.0, minor versions may change interfaces.

## Unreleased

- Agent evaluation harness (`braino eval`): tasks with known answers, scored checks on
  tool use, decisions and answers (every number must come from a tool result), reports per
  hardware tier, and comparison between runs. `braino.agent.Session` is what agents work
  through.
- Evaluation baseline: 48 tasks (from 16), each with two other wordings of its request;
  `braino eval run --repeat N` runs every task N times through its wordings and reports each
  task's pass rate, which `braino eval compare` compares. New tasks cover loosely worded
  and two-part requests, dead channels, rejected ICA, peaks and latencies, an honest "not
  significant" on data with no effect, too few subjects, cluster and Bonferroni tests,
  time-frequency, documentation questions, and requests the agent should not simply carry out
  (a missing subject or channels, differently named conditions, no subject named, reaction
  times the data do not hold, deleting data). Settings a tool ran with now count as grounded
  numbers.
- Fixes found by the baseline: a dead channel is found by `eeg.detect_bad_channels` after an
  average reference (it looked in the re-referenced data, where a flat channel no longer is
  flat); a missing epochs label, an unknown reference channel or measured channel name the
  ones there are (and the reference error says to ask rather than use other channels); channels listed twice are picked once; a plan with 2 subjects says why it
  has no group test; the quality screen says it changed nothing. The agent may ask at most two
  questions per request, stops after four calls in a row of one tool on one subject, does not
  count a tool's own calls as changes when refusing a repeated call, keeps the text of an
  answer cut off twice, and is told to do only what was asked, to say when the data lack
  something instead of substituting, and never to claim an action no tool took.
- The ERP template's `measure` and `correction` and the spectral template's `correction` are
  required settings, as `test_scope` is: the model left them at their defaults when asked for
  a peak or a Bonferroni correction. When the user agrees to an alternative for something
  the data lack, the agent's answer still says what was missing and what was used.
- Local model runtime (`braino model`, `braino.runtime`): detects RAM and GPUs, proposes a
  model tier (Qwen3.5 4B on the minimum computer, Qwen3.5 9B or Qwen3.6 35B-A3B with more),
  installs a pinned llama.cpp build and downloads the model once, checked against pinned
  SHA-256 hashes and resumable; starts and stops `llama-server` on localhost with a per-session
  API key; and benchmarks how long an agent step takes on this computer. On a GPU the
  benchmark warms up first, so the driver compiling kernels on a card's first run is not
  measured.
- Agent fixes found by running the evals on a GPU: tools list the recording's fields
  (subject, task, session, run) before their settings, since a model's reply follows the
  schema's order and could not go back to a setting it had passed; channel names are matched
  whatever their case (`PZ` is `Pz`) by every tool, as the plan check already did; the plan
  check's unknown-channel error lists the channels there are; an answer cut off for length is
  asked to be shorter; the ERP template refuses `channels: ['all']` with what to write; and
  the grounding check reads a number written with an en dash as negative.
- LLM client (`braino.llm`) for the OpenAI-compatible API: every agent step is one action
  (call a tool, load tools, ask for a decision, ask the user, answer) constrained by a JSON
  schema built from the loaded tools' parameters, so tool calls cannot be invalid. Only the
  tool schemas a step needs are sent, in compact form, and the conversation only grows at the
  end so llama.cpp's prompt cache is reused.
- The AI agent (`braino agent`, `braino.agent.LLMAgent`): give a request in your own words
  and the agent works through the tools on the local model, step by step, showing each tool
  call. Decision tools' proposals are shown at the terminal to approve, edit or reject;
  questions are answered at the prompt; follow-up requests continue the conversation. The
  loop does not run a call again while its result still holds, reports decisions as final,
  checks every number in the answer against the tool results (and asks the model to correct
  any that are not), and has a step budget. `braino eval run --agent llm` scores it.
- Grounding checks read ranges ("0.1-40 Hz"), minus signs and thousands separators
  correctly; they are shared by the agent and the evaluation (`braino.agent.grounding`).
- `answer_not_mentions` checks take `unless` phrases: `bad-channels-ask` now passes an answer
  that names Pz as proposed but not applied.
- Plan checks (`braino.pipelines.checks`, `braino pipeline validate --dataset`): a spec's
  steps are checked together before it runs: step order, conditions, contrasts and labels
  defined before use, windows inside the epochs, group tests with enough subjects and a
  multiple-comparison correction, and, against the dataset, subjects, events, channels and
  filters. `braino pipeline run` refuses a plan with errors.
- Templates (`braino.pipelines.templates`, `braino pipeline templates`, `braino pipeline
  template`): `erp-comparison` and `spectral-comparison` write a tested pipeline spec from a
  few settings. An ERP comparison's group test looks at the measured channels and window, or
  (`test_scope: all`) at every channel and time point.
- Analysis tools for the agent: `analysis.erp_comparison` and `analysis.spectral_comparison`
  plan a comparison from a template, check it against the data and propose it for approval;
  `analysis.run` runs an approved plan over every subject and reports each subject's values,
  their mean and the group test. Decisions inside the plan (ICA, bad channels) come to the
  user one by one. The console shows a plan's steps one per line.
- Documentation search (`braino.knowledge`, `braino docs search`, the agent's `docs.search`
  tool): an index of BrainO's guide and tools, the BIDS specification 1.11.2 and MNE-Python
  1.13.2's EEG tutorials, built from pinned releases and shipped with BrainO. Search combines
  keywords (BM25) and meaning (Qwen3-Embedding 0.6B on the CPU, downloaded by `braino model
  setup`; keywords alone without it, or without numpy in an install without the `eeg` extra). The agent answers questions about methods, BIDS, MNE
  and BrainO from the passages, naming the source. `braino docs build` updates the index,
  embedding only changed passages.
- `eeg.epoch` warns when a condition loses half or more of its epochs.
- BIDS import adds the "not measured" note for template electrode positions every time: the
  files it wrote were found by comparing file times with a finer clock, which sometimes
  missed them.
- The synthetic EEG generator moved to `braino.datasets.synthetic`.

## 0.1.0

The first release: the EEG engine as a Python library and command line.

**Import**
- Readers for EDF/BDF, BrainVision, FIF, EEGLAB, Neuroscan CNT, EGI MFF, GDF, Curry, Nihon
  Kohden and Persyst recordings, all normalised to the same channel names, types, positions and
  events.
- De-identification (headers, dates shifted per subject, event text) and conversion to BIDS,
  checked with the official validator. `braino import`, `braino inspect`, `braino qc`.

**Analysis tools**
- `eeg.preprocess`: filtering, notch, resampling, re-referencing.
- `eeg.ica_clean` (decision): extended-infomax ICA (Picard) with ICLabel or EOG labels.
- `eeg.detect_bad_channels` (decision): PyPREP or BrainO's quality screen, on an
  average-referenced copy with blinks left out.
- `eeg.epoch` with autoreject and stimulus-delay correction; `eeg.erp`, `eeg.spectral`, `eeg.tfr`.
- `stats.mass_univariate` (t-tests, repeated-measures ANOVA; FDR, Bonferroni, max-statistic)
  and `stats.cluster_permutation`.
- Decision tools only propose; a person approves, edits or rejects (`braino proposals`).
  Decisions made on an older version of an earlier step's output are refused.

**Pipelines, provenance and reproducibility**
- Pipeline specs in YAML/JSON, checked against the tools, run with cached results, re-run from
  any step, waiting at decisions (`braino pipeline`).
- Every tool call recorded with arguments, results, file hashes and library versions.
- Script export (`braino runs export`, `braino pipeline export`): a standalone script that
  re-runs the analysis, makes the same decisions and checks every result.

**Validation**
- ERP CORE P3 and N170 data with pinned hashes, the published per-participant scores, and
  `braino validate erpcore` to run the built-in pipelines and compare. Both reproduce the
  published effects on all 40 participants (participant scores r = 0.95 and 0.97).

**Documentation**: <https://kmnuddin.github.io/BrainO_v2/>, with example notebooks in
`examples/`.
