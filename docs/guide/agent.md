# Asking the agent

BrainO's AI agent analyses a dataset from a request in your own words. It runs on the local
model on your own computer (see [the local AI model](model.md)), so nothing leaves the
machine. It works only through BrainO's tools, the same ones you can run yourself, and it
never makes a decision for you.

```bash
braino model setup                                   # once: install and download the model
braino agent --dataset ./my-study "Check the data quality of subject 02."
braino agent --dataset ./my-study                    # type requests one after another
```

## What you see

Each step the agent takes is printed as it happens: the tool it runs, with its arguments, and
how it went.

```text
> eeg.preprocess(subject="02")
  done in 3.1 s
> eeg.detect_bad_channels(subject="02", method="qc")
  done in 2.4 s, proposal 20261102T101500Z-1a2b3c-2

Decision needed: eeg.detect_bad_channels proposes
    recording: derivatives/braino/sub-02/eeg/sub-02_task-oddball_desc-preproc_eeg.fif
    channels: T8, Pz
    reasons: {"T8": ["amplitude far from other channels"], "Pz": ["unusual power spectrum"]}
The agent says:
    T8 and Pz look bad in subject 02.
Approve, edit or reject? [a/e/r]: e
Fields to change, as JSON (e.g. {"recording": ...}; empty to go back): {"channels": ["T8"]}
Note for the record (optional): Pz carries the P3

T8 is now marked bad in subject 02; Pz was kept, as you decided.
```

On a computer without a GPU each step takes 5 to 40 seconds; the first step of a session is
the slowest. Add `--verbose` to see the model's replies and how long each took.

## Decisions are yours

Some tools only *propose*: which channels are bad, which ICA components to remove. When the
agent has such a proposal, it shows it to you with its explanation, and you choose:

- **a** (approve): the proposal is applied as it is.
- **e** (edit): change some of its fields first, as JSON, for example a shorter list of
  channels. Changes are checked before anything is applied; a mistake asks again.
- **r** (reject): nothing is changed.

You can add a note to either. The decision is recorded with your name, your note and what
was applied, in the run's provenance, and the agent is told the decision is final: it
reports what was applied, and does not ask again.

The agent can also ask you a question when your request is unclear (which subject, which
conditions); type the answer at the prompt.

## Comparing conditions

To compare two conditions, for one subject or across all of them, the agent does not plan
each step itself. It fills in a tested [template](pipelines.md#templates) (`erp-comparison`
or `spectral-comparison`) with the conditions, their events, and where and when to measure.
The plan is [checked](pipelines.md#plan-checks) against the data first: a condition whose
events do not occur, or a channel that does not exist, goes back to the agent to fix. Then
the whole plan is shown to you to approve before anything runs:

```text
> analysis.erp_comparison(condition_a="rare", events_a=["stimulus/rare"], ...)
  done in 0.4 s, proposal 20261103T091200Z-4d5e6f-2

Decision needed: analysis.erp_comparison proposes
    template: erp-comparison
    name: erp-rare-vs-frequent
    task: oddball
    subjects: 01, 02, 03, 04, 05, 06
    steps:
        eeg.preprocess(l_freq=0.1, h_freq=30.0, reference="average")
        eeg.epoch(conditions={"rare": ["stimulus/rare"], ...}, tmin=-0.2, tmax=0.8, ...)
        eeg.erp(contrasts=[...], measures=[{"name": "amplitude", "channels": ["Pz"], ...}])
    group_steps:
        stats.mass_univariate(conditions=["rare_minus_frequent"], channels=["Pz"], ...)
    warnings: (none)
    settings: {...}
Approve, edit or reject? [a/e/r]: a
> analysis.run(name="erp-rare-vs-frequent")
    sub-01     preprocess       ran
    sub-01     epochs           ran
    ...
    group      group_test       ran
  done in 11.2 s
```

`analysis.run` runs only plans you approved. It reports each subject's values, their mean
and the group test, and every step is recorded as its own tool call. To change the subjects,
edit `subjects` before approving. To change a setting, reject the plan with a note saying
what to change, and the agent plans again. If the plan includes ICA or bad-channel
detection, each subject's proposal comes to you in turn, and the run continues once you have
decided them. The approved spec is kept in `derivatives/braino/pipelines/<name>/spec.yaml`:
`braino pipeline run` reruns it without the agent.

## Looking things up

For questions about methods, data formats or BrainO itself ("why high-pass before ICA?",
"which columns must events.tsv have?"), the agent searches the documentation that ships
with BrainO (`docs.search`) and answers from what it finds, naming the source, instead of
from what the model remembers. The index holds:

| Source | What | Licence |
|---|---|---|
| BrainO | this guide, and every tool's description and parameters | GPL-3.0-or-later |
| BIDS specification 1.11.2 | common principles, dataset description, participants and scans, events, the EEG chapter, derivatives, units, HED | CC-BY-4.0 |
| MNE-Python 1.13.2 | the EEG tutorials (raw data, preprocessing, ICA, referencing, epochs, ERPs, spectra, statistics), the glossary, notes on interpolation and units | BSD-3-Clause |

You can search it yourself:

```bash
braino docs search "why high-pass filter before ICA"
braino docs search "required columns" --source bids --full
braino docs sources        # versions, licences, sizes
```

Search matches keywords, and also meaning, so a question worded differently from the
documentation still finds it: that uses a small embedding model (Qwen3-Embedding 0.6B,
0.6 GB, downloaded by `braino model setup`). It runs on the CPU only while a search needs
it, beside the chat model, and embeds a question in under a second. Without it, search
uses keywords alone.

The index is built in advance from pinned releases of each source and ships with BrainO;
nothing is downloaded or sent anywhere when you search. (Developers: after changing the
guide or a tool's description, `braino docs build` updates it, embedding only the passages
that changed; a test fails until it is current.)

## What the agent may not do

The loop around the model keeps it on track:

- **Only valid tool calls.** The model's every step must match the tools' own parameter
  schemas (see [how the agent talks to the model](model.md#how-the-agent-talks-to-the-model)).
- **No repeated work.** A call identical to an earlier one is not run again while nothing
  has changed (its result still holds), and a call that failed is not retried unchanged
  until something else has run. After two such calls in a row the agent must answer.
- **No invented numbers.** Before the answer reaches you, every number in it is checked
  against what the tools returned. A number no tool gave is sent back to the model to
  correct; if one still remains, it is listed under the answer so you can check it.
- **A step budget.** A request may take up to 60 steps (`--max-steps`); at the last one, or
  when the conversation nearly fills the model's memory, the agent must answer with what it
  has.

## Afterwards

Every tool call, proposal and decision of the session is in its run record:

```bash
braino runs show <run-id> --dataset ./my-study
braino runs export <run-id> --dataset ./my-study -o analysis.py   # a script that redoes it
```

## Another model server

`--server` uses an OpenAI-compatible server that is already running instead of starting
the local model, for example `braino model serve` in another terminal, or Ollama
(`--other-server` for a server that is not llama.cpp):

```bash
braino agent --dataset ./my-study --server http://localhost:11434/v1 --other-server \
    --model-name my-qwen "Compare rare and frequent trials at Pz."
```

## From Python

```python
from pathlib import Path

from braino.agent import ConsoleSession, LLMAgent
from braino.context import RunContext
from braino.llm import ChatClient
from braino.runtime import start_server

with start_server() as server:
    agent = LLMAgent(ChatClient.for_server(server), ctx_size=server.ctx_size)
    session = ConsoleSession(RunContext(dataset_root=Path("my-study")))
    answer = agent.solve("Check the data quality of subject 02.", session)
    print(answer.text)
    answer = agent.follow_up("Now preprocess it.", session)  # same conversation
```

A `Session` decides who answers the agent: `ConsoleSession` asks at the terminal; the
evaluation harness uses a scripted user (`braino eval run --agent llm`, see
[evaluating the agent](evals.md)).
