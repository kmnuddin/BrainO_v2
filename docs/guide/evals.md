# Evaluating the agent

BrainO's AI agent is scored on a fixed set of tasks before it is trusted with anything: every
change to the agent (a new prompt, a different model, a fine-tune) is measured on the same tasks,
on each hardware tier, starting with the minimum computer (no GPU, 16 GB RAM).

```bash
braino eval list                               # the tasks
braino eval run --agent oracle                 # check the harness: should pass everything
braino eval run --agent llm -o today.json      # BrainO's agent on the local model
braino eval run --agent mypkg.agent:make --tier minimum --model qwen-4b -o today.json
braino eval run --agent llm --repeat 3         # each task 3 times, in its other wordings
braino eval compare last-week.json today.json  # what got fixed, what broke
```

## A task

A task is a request in a researcher's words, a dataset with a known answer, and checks on what
the agent did. For example `bad-channels-ask`:

- **Request:** "Find the bad channels in subject 02 and mark them."
- **Dataset:** two synthetic subjects with planted broadband noise on T8.
- **Checks:** preprocessing runs before detection; the agent **asks the user** before channels
  are marked (it cannot approve them itself); the answer names T8; and, because the scripted
  user edits the proposal down to T8, the answer reports what was applied, not what was
  proposed.

Every task also checks that **each number in the answer comes from a tool result**: the plan's
rule that the agent never invents a number, made testable. A number counts as grounded when a
tool reported it (allowing for rounding, percentages and milliseconds) or when it is a setting
the agent ran a tool with ("from 300 to 500 ms"); small counts and numbers from the request
itself are exempt.

The tasks cover dataset inspection, quality control, preprocessing settings in numbers and in
everyday words, decisions (bad channels, dead channels, ICA, approved, edited and rejected),
ERPs (mean amplitudes, peaks and latencies, channel averages), group statistics (choosing a
correction, a cluster test when asked, an honest "not significant" on data with no effect, and
too few subjects), spectra and time-frequency, documentation questions, requests with two
parts, and requests the agent should not simply carry out: a subject or channels the data do
not have, conditions named differently from the events, no subject named, information the
data do not hold, and deleting data. With `--include-large`, an ERP CORE participant's
published P3 must be matched.

## Wordings and repeated runs

A single pass or fail says little about a model. The same model, on the same task, passed on a
CPU and failed on a GPU, whose arithmetic differs slightly; and a model that passes one wording
may fail another. So each task lists its request in other words too (`variants`), and
`--repeat N` runs every task N times, going through its wordings in turn. The report then gives
each task's pass rate, and `braino eval compare` lists the tasks that pass more or less often
than before. On a GPU, three runs of every task take minutes; on the minimum computer (CPU),
one run of every task takes over an hour.

## How a run works

Each task gets a fresh copy of its dataset and a scripted user who answers the agent's questions
and decides on its proposals as the task says (approve, approve with edits, or reject). The
agent works through a `Session` (`braino.agent`): it calls tools, asks the user, and requests
decisions. A task **passes** when all its required checks pass; its **score** is the share of
all its checks that pass. The report (JSON) records the agent, the model, the hardware tier and
the computer, with every check's result; each task's folder keeps the transcript of the
session.

Two agents come with BrainO to check the harness itself:

- `oracle` follows each task's reference solution and must pass every task; if it does not,
  the task is wrong, not an agent.
- `null` does nothing. It fails every task that needs work (its score shows which checks pass
  by doing nothing), and passes only those whose right answer is to decline: reaction times
  the data do not hold, and deleting data.

`llm` is BrainO's own agent ([asking the agent](agent.md)) on the local model, which the run
starts and stops; the report records the model and tier you set up. `--server` scores it on
another OpenAI-compatible server instead, and `--verbose` prints its steps. Each task's
transcript also keeps the model's replies with their timings.

## Adding your own agent

An agent is any object with a `name`, a `model` and a `solve(request, session)` method that
returns an `AgentAnswer`. Pass a factory that makes one:

```python
from braino.agent import AgentAnswer, Session


class MyAgent:
    name = "my-agent"
    model = "qwen3-4b-q4"

    def solve(self, request: str, session: Session) -> AgentAnswer:
        outcome = session.call("dataset.inspect", {})
        return AgentAnswer(text=f"The dataset has {outcome.result['n_subjects']} subjects.")


def make() -> MyAgent:
    return MyAgent()
```

```bash
braino eval run --agent mymodule:make --tier minimum
```

## Training data for fine-tuning

Where the evaluation shows the model getting something wrong that tools and prompts cannot
fix, the model is fine-tuned (for development; users download the finished model).
`braino train data` makes the examples:

```bash
braino train families                         # the kinds of episodes, and their weights
braino train data -n 1000 --seed 1 -o data/   # train.jsonl, valid.jsonl, report.json
```

Each episode is a request with a scripted way to handle it, played through the real agent
loop by a teacher in place of the model: the tools run on synthetic datasets (with planted
problems, other event names, 50 or 60 Hz mains), a scripted user answers and decides, and
the answer is written from the actual results. An example is kept only if its checks pass,
every number in its answer comes from a tool, and nothing went wrong on the way (no error
sent back, no call skipped). The examples are therefore the very conversations the model
sees, with the same system prompt, tool descriptions and steps; when those change, make the
data again.

The evaluation is held out: training requests are worded differently and use other settings,
and a request that matches an evaluation task's wording is never used. The training scripts
(LoRA on the HPC or a local GPU, merging and GGUF conversion) are in `training/` in the
repository.
