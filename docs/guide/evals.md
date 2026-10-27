# Evaluating the agent

BrainO's AI agent is scored on a fixed set of tasks before it is trusted with anything: every
change to the agent (a new prompt, a different model, a fine-tune) is measured on the same tasks,
on each hardware tier, starting with the minimum computer (no GPU, 16 GB RAM).

```bash
braino eval list                               # the tasks
braino eval run --agent oracle                 # check the harness: should pass everything
braino eval run --agent mypkg.agent:make --tier minimum --model qwen-4b -o today.json
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
tool reported it (allowing for rounding, percentages and milliseconds); small counts and
numbers from the request itself are exempt.

The tasks cover dataset inspection, quality control, preprocessing settings, decisions (bad
channels, rejection, ICA), ERPs, group statistics (including choosing a multiple-comparison
correction), spectra, refusing to guess, and, with `--include-large`, an ERP CORE participant
whose published P3 the answer must match.

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
- `null` does nothing, and must fail every task (its score shows which checks pass by doing
  nothing).

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
