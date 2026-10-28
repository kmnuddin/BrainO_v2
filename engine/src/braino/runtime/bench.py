"""A short benchmark that tells the user how fast the agent will be on their computer.

It sends the server one prompt made of BrainO's own tool schemas (what the agent reads on
every step) and asks for a short answer, then turns the measured speeds into the time of an
agent step:

- a **first step** reads ``FIRST_STEP_PROMPT`` new tokens (instructions, the step's tool
  schemas and the request) and writes ``STEP_OUTPUT`` tokens (a tool call or a short answer);
- a **later step** reads only what is new since the last one (``NEXT_STEP_PROMPT`` tokens: a
  tool result), because the server keeps the rest in its prompt cache.

On a CPU, reading is the slow part, which is why the agent sends only the schemas a step needs.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from braino.runtime.server import LlamaServer, ServerError
from braino.runtime.settings import Benchmark, Rating

BENCH_PROMPT_TOKENS = 1024
BENCH_OUTPUT_TOKENS = 48
FIRST_STEP_PROMPT = 2000
NEXT_STEP_PROMPT = 400
STEP_OUTPUT = 120
#: First-step times (s) up to which the agent counts as fast, then usable; beyond, slow.
FAST_S = 15.0
USABLE_S = 60.0


def bench_prompt(tokens: int = BENCH_PROMPT_TOKENS) -> str:
    """About ``tokens`` tokens of realistic agent input: the tool schemas, as the model sees
    them (at roughly four characters a token)."""
    from braino.tools import registry

    text = "You are BrainO's analysis agent. These are your tools:\n"
    text += json.dumps(registry.schemas(), indent=1)
    while len(text) < tokens * 4:
        text += text
    return text[: tokens * 4] + "\nSummarise in one sentence what the first tool does:"


def run_benchmark(server: LlamaServer, model: str, backend: str) -> Benchmark:
    """Measure ``server`` (already started) and estimate agent step times."""
    payload = {
        "prompt": bench_prompt(),
        "n_predict": BENCH_OUTPUT_TOKENS,
        "temperature": 0.0,
        "cache_prompt": False,  # measure reading the whole prompt
        "ignore_eos": True,  # always write the full count, so the speed is comparable
    }
    response = server.post("/completion", payload)
    timings = response.get("timings") if isinstance(response, dict) else None
    if not timings:
        raise ServerError(f"the server's answer has no timings: {str(response)[:200]}")
    prompt_tps = float(timings["prompt_per_second"])
    gen_tps = float(timings["predicted_per_second"])
    if prompt_tps <= 0 or gen_tps <= 0:
        raise ServerError(f"the server reported no speed: {timings}")
    first = FIRST_STEP_PROMPT / prompt_tps + STEP_OUTPUT / gen_tps
    later = NEXT_STEP_PROMPT / prompt_tps + STEP_OUTPUT / gen_tps
    return Benchmark(
        model=model,
        backend=backend,
        placement=server.placement,
        load_seconds=server.load_seconds or 0.0,
        prompt_tokens=int(timings["prompt_n"]),
        prompt_per_second=round(prompt_tps, 1),
        generated_tokens=int(timings["predicted_n"]),
        generated_per_second=round(gen_tps, 1),
        first_step_seconds=round(first, 1),
        next_step_seconds=round(later, 1),
        rating=rate(first),
        measured_at=datetime.now(UTC),
    )


def rate(first_step_seconds: float) -> Rating:
    if first_step_seconds <= FAST_S:
        return "fast"
    if first_step_seconds <= USABLE_S:
        return "usable"
    return "slow"


def describe(result: Benchmark) -> str:
    """The benchmark in a sentence or two, for the user."""
    verdict = {
        "fast": "The agent will feel quick on this computer.",
        "usable": "The agent is usable on this computer; expect short waits.",
        "slow": (
            "The agent will be slow on this computer; a smaller model, a GPU or closing "
            "other programs would help. The analysis tools themselves are not affected."
        ),
    }[result.rating]
    return (
        f"Reads {result.prompt_per_second:g} tokens/s and writes "
        f"{result.generated_per_second:g} tokens/s. A request's first step takes about "
        f"{result.first_step_seconds:g} s and each later step about "
        f"{result.next_step_seconds:g} s (model loaded in {result.load_seconds:g} s). {verdict}"
    )
