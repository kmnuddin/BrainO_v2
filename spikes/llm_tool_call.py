"""Spike: can a local Qwen model call BrainO tools through the OpenAI-compatible API?

Sends the tool schemas to a llama.cpp server, runs whatever tool calls the model makes through
a registry, feeds the results (or errors) back and prints the transcript. Besides the built-in
``system.info``, a spike-only registry adds demo tools over a fixed fake recording: two read
tools that take arguments and a decision tool, whose proposal the script (standing in for the
user) approves with ``decide()``. The model is never given ``decide()``.

Each case checks which tools were called, with which arguments, what the answer says, and that
every decimal number in the answer came from the prompt or a tool result.

    llama-server -m Qwen3.5-9B-Q4_K_M.gguf --jinja -ngl 99 -c 8192 --port 8080
    python spikes/llm_tool_call.py [--base-url http://127.0.0.1:8080/v1]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal

from braino.tools import PendingProposal, Risk, RunContext, ToolRegistry, decide, registry
from pydantic import BaseModel, Field

SYSTEM_PROMPT = (
    "You are the BrainO analysis assistant. Use the provided tools to answer questions about "
    "the BrainO installation and the loaded EEG recording. Never state a version, measurement "
    "or other value that did not come from a tool result."
)

# A fake 11-channel recording. T7 is noisy, C4 has line noise and Oz is flat.
CHANNELS: dict[str, dict[str, float]] = {
    "Fp1": {"rms_uv": 24.6, "line_noise_db": 2.1, "flat_fraction": 0.0},
    "Fp2": {"rms_uv": 23.9, "line_noise_db": 2.4, "flat_fraction": 0.0},
    "F3": {"rms_uv": 14.2, "line_noise_db": 1.8, "flat_fraction": 0.0},
    "Fz": {"rms_uv": 13.7, "line_noise_db": 1.6, "flat_fraction": 0.0},
    "F4": {"rms_uv": 14.9, "line_noise_db": 2.0, "flat_fraction": 0.0},
    "T7": {"rms_uv": 92.3, "line_noise_db": 3.2, "flat_fraction": 0.0},
    "C3": {"rms_uv": 12.8, "line_noise_db": 1.9, "flat_fraction": 0.0},
    "Cz": {"rms_uv": 11.5, "line_noise_db": 1.4, "flat_fraction": 0.0},
    "C4": {"rms_uv": 16.1, "line_noise_db": 17.8, "flat_fraction": 0.0},
    "Pz": {"rms_uv": 12.3, "line_noise_db": 1.7, "flat_fraction": 0.0},
    "Oz": {"rms_uv": 0.4, "line_noise_db": 0.2, "flat_fraction": 0.97},
}

spike_registry = ToolRegistry()
spike_registry.register(registry.get("system.info"))


class ListChannelsInput(BaseModel):
    pass


class ListChannelsOutput(BaseModel):
    channels: list[str]
    sampling_rate_hz: float


@spike_registry.tool(name="demo.list_channels", risk=Risk.READ)
def list_channels(params: ListChannelsInput) -> ListChannelsOutput:
    """List the EEG channels of the loaded recording and its sampling rate."""
    return ListChannelsOutput(channels=list(CHANNELS), sampling_rate_hz=500.0)


Metric = Literal["rms_uv", "line_noise_db", "flat_fraction"]


class ChannelStatsInput(BaseModel):
    channels: list[str] = Field(description="Channel names, e.g. ['Fz', 'Cz']")
    metric: Metric = Field(
        description="rms_uv: RMS amplitude in microvolts; line_noise_db: power at 60 Hz above "
        "the neighbouring frequencies; flat_fraction: fraction of the recording that is flat"
    )


class ChannelStatsOutput(BaseModel):
    metric: Metric
    values: dict[str, float]


@spike_registry.tool(name="demo.channel_stats", risk=Risk.READ)
def channel_stats(params: ChannelStatsInput) -> ChannelStatsOutput:
    """Measure one quality metric on some channels of the loaded recording."""
    unknown = [ch for ch in params.channels if ch not in CHANNELS]
    if unknown:
        raise ValueError(f"unknown channels {unknown}; available: {list(CHANNELS)}")
    return ChannelStatsOutput(
        metric=params.metric, values={ch: CHANNELS[ch][params.metric] for ch in params.channels}
    )


class BadChannelsInput(BaseModel):
    max_rms_uv: float = Field(default=50.0, description="Channels louder than this are bad")
    max_line_noise_db: float = Field(default=10.0, description="Line noise above this is bad")
    max_flat_fraction: float = Field(default=0.5, description="Flatter than this is bad")


class BadChannelsProposal(BaseModel):
    channels: list[str] = Field(description="Channels proposed for removal")
    reasons: dict[str, str]


class DropResult(BaseModel):
    dropped: list[str]
    remaining: list[str]


def drop_channels(proposal: BadChannelsProposal, ctx: RunContext) -> DropResult:
    return DropResult(
        dropped=proposal.channels, remaining=[ch for ch in CHANNELS if ch not in proposal.channels]
    )


@spike_registry.tool(name="demo.propose_bad_channels", risk=Risk.DECISION, apply=drop_channels)
def propose_bad_channels(params: BadChannelsInput) -> BadChannelsProposal:
    """Find bad channels in the loaded recording and propose dropping them."""
    reasons: dict[str, str] = {}
    for ch, stats in CHANNELS.items():
        if stats["rms_uv"] > params.max_rms_uv:
            reasons[ch] = f"RMS {stats['rms_uv']} uV > {params.max_rms_uv}"
        elif stats["line_noise_db"] > params.max_line_noise_db:
            reasons[ch] = f"line noise {stats['line_noise_db']} dB > {params.max_line_noise_db}"
        elif stats["flat_fraction"] > params.max_flat_fraction:
            reasons[ch] = f"flat {stats['flat_fraction']} > {params.max_flat_fraction}"
    return BadChannelsProposal(channels=list(reasons), reasons=reasons)


@dataclass
class Case:
    prompt: str
    expected_tools: list[str]  # BrainO tool names that must be called; [] means no tool at all
    expected_args: dict[str, dict[str, Any]] = field(default_factory=dict)  # per tool, subset
    answer_contains: list[str] = field(default_factory=list)  # case-insensitive
    approve: bool = False  # approve any proposal afterwards, as the user would


CASES = [
    Case("What version of BrainO is installed?", ["system.info"], answer_contains=["0.1.0"]),
    Case("Which Python version and operating system is the engine running on?", ["system.info"]),
    Case(
        "Is this BrainO install new enough to have Python 3.11 support? Check first.",
        ["system.info"],
    ),
    Case("In one sentence, what is an ERP in EEG research?", []),
    Case(
        "What is the RMS amplitude of Fz and Cz?",
        ["demo.channel_stats"],
        {"demo.channel_stats": {"channels": ["Fz", "Cz"], "metric": "rms_uv"}},
        answer_contains=["13.7", "11.5"],
    ),
    Case(
        "Which channel has the most line noise?",
        ["demo.channel_stats"],
        {"demo.channel_stats": {"metric": "line_noise_db"}},
        answer_contains=["C4", "17.8"],
    ),
    Case(
        "What is the RMS amplitude of channel Fpz?", ["demo.channel_stats"], answer_contains=["Fpz"]
    ),
    Case(
        "Find the bad channels in this recording and drop them.",
        ["demo.propose_bad_channels"],
        answer_contains=["T7", "C4", "Oz", "approv"],
        approve=True,
    ),
    Case(
        "Propose dropping bad channels, but only treat channels with RMS above 80 microvolts as "
        "too loud.",
        ["demo.propose_bad_channels"],
        {"demo.propose_bad_channels": {"max_rms_uv": 80}},
        answer_contains=["approv"],
    ),
]


def chat(base_url: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
    body = json.dumps({"messages": messages, "tools": tools, "temperature": 0}).encode()
    request = urllib.request.Request(
        f"{base_url}/chat/completions", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def call_tool(name: str, arguments: dict[str, Any], ctx: RunContext) -> tuple[str, str]:
    """Run a tool the model asked for; returns (BrainO tool name, JSON result or error)."""
    try:
        tool = spike_registry.get(name)
        name = tool.name
        return name, tool.run(arguments, ctx).model_dump_json()
    except Exception as exc:  # errors go back to the model, as they will in the agent
        return name, json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for key, value in expected.items():
        got = actual.get(key)
        if isinstance(value, list):
            if sorted(value) != sorted(got or []):
                return False
        elif got != value:
            return False
    return True


def run_case(base_url: str, case: Case, max_turns: int = 6) -> bool:
    tools = spike_registry.schemas()
    ctx = RunContext()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": case.prompt},
    ]
    calls_made: list[tuple[str, dict[str, Any]]] = []
    seen_text = case.prompt
    proposals: list[str] = []
    start = time.perf_counter()
    print(f"\n=== {case.prompt}")
    for _ in range(max_turns):
        message = chat(base_url, messages, tools)["choices"][0]["message"]
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            break
        for call in calls:
            arguments = json.loads(call["function"]["arguments"] or "{}")
            name, result = call_tool(call["function"]["name"], arguments, ctx)
            calls_made.append((name, arguments))
            seen_text += result
            if '"status":"pending"' in result:
                proposals.append(PendingProposal.model_validate_json(result).proposal_id)
            print(f"  tool call: {name}({arguments}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})
    answer = (message.get("content") or "").strip()
    print(f"  answer ({time.perf_counter() - start:.1f}s): {answer}")

    problems = []
    called = [name for name, _ in calls_made]
    if not case.expected_tools and called:
        problems.append(f"expected no tool, called {called}")
    problems += [f"did not call {t}" for t in case.expected_tools if t not in called]
    for tool, expected in case.expected_args.items():
        if not any(n == tool and args_match(expected, a) for n, a in calls_made):
            problems.append(f"{tool} never called with {expected}")
    problems += [
        f"answer lacks {s!r}" for s in case.answer_contains if s.lower() not in answer.lower()
    ]
    invented = [n for n in re.findall(r"\d+\.\d+(?:\.\d+)*", answer) if n not in seen_text]
    if invented:
        problems.append(f"numbers not from tools: {invented}")
    if not answer:
        problems.append("empty answer")

    if case.approve:
        if not proposals:
            problems.append("no proposal to approve")
        for proposal_id in proposals:
            record = decide(
                proposal_id, ctx, approve=True, decided_by="user:spike", tools=spike_registry
            )
            print(f"  user approved {proposal_id} -> {record.decision.apply_result}")  # type: ignore[union-attr]

    print(f"  {'PASS' if not problems else 'FAIL: ' + '; '.join(problems)}")
    return not problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    args = parser.parse_args()
    print("tools sent:", [t["function"]["name"] for t in spike_registry.schemas()])
    results = [run_case(args.base_url, case) for case in CASES]
    print(f"\n{sum(results)}/{len(results)} cases passed")


if __name__ == "__main__":
    main()
