"""Training data: episodes played through the agent loop by a teacher, held out from the evals."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from braino.cli import app
from braino.evals.tasks import Check, DatasetSpec, SyntheticData, builtin_tasks
from braino.llm.steps import step_schema
from braino.tools import registry
from braino.training import (
    FAMILIES,
    Call,
    Episode,
    Example,
    Rejected,
    Say,
    build,
    check_held_out,
    run_episode,
    split,
)
from braino.training.episodes import _options

runner = CliRunner()
SMALL = DatasetSpec(synthetic=SyntheticData(subjects=["01", "02"], seconds=30))


def test_the_teacher_reads_what_a_step_allows() -> None:
    tools = [registry.get("dataset.inspect"), registry.get("eeg.qc_report")]
    schema = step_schema(tools, ["eeg.preprocess"], ["p-1"])
    assert _options(schema) == ({"dataset.inspect", "eeg.qc_report"}, {"p-1"})


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_families_make_valid_episodes(name: str) -> None:
    for seed in range(25):
        episode = FAMILIES[name](random.Random(seed))
        assert episode.request and episode.script and episode.checks
        assert isinstance(episode.script[-1], Say)
        episode.task()  # the checks and dataset validate as an evaluation task


def test_the_evaluation_is_held_out() -> None:
    task = builtin_tasks()[0]
    assert check_held_out([task.request, "something else entirely"]) == [task.request]
    assert check_held_out([task.request.upper() + "!"]) == [task.request.upper() + "!"]
    requests = [
        FAMILIES[name](random.Random(seed)).request for name in FAMILIES for seed in range(60)
    ]
    assert check_held_out(requests) == []


# --- playing episodes (needs the EEG stack for the synthetic data) ---------------------------


def test_an_episode_becomes_the_conversation_the_model_sees(tmp_path: Path) -> None:
    pytest.importorskip("mne")
    episode = FAMILIES["inspect"](random.Random(3))
    example = run_episode(episode, tmp_path / "ep")
    assert isinstance(example, Example), example
    roles = [m.role for m in example.messages]
    assert roles[:2] == ["system", "user"] and example.messages[1].content == episode.request
    steps = [m for m in example.messages if m.role == "assistant"]
    # The tools loaded from the request's words were not written by the model.
    assert steps[0].weight == 0 and json.loads(steps[0].content)["action"] == "load_tools"
    assert all(m.weight == 1 for m in steps[1:])
    assert json.loads(steps[-1].content)["action"] == "answer"
    assert json.loads(steps[1].content) == {
        "action": "call",
        "tool": "dataset.inspect",
        "arguments": {"max_recordings": 20},
    }


def test_a_tool_that_is_not_loaded_is_loaded_first(tmp_path: Path) -> None:
    pytest.importorskip("mne")
    episode = Episode(
        family="t",
        request="Tell me something.",  # preloads dataset.inspect only
        dataset=SMALL,
        script=[
            Call("system.info", {}, id="sys"),
            Say(lambda r: f"BrainO {r['sys']['braino_version']}."),
        ],
        checks=[Check(kind="tool_called", tool="system.info")],
        grounded=False,
    )
    example = run_episode(episode, tmp_path / "ep")
    assert isinstance(example, Example), example
    actions = [json.loads(m.content) for m in example.messages if m.weight == 1]
    assert [a["action"] for a in actions] == ["load_tools", "call", "answer"]
    assert actions[0]["tools"] == ["system.info"]


def test_episodes_that_would_teach_a_mistake_are_refused(tmp_path: Path) -> None:
    pytest.importorskip("mne")
    invented = Episode(
        family="t",
        request="How many subjects are there?",
        dataset=SMALL,
        script=[Call("dataset.inspect", {}, id="ds"), Say("There are 42 subjects.")],
        checks=[Check(kind="tool_called", tool="dataset.inspect")],
    )
    out = run_episode(invented, tmp_path / "a")
    assert isinstance(out, Rejected) and "42" in out.reason

    failing = Episode(
        family="t",
        request="How many subjects are there?",
        dataset=SMALL,
        script=[Call("dataset.inspect", {}, id="ds"), Say("There are two.")],
        checks=[Check(kind="tool_called", tool="eeg.qc_report")],
    )
    out = run_episode(failing, tmp_path / "b")
    assert isinstance(out, Rejected) and "checks failed" in out.reason

    broken = Episode(
        family="t",
        request="Preprocess subject 01.",
        dataset=SMALL,
        script=[Call("eeg.preprocess", {"subject": "07"}), Say("Done.")],
        checks=[Check(kind="tool_called", tool="eeg.preprocess")],
    )
    out = run_episode(broken, tmp_path / "c")
    assert isinstance(out, Rejected) and "eeg.preprocess failed" in out.reason


def test_an_expected_error_is_part_of_the_episode(tmp_path: Path) -> None:
    pytest.importorskip("mne")
    episode = FAMILIES["missing-reference"](random.Random(0))
    example = run_episode(episode, tmp_path / "ep")
    assert isinstance(example, Example), example
    results = [m.content for m in example.messages if m.content.startswith("<tool_response>")]
    assert any("reference channels not among the EEG channels" in r for r in results)


def test_build_and_cli(tmp_path: Path) -> None:
    pytest.importorskip("mne")
    examples, report = build(4, seed=1, families=["refuse", "inspect"], workdir=tmp_path / "w")
    assert len(examples) == 4 and sum(report.kept.values()) == 4 and not report.rejected
    train, valid = split(examples, 0.25)
    assert len(train) == 3 and len(valid) == 1

    out = tmp_path / "data"
    result = runner.invoke(
        app,
        ["train", "data", "-o", str(out), "-n", "3", "--families", "refuse,system",
         "--workdir", str(tmp_path / "w2")],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    lines = (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    lines += (out / "valid.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert {json.loads(x)["family"] for x in lines} <= {"refuse", "system"}
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["attempted"] == 3
    unknown = runner.invoke(app, ["train", "data", "-o", str(out), "--families", "nope"])
    assert unknown.exit_code != 0 and "no family nope" in unknown.output


def test_synthetic_event_names_and_mains() -> None:
    pytest.importorskip("mne")
    from braino.datasets.synthetic import make_raw

    raw = make_raw(20, events={"rare": "Stimulus/S  2"}, line_freq=60.0)
    assert set(raw.annotations.description) == {"Stimulus/S  2", "stimulus/frequent"}
    assert raw.info["line_freq"] == 60.0
