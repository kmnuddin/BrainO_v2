"""Templates and plan checks: rules on whole specs, specs built from templates, the analysis
tools (plan, approve, run) and the agent working through them."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from scripted_llm import ScriptedServer
from typer.testing import CliRunner

from braino.agent import LLMAgent, Session
from braino.cli import app
from braino.pipelines import PipelineSpec, builtin_specs, load_spec
from braino.pipelines.checks import check_plan
from braino.pipelines.templates import (
    ErpComparisonSettings,
    SpectralComparisonSettings,
    TemplateError,
    build,
    get_template,
)
from braino.tools import RunContext, decide, registry
from braino.tools.analysis import AnalysisResult

RARE: dict[str, Any] = {"condition_a": "rare", "events_a": ["stimulus/rare"]}
FREQUENT: dict[str, Any] = {"condition_b": "frequent", "events_b": ["stimulus/frequent"]}
ERP: dict[str, Any] = {
    **RARE,
    **FREQUENT,
    "test_scope": "measured",
    "channels": ["Pz"],
    "tmin": 0.3,
    "tmax": 0.5,
}
FAST: dict[str, Any] = {"reject": "peak_to_peak", "peak_to_peak_uv": 100.0}


def spec(steps: list[dict[str, Any]], group: list[dict[str, Any]] | None = None,
         **fields: Any) -> PipelineSpec:  # fmt: skip
    return PipelineSpec.model_validate(
        {"name": "t", "task": "oddball", "steps": steps, "group": group or [], **fields}
    )


def messages(s: PipelineSpec, **kwargs: Any) -> list[str]:
    return [str(f) for f in check_plan(s, **kwargs).findings]


PRE: dict[str, Any] = {"id": "pre", "tool": "eeg.preprocess"}
EPOCH: dict[str, Any] = {
    "id": "ep",
    "tool": "eeg.epoch",
    "args": {"conditions": {"rare": ["stimulus/rare"], "frequent": ["stimulus/frequent"]}},
}
ERP_STEP: dict[str, Any] = {
    "id": "erp",
    "tool": "eeg.erp",
    "args": {"contrasts": [{"name": "diff", "plus": "rare", "minus": "frequent"}]},
}


# --- plan checks on the spec alone --------------------------------------------------------


@pytest.mark.parametrize("name", sorted(builtin_specs()))
def test_builtin_specs_pass_the_plan_checks(name: str) -> None:
    assert check_plan(load_spec(name)).findings == []


def test_order() -> None:
    found = messages(
        spec(
            [
                {"id": "erp", "tool": "eeg.erp"},
                EPOCH,
                {"id": "ica", "tool": "eeg.ica_clean"},
            ],
            [
                {
                    "id": "g",
                    "tool": "stats.mass_univariate",
                    "args": {"data": "psd", "conditions": ["rare"]},
                }
            ],
        )
    )
    assert "error: erp: eeg.erp needs eeg.preprocess before it" in found
    assert "error: erp: eeg.erp reads epochs: put eeg.epoch before it" in found
    assert any("ica: eeg.ica_clean comes after eeg.epoch" in f for f in found)
    assert any("ep: eeg.epoch needs eeg.preprocess" in f for f in found)
    assert any(f.startswith("error: g:") and "no step runs eeg.spectral" in f for f in found)
    assert check_plan(spec([PRE, EPOCH, ERP_STEP])).ok


def test_names_and_windows() -> None:
    epoch = {**EPOCH, "args": {**EPOCH["args"], "baseline": [-0.5, 0.0]}}
    erp = {
        "id": "erp",
        "tool": "eeg.erp",
        "args": {
            "contrasts": [{"name": "diff", "plus": "target", "minus": "frequent"}],
            "measures": [{"name": "P3", "channels": ["Pz"], "tmin": 0.3, "tmax": 1.2}],
        },
    }
    found = messages(
        spec(
            [PRE, epoch, erp, {"id": "tfr", "tool": "eeg.tfr", "args": {"label": "other"}}],
            [{"id": "g", "tool": "stats.mass_univariate", "args": {"conditions": ["d"]}}],
        )
    )
    assert any("baseline [-0.5, 0.0] is not inside the epoch [-0.2, 0.8]" in f for f in found)
    assert any("no condition 'target'; defined: frequent, rare" in f for f in found)
    assert any("measure 'P3': the window [0.3, 1.2] s is not inside" in f for f in found)
    assert any("no epochs labelled 'other'" in f for f in found)
    assert any("no condition or contrast 'd'; defined: diff, frequent, rare" in f for f in found)


def test_statistics() -> None:
    def group(tool: str, **args: Any) -> list[dict[str, Any]]:
        return [{"id": "g", "tool": tool, "args": {"conditions": ["diff"], **args}}]

    uncorrected = spec([PRE, EPOCH, ERP_STEP], group("stats.mass_univariate", correction="none"))
    [finding] = check_plan(uncorrected).findings
    assert finding.severity == "warning" and "no correction for multiple" in finding.message
    few = spec([PRE, EPOCH, ERP_STEP], group("stats.cluster_permutation"), subjects=list("1234"))
    assert any("only 16 sign-flip permutations" in f for f in messages(few))
    two = spec([PRE, EPOCH, ERP_STEP], group("stats.mass_univariate"), subjects=["01", "02"])
    assert any("needs at least 3 subjects" in f for f in messages(two))


# --- templates ----------------------------------------------------------------------------


def test_erp_template_steps() -> None:
    template = get_template("erp-comparison")
    settings = ErpComparisonSettings.model_validate({**ERP, "ica": True})
    one = template.build(settings, "oddball", ["01"])
    assert one.name == "erp-rare-vs-frequent" and one.group == []
    assert [s.tool for s in one.steps] == [
        "eeg.preprocess",
        "eeg.ica_clean",
        "eeg.epoch",
        "eeg.erp",
    ]
    erp = one.step("erp").args
    assert erp["contrasts"] == [
        {"name": "rare_minus_frequent", "plus": "rare", "minus": "frequent"}
    ]
    assert erp["measures"][0] == {"name": "amplitude", "channels": ["Pz"], "tmin": 0.3,
                                  "tmax": 0.5, "kind": "mean"}  # fmt: skip
    assert template.build(settings, "oddball", ["01", "02"]).group == []
    [group] = template.build(settings, "oddball", ["01", "02", "03"]).group
    assert (group.tool, group.args) == (
        "stats.mass_univariate",
        {"conditions": ["rare_minus_frequent"], "channels": ["Pz"], "tmin": 0.3, "tmax": 0.5,
         "correction": "fdr"},
    )  # fmt: skip
    everywhere = ErpComparisonSettings.model_validate({**ERP, "test_scope": "all"})
    [group] = template.build(everywhere, "oddball", list("123")).group
    assert group.args == {"conditions": ["rare_minus_frequent"], "tmin": -0.2, "tmax": 0.8,
                          "correction": "fdr"}  # fmt: skip
    cluster = ErpComparisonSettings.model_validate(
        {**ERP, "test": "cluster_permutation", "test_scope": "all"}
    )
    [group] = template.build(cluster, "oddball", list("123")).group
    assert (group.tool, group.args) == (
        "stats.cluster_permutation",
        {"conditions": ["rare_minus_frequent"], "tmin": -0.2, "tmax": 0.8},
    )
    for s in (settings, everywhere, cluster):
        assert check_plan(template.build(s, "oddball", list("12345678"))).ok


def test_spectral_template_steps() -> None:
    template = get_template("spectral-comparison")
    settings = SpectralComparisonSettings.model_validate({**RARE, **FREQUENT, "band": "theta",
                                                          "fmin": 4, "fmax": 8})  # fmt: skip
    s = template.build(settings, "oddball", ["01", "02", "03"])
    assert s.step("spectra").args["bands"] == {"theta": [4.0, 8.0]}
    assert s.step("epochs").args["baseline"] is None
    assert s.group[0].args == {"data": "psd", "conditions": ["rare", "frequent"], "fmin": 4.0,
                               "fmax": 8.0, "correction": "fdr"}  # fmt: skip
    assert check_plan(s).ok


def test_settings_are_checked() -> None:
    with pytest.raises(ValidationError, match="different names"):
        ErpComparisonSettings.model_validate({**ERP, "condition_b": "rare"})
    with pytest.raises(ValidationError, match="in both conditions"):
        ErpComparisonSettings.model_validate({**ERP, "events_b": ["stimulus/rare"]})
    with pytest.raises(ValidationError, match="tmin must be before tmax"):
        ErpComparisonSettings.model_validate({**ERP, "tmin": 0.5, "tmax": 0.3})
    with pytest.raises(ValidationError, match="above the low-pass"):
        SpectralComparisonSettings.model_validate({**RARE, **FREQUENT, "fmax": 60})
    with pytest.raises(TemplateError, match="no template 'nope'"):
        get_template("nope")


# --- with a dataset -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def original(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("mne")
    pytest.importorskip("mne_bids")
    pytest.importorskip("pybv")
    from braino.datasets.synthetic import write_dataset

    root = tmp_path_factory.mktemp("templates") / "bids"
    return write_dataset(root, ["01", "02", "03", "04"], task="oddball", seconds=40.0)


@pytest.fixture
def dataset(original: Path, tmp_path: Path) -> Path:
    """A fresh copy, so each test's results start from nothing."""
    return Path(shutil.copytree(original, tmp_path / "bids"))


def test_plan_checks_against_the_data(original: Path) -> None:
    epoch = {
        "id": "ep",
        "tool": "eeg.epoch",
        "args": {"conditions": {"rare": ["stimulus/rar"], "frequent": ["stimulus/frequent",
                                                                       "stimulus/x"]}},
    }  # fmt: skip
    erp = {
        "id": "erp",
        "tool": "eeg.erp",
        "args": {"measures": [{"name": "P3", "channels": ["Pzz"], "tmin": 0.3, "tmax": 0.5}]},
    }
    pre = {"id": "pre", "tool": "eeg.preprocess",
           "args": {"resample_hz": 50, "reference": ["M1"]}}  # fmt: skip
    found = messages(spec([pre, epoch, erp], subjects=["01", "09"]), dataset=original)
    assert any("no oddball recording for subject(s) 09" in f for f in found)
    assert any("condition 'rare': none of its events (stimulus/rar) occur" in f
               and "stimulus/rare" in f for f in found)  # fmt: skip
    assert "warning: ep: condition 'frequent': events stimulus/x do not occur" in found
    assert any("the low-pass at 40 Hz is not below half the sampling rate (50 Hz)" in f
               for f in found)  # fmt: skip
    assert "error: pre: reference: no channel M1 in the recordings" in found
    assert "error: erp: measure 'P3': no channel Pzz in the recordings" in found
    good = build("erp-comparison", ERP, original)
    assert good.subjects == ["01", "02", "03", "04"]
    assert check_plan(good, dataset=original).findings == []


def test_template_task_and_subjects(original: Path) -> None:
    assert build("erp-comparison", {**ERP, "subjects": ["02"]}, original).task == "oddball"
    with pytest.raises(TemplateError, match="no recording of task 'rest'"):
        build("erp-comparison", {**ERP, "task": "rest"}, original)
    with pytest.raises(TemplateError, match=r"subject\(s\) 07"):
        build("erp-comparison", {**ERP, "subjects": ["07"]}, original)


# --- the analysis tools -------------------------------------------------------------------


def _plan(ctx: RunContext, tool: str = "analysis.erp_comparison", **settings: Any) -> Any:
    return registry.get(tool).run(settings, ctx)


def _approve(ctx: RunContext, proposal_id: str, **edits: Any) -> dict[str, Any]:
    """Approve a proposal; what applying it returned."""
    record = decide(proposal_id, ctx, approve=True, decided_by="test:user", edits=edits)
    assert record.decision is not None and record.decision.apply_result is not None
    return record.decision.apply_result


def _run(ctx: RunContext, name: str) -> AnalysisResult:
    result = registry.get("analysis.run").run({"name": name}, ctx)
    assert isinstance(result, AnalysisResult)
    return result


def test_plan_approve_and_run(dataset: Path) -> None:
    ctx = RunContext(dataset_root=dataset, run_id="r1")
    proposal = _plan(ctx, **ERP, **FAST)
    p = proposal.proposal
    assert p["name"] == "erp-rare-vs-frequent" and p["subjects"] == ["01", "02", "03", "04"]
    assert p["steps"][0] == 'eeg.preprocess(l_freq=0.1, h_freq=30.0, reference="average")'
    assert p["group_steps"][0].startswith("stats.mass_univariate(")
    assert p["warnings"] == []

    with pytest.raises(Exception, match="no approved plan 'erp-rare-vs-frequent'"):
        _run(ctx, "erp-rare-vs-frequent")
    saved = _approve(ctx, proposal.proposal_id, subjects=["01", "02", "03"])
    assert saved["spec"] == "derivatives/braino/pipelines/erp-rare-vs-frequent/spec.yaml"
    assert load_spec(dataset / saved["spec"]).subjects == ["01", "02", "03"]

    result = _run(ctx, saved["name"])
    assert result.status == "complete" and result.n_subjects == 3 and result.unit == "µV"
    assert [row["subject"] for row in result.per_subject] == ["01", "02", "03"]
    assert set(result.mean) == {"rare", "frequent", "rare_minus_frequent"}
    assert result.mean["rare_minus_frequent"] > 2  # the planted P3
    group = result.group_test
    assert group is not None and group["n_significant"] > 0 and group["peak_channel"] == "Pz"
    tools = [c.tool for c in ctx.provenance.calls]
    assert tools[:2] == ["analysis.erp_comparison", "analysis.erp_comparison.apply"]
    assert tools.count("eeg.erp") == 3 and tools[-1] == "stats.mass_univariate"

    again = _run(ctx, saved["name"])
    assert again.status == "complete" and again.mean == result.mean
    assert len(ctx.provenance.calls) == len(tools)  # every step reused


def test_a_plan_that_does_not_fit_the_data_fails(dataset: Path) -> None:
    ctx = RunContext(dataset_root=dataset, run_id="r1")
    with pytest.raises(TemplateError, match=r"none of its events \(stimulus/x\) occur"):
        _plan(ctx, **{**ERP, "events_a": ["stimulus/x"]})
    with pytest.raises(TemplateError, match="no channel Cz9"):
        _plan(ctx, **{**ERP, "channels": ["Cz9"]})


def test_decision_steps_in_a_plan_wait_for_the_user(dataset: Path) -> None:
    pytest.importorskip("pyprep")
    ctx = RunContext(dataset_root=dataset, run_id="r1")
    proposal = _plan(ctx, **ERP, **FAST, subjects=["01", "02", "03"], bad_channels=True)
    name = _approve(ctx, proposal.proposal_id)["name"]
    first = _run(ctx, name)
    assert first.status == "waiting" and first.group_test is None
    assert [(w.subject, w.tool) for w in first.waiting] == [
        ("01", "eeg.detect_bad_channels"),
        ("02", "eeg.detect_bad_channels"),
        ("03", "eeg.detect_bad_channels"),
    ]
    assert first.next is not None and "call analysis.run" in first.next
    for w in first.waiting:
        _approve(ctx, w.proposal_id)
    done = _run(ctx, name)
    assert done.status == "complete" and done.group_test is not None


def test_spectral_comparison_runs(dataset: Path) -> None:
    ctx = RunContext(dataset_root=dataset, run_id="r1")
    proposal = _plan(ctx, "analysis.spectral_comparison", **RARE, **FREQUENT, **FAST,
                     subjects=["01", "02", "03"])  # fmt: skip
    result = _run(ctx, _approve(ctx, proposal.proposal_id)["name"])
    assert result.status == "complete" and result.unit == "µV²"
    assert set(result.per_subject[0]) == {"subject", "rare", "frequent", "epochs_kept"}
    assert result.group_test is not None and result.group_test["n_tests"] > 0


# --- the agent ----------------------------------------------------------------------------


class ApprovingSession(Session):
    def __init__(self, root: Path) -> None:
        super().__init__(RunContext(dataset_root=root, run_id="r1"))
        self.shown: list[str] = []

    @property
    def user_name(self) -> str:
        return "test:user"

    def _decide(
        self, tool: str, proposal: dict[str, Any], message: str
    ) -> tuple[bool, dict[str, Any], str | None]:
        self.shown.append(tool)
        return True, {}, None

    def _reply(self, question: str) -> str:
        return "No preference."


@pytest.fixture
def server() -> Iterator[ScriptedServer]:
    s = ScriptedServer()
    yield s
    s.close()


def _call(tool: str, **arguments: Any) -> dict[str, Any]:
    return {"action": "call", "tool": tool, "arguments": arguments}


def test_the_agent_plans_has_it_approved_and_runs_it(server: ScriptedServer, dataset: Path) -> None:
    pytest.importorskip("pyprep")
    settings = {**ERP, **FAST, "subjects": ["01", "02", "03"], "bad_channels": True}
    server.replies += [
        _call("analysis.erp_comparison", **settings),
        {"action": "decide", "proposal_id": "r1-1", "message": "The plan."},
        _call("analysis.run", name="erp-rare-vs-frequent"),
        # analysis.run: preprocess and bad channels for each subject (calls 3 to 8)
        {"action": "decide", "proposal_id": "r1-4", "message": "Bad channels of 01."},
        {"action": "decide", "proposal_id": "r1-6", "message": "Bad channels of 02."},
        {"action": "decide", "proposal_id": "r1-8", "message": "Bad channels of 03."},
        _call("analysis.run", name="erp-rare-vs-frequent"),
        {"action": "answer", "text": "The difference is significant at Pz."},
    ]
    session = ApprovingSession(dataset)
    request = "Across all subjects, compare rare minus frequent trials at Pz."
    answer = LLMAgent(server.client()).solve(request, session)

    assert answer.text == "The difference is significant at Pz." and answer.stopped is None
    assert session.shown == [
        "analysis.erp_comparison",
        "eeg.detect_bad_channels",
        "eeg.detect_bad_channels",
        "eeg.detect_bad_channels",
    ]
    first = server.requests[0]["messages"]
    assert "analysis.erp_comparison" in first[2]["content"]  # loaded for the first step
    top = [c.tool for c in session.calls if c.within is None]
    assert top == ["analysis.erp_comparison", "analysis.run", "analysis.run"]
    steps = [c.tool for c in session.calls if c.within == "analysis.run"]
    assert steps.count("eeg.erp") == 3 and steps[-1] == "stats.mass_univariate"
    # the waiting decisions were in the schema of the step after the first run
    decide_options = json.dumps(server.requests[3]["response_format"])
    assert all(f'"r1-{i}"' in decide_options for i in (4, 6, 8))


def test_the_console_shows_a_plan_one_step_per_line(tmp_path: Path) -> None:
    from braino.agent import ConsoleSession

    lines: list[str] = []
    console = ConsoleSession(RunContext(dataset_root=tmp_path), prompt=lambda *a, **k: "a",
                             echo=lines.append)  # fmt: skip
    steps = ['eeg.preprocess(l_freq=0.1, h_freq=30.0, reference="average")', "eeg.erp(x=1)"]
    console._decide("analysis.erp_comparison", {"subjects": ["01", "02"], "steps": steps}, "")
    text = "\n".join(lines)
    assert "    subjects: 01, 02\n" in text
    assert f"    steps:\n        {steps[0]}\n        {steps[1]}" in text


# --- command line -------------------------------------------------------------------------


def test_cli(original: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    listed = runner.invoke(app, ["pipeline", "templates"])
    assert listed.exit_code == 0
    assert "erp-comparison" in listed.output and "condition_a (required)" in listed.output

    out = tmp_path / "plan.yaml"
    command = ["pipeline", "template", "erp-comparison", "--settings", json.dumps(ERP)]
    made = runner.invoke(app, [*command, "--dataset", str(original), "-o", str(out)])
    assert made.exit_code == 0, made.output
    assert load_spec(out).name == "erp-rare-vs-frequent"
    ok = runner.invoke(app, ["pipeline", "validate", str(out), "--dataset", str(original)])
    assert ok.exit_code == 0 and "OK" in ok.output

    bad = out.read_text(encoding="utf-8").replace("stimulus/rare", "stimulus/x")
    out.write_text(bad, encoding="utf-8")
    failed = runner.invoke(app, ["pipeline", "validate", str(out), "--dataset", str(original)])
    assert failed.exit_code == 1 and "none of its events (stimulus/x) occur" in failed.output
    refused = runner.invoke(app, ["pipeline", "run", str(out), "--dataset", str(original)])
    assert refused.exit_code == 1 and "nothing was run" in refused.output
