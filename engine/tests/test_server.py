"""The local engine service: routes, token, jobs and their events."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from braino import __version__
from braino.server import JobConflictError, JobRunner, serve
from braino.server.app import create_app
from braino.server.jobs import parallel_limit
from braino.server.store import PipelineStore

TOKEN = "a" * 48
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
JOB_TIMEOUT = 300.0
EVENTS_URL = "ws://127.0.0.1/api/events"

SPEC = {
    "name": "my-oddball",
    "description": "made in the builder",
    "task": "oddball",
    "steps": [
        {"id": "preprocess", "tool": "eeg.preprocess", "args": {"h_freq": 30.0}},
        {"id": "bad_channels", "tool": "eeg.detect_bad_channels", "args": {"method": "qc"}},
    ],
}


# --- workers for the job tests (a job's process imports them as test_server:<name>) ---------


def slow_worker(kind: str, payload: dict[str, Any], queue: Any, threads: int) -> None:
    queue.put(("started", {"run_id": "run-1"}))
    queue.put(("progress", {"fraction": 0.5, "message": "half way", "event": {"n": 1}}))
    time.sleep(payload.get("seconds", 60))
    queue.put(("done", {"threads": threads}))


def crashing_worker(kind: str, payload: dict[str, Any], queue: Any, threads: int) -> None:
    os._exit(3)


def failing_worker(kind: str, payload: dict[str, Any], queue: Any, threads: int) -> None:
    queue.put(("failed", "ValueError: no good"))


# --- fixtures -----------------------------------------------------------------------------


@pytest.fixture
def runner() -> Iterator[JobRunner]:
    jobs = JobRunner(max_parallel=2, threads=1)
    yield jobs
    jobs.shutdown()


@pytest.fixture
def client(runner: JobRunner, tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(token=TOKEN, runner=runner, pipelines=PipelineStore(tmp_path / "pipelines"))
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


def wait_done(client: TestClient, job: dict[str, Any]) -> dict[str, Any]:
    deadline = time.monotonic() + JOB_TIMEOUT
    while time.monotonic() < deadline:
        info = client.get(f"/api/jobs/{job['id']}", headers=HEADERS).json()
        if info["status"] in ("done", "failed", "cancelled"):
            return info  # type: ignore[no-any-return]
        time.sleep(0.2)
    raise AssertionError(f"job {job['id']} did not finish")


# --- the token and the host ---------------------------------------------------------------


def test_health_needs_no_token_but_everything_else_does(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok", "version": __version__}
    assert client.get("/api/tools").status_code == 401
    assert client.get("/api/tools", headers={"X-BrainO-Token": "wrong"}).status_code == 401
    assert client.get("/api/tools", headers={"X-BrainO-Token": TOKEN}).status_code == 200
    assert client.get("/api/tools", headers=HEADERS).status_code == 200


def test_only_requests_naming_this_computer_are_answered(client: TestClient) -> None:
    response = client.get("http://attacker.example/api/health")
    assert response.status_code == 400


def test_the_service_listens_on_this_computer_only() -> None:
    with pytest.raises(ValueError, match="only listens on this computer"):
        serve(host="0.0.0.0")


# --- reads --------------------------------------------------------------------------------


def test_tools_with_their_schemas(client: TestClient) -> None:
    tools = {t["name"]: t for t in client.get("/api/tools", headers=HEADERS).json()}
    preprocess = tools["eeg.preprocess"]
    assert preprocess["risk"] == "compute"
    assert preprocess["per_recording"] and preprocess["needs_dataset"]
    assert "h_freq" in preprocess["input_schema"]["properties"]
    assert tools["eeg.detect_bad_channels"]["risk"] == "decision"
    assert tools["stats.cluster_permutation"]["per_recording"] is False
    assert client.get("/api/tools/eeg.erp", headers=HEADERS).json()["name"] == "eeg.erp"
    missing = client.get("/api/tools/eeg.nope", headers=HEADERS)
    assert missing.status_code == 404 and "eeg.nope" in missing.json()["detail"]


def test_system(client: TestClient) -> None:
    info = client.get("/api/system", headers=HEADERS).json()
    assert info["version"] == __version__
    assert info["jobs"] == {"max_parallel": 2, "threads_per_job": 1}
    assert "cpu_count" in info["hardware"]


def test_unknown_folders_are_refused(client: TestClient, tmp_path: Path) -> None:
    missing = tmp_path / "nowhere"
    response = client.get(f"/api/datasets/summary?path={missing}", headers=HEADERS)
    assert response.status_code == 400
    assert "no such folder" in response.json()["detail"]


def test_tool_arguments_are_checked_before_queueing(client: TestClient) -> None:
    response = client.post(
        "/api/jobs/tool",
        headers=HEADERS,
        json={"tool": "eeg.preprocess", "arguments": {"subject": "01", "h_freq": -1}},
    )
    assert response.status_code == 422
    assert client.get("/api/jobs", headers=HEADERS).json() == []


# --- pipelines ----------------------------------------------------------------------------


def test_saving_and_checking_pipelines(client: TestClient, tmp_path: Path) -> None:
    listed = client.get("/api/pipelines", headers=HEADERS).json()
    assert {"erpcore-p3", "erpcore-n170"} <= {p["name"] for p in listed}
    assert {p["source"] for p in listed} == {"builtin"}

    saved = client.put("/api/pipelines/my-oddball", headers=HEADERS, json={"spec": SPEC})
    assert saved.status_code == 200, saved.text
    assert (tmp_path / "pipelines" / "my-oddball.yaml").is_file()
    mine = client.get("/api/pipelines/my-oddball", headers=HEADERS).json()
    assert mine["source"] == "user"
    assert [s["id"] for s in mine["spec"]["steps"]] == ["preprocess", "bad_channels"]
    listed = client.get("/api/pipelines", headers=HEADERS).json()
    assert ("my-oddball", "user") in {(p["name"], p["source"]) for p in listed}

    builtin = {**SPEC, "name": "erpcore-p3"}
    response = client.put("/api/pipelines/erpcore-p3", headers=HEADERS, json={"spec": builtin})
    assert response.status_code == 409
    response = client.put("/api/pipelines/other", headers=HEADERS, json={"spec": SPEC})
    assert response.status_code == 422 and "must be 'other'" in response.json()["detail"]
    bad = {**SPEC, "steps": [{"id": "x", "tool": "eeg.nope"}]}
    response = client.put("/api/pipelines/my-oddball", headers=HEADERS, json={"spec": bad})
    assert response.status_code == 422 and "no tool 'eeg.nope'" in response.json()["detail"]
    response = client.put("/api/pipelines/..%2Fescape", headers=HEADERS, json={"spec": SPEC})
    assert response.status_code in (404, 422)

    assert client.delete("/api/pipelines/erpcore-p3", headers=HEADERS).status_code == 409
    assert client.delete("/api/pipelines/my-oddball", headers=HEADERS).status_code == 200
    assert client.get("/api/pipelines/my-oddball", headers=HEADERS).status_code == 404


def test_pipeline_check(client: TestClient) -> None:
    good = client.post("/api/pipelines/check", headers=HEADERS, json={"spec": SPEC}).json()
    assert good["valid"] and good["problems"] == []
    bad_spec = {**SPEC, "steps": [{"id": "pre", "tool": "eeg.preprocess", "args": {"h_freq": -1}}]}
    bad = client.post("/api/pipelines/check", headers=HEADERS, json={"spec": bad_spec}).json()
    assert not bad["valid"]
    assert any("h_freq" in p for p in bad["problems"])


# --- jobs ---------------------------------------------------------------------------------


def test_parallel_limit() -> None:
    assert parallel_limit(12, 16.0) == 3  # memory: (16 - 6) // 3
    assert parallel_limit(4, 64.0) == 2  # processors: 4 // 2
    assert parallel_limit(1, 8.0) == 1
    assert parallel_limit(None, None) == 1
    assert parallel_limit(64, 512.0) == 8


def test_jobs_queue_report_and_cancel() -> None:
    runner = JobRunner(max_parallel=1, threads=3, worker="test_server:slow_worker")
    messages: list[dict[str, Any]] = []
    runner.subscribe(messages.append)
    try:
        first = runner.submit("tool", {}, title="first", key="same")
        with pytest.raises(JobConflictError, match="already working on this"):
            runner.submit("tool", {}, title="clash", key="same")
        second = runner.submit("tool", {"seconds": 0}, title="second")
        assert runner.get(second.id).status == "queued"  # one at a time

        deadline = time.monotonic() + 60
        while not runner.get(first.id).events and time.monotonic() < deadline:
            time.sleep(0.1)
        running = runner.get(first.id)
        assert running.status == "running" and running.run_id == "run-1"
        assert running.progress == 0.5 and running.message == "half way"
        assert running.events == [{"n": 1}]

        runner.cancel(first.id)
        assert runner.wait(first.id, timeout=30).status == "cancelled"
        done = runner.wait(second.id, timeout=60)
        assert done.status == "done" and done.result == {"threads": 3}
        # The finished job's key is free again.
        runner.cancel(runner.submit("tool", {}, title="again", key="same").id)
    finally:
        runner.shutdown()
    kinds = {m["type"] for m in messages}
    assert kinds == {"job", "job_event"}
    statuses = [m["job"]["status"] for m in messages if m["type"] == "job"]
    assert {"queued", "running", "cancelled", "done"} <= set(statuses)
    assert all("events" not in m["job"] for m in messages if m["type"] == "job")


def test_failed_and_crashed_jobs() -> None:
    failing = JobRunner(max_parallel=1, threads=1, worker="test_server:failing_worker")
    crashing = JobRunner(max_parallel=1, threads=1, worker="test_server:crashing_worker")
    try:
        failed = failing.wait(failing.submit("tool", {}, title="f").id, timeout=60)
        assert failed.status == "failed" and failed.error == "ValueError: no good"
        crashed = crashing.wait(crashing.submit("tool", {}, title="c").id, timeout=60)
        assert crashed.status == "failed"
        assert crashed.error == "the job's process stopped (exit code 3)"
    finally:
        failing.shutdown()
        crashing.shutdown()


def test_cancelling_a_queued_job_and_unknown_jobs(client: TestClient, runner: JobRunner) -> None:
    assert client.get("/api/jobs/job-nope", headers=HEADERS).status_code == 404
    assert client.post("/api/jobs/job-nope/cancel", headers=HEADERS).status_code == 404


def test_a_tool_job_from_start_to_end(client: TestClient) -> None:
    job = client.post("/api/jobs/tool", headers=HEADERS, json={"tool": "system.info"}).json()
    assert job["status"] in ("queued", "running") and job["title"] == "system.info"
    done = wait_done(client, job)
    assert done["status"] == "done", done["error"]
    assert done["result"]["tool"] == "system.info"
    assert done["result"]["result"]["braino_version"] == __version__
    assert done["run_id"] == done["result"]["run_id"]


def test_events_over_the_websocket(client: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(EVENTS_URL, subprotocols=["braino", "wrong"]) as ws,
    ):
        ws.receive_json()
    with client.websocket_connect(EVENTS_URL, subprotocols=["braino", TOKEN]) as ws:
        assert ws.receive_json() == {"type": "jobs", "jobs": []}
        job = client.post("/api/jobs/tool", headers=HEADERS, json={"tool": "system.info"}).json()
        seen = []
        while True:
            message = ws.receive_json()
            assert message["type"] == "job" and message["job"]["id"] == job["id"]
            seen.append(message["job"]["status"])
            if message["job"]["status"] in ("done", "failed"):
                break
        assert seen[-1] == "done"
        assert "running" in seen


# --- with real data (needs the science stack) ---------------------------------------------


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    pytest.importorskip("mne")
    pytest.importorskip("mne_bids")
    pytest.importorskip("pybv")
    from braino.datasets.synthetic import write_dataset

    return write_dataset(tmp_path / "bids", ["01", "02"], task="oddball", seconds=30.0)


def test_a_pipeline_waits_for_a_decision_made_in_the_app(client: TestClient, dataset: Path) -> None:
    summary = client.get(f"/api/datasets/summary?path={dataset}", headers=HEADERS).json()
    assert summary["subjects"] == ["01", "02"] and summary["tasks"] == ["oddball"]

    request = {"dataset": str(dataset), "spec": SPEC, "subjects": ["01"]}
    job = client.post("/api/jobs/pipeline", headers=HEADERS, json=request).json()
    clash = client.post("/api/jobs/pipeline", headers=HEADERS, json=request)
    assert clash.status_code == 409
    done = wait_done(client, job)
    assert done["status"] == "done", done["error"]
    assert done["result"]["counts"]["ran"] == 1 and done["result"]["counts"]["pending"] == 1
    assert [e["step"] for e in done["events"]] == ["preprocess", "bad_channels"]
    assert done["progress"] == 1.0

    pending = client.get(f"/api/proposals?dataset={dataset}&status=pending", headers=HEADERS).json()
    assert len(pending) == 1 and pending[0]["tool"] == "eeg.detect_bad_channels"
    proposal_id = pending[0]["proposal_id"]

    client.put("/api/pipelines/my-oddball", headers=HEADERS, json={"spec": SPEC})
    state = client.get(f"/api/pipelines/my-oddball/state?dataset={dataset}", headers=HEADERS).json()
    assert state["subjects"]["01"]["bad_channels"]["status"] == "waiting"
    assert state["subjects"]["01"]["preprocess"]["status"] == "done"

    decision = {"dataset": str(dataset), "approve": True, "note": "looked fine"}
    job = client.post(f"/api/proposals/{proposal_id}/decide", headers=HEADERS, json=decision).json()
    decided = wait_done(client, job)
    assert decided["status"] == "done", decided["error"]
    assert decided["result"]["status"] == "approved"
    assert decided["result"]["decision"]["note"] == "looked fine"
    again = client.post(f"/api/proposals/{proposal_id}/decide", headers=HEADERS, json=decision)
    assert again.status_code == 409

    runs = client.get(f"/api/runs?dataset={dataset}", headers=HEADERS).json()
    assert len(runs) == 2
    log = client.get(f"/api/runs/{runs[0]}?dataset={dataset}", headers=HEADERS).json()
    assert [c["tool"] for c in log["calls"]] == ["eeg.preprocess", "eeg.detect_bad_channels"]


def test_a_pipeline_that_fails_its_check_does_not_run(client: TestClient, dataset: Path) -> None:
    spec = {**SPEC, "task": "nback"}
    request = {"dataset": str(dataset), "spec": spec}
    response = client.post("/api/jobs/pipeline", headers=HEADERS, json=request)
    assert response.status_code == 422
    assert response.json()["detail"]["valid"] is False
    assert client.get("/api/jobs", headers=HEADERS).json() == []


def test_detecting_recordings_in_a_folder(client: TestClient, tmp_path: Path) -> None:
    pytest.importorskip("mne")
    from braino.datasets.synthetic import make_raw

    folder = tmp_path / "new_study"
    folder.mkdir()
    make_raw(seed=1, seconds=5.0).save(folder / "s01_raw.fif")
    (folder / "notes.txt").write_text("not a recording", encoding="utf-8")
    (folder / "lonely.vhdr").write_text("Brain Vision Data Exchange Header File", encoding="utf-8")
    found = client.post("/api/import/detect", headers=HEADERS, json={"paths": [str(folder)]}).json()
    assert [(Path(r["path"]).name, r["format"]) for r in found["recordings"]] == [
        ("s01_raw.fif", "fif")
    ]
    assert {Path(s["path"]).name for s in found["skipped"]} == {"notes.txt", "lonely.vhdr"}
    missing = client.post(
        "/api/import/detect", headers=HEADERS, json={"paths": [str(tmp_path / "x")]}
    )
    assert missing.status_code == 404


# --- the command --------------------------------------------------------------------------


def test_braino_serve_starts_answers_and_stops_with_its_input() -> None:
    import httpx

    env = {**os.environ, "BRAINO_SERVER_TOKEN": TOKEN}
    process = subprocess.Popen(
        [sys.executable, "-c", "from braino.cli import app; app()", "serve", "--exit-with-stdin"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
        text=True,
    )
    try:
        assert process.stdout is not None and process.stdin is not None
        ready = json.loads(process.stdout.readline())["braino_server"]
        assert ready["url"].startswith("http://127.0.0.1:") and "token" not in ready
        assert httpx.get(f"{ready['url']}/api/health", timeout=30).json()["status"] == "ok"
        tools = httpx.get(f"{ready['url']}/api/tools", headers=HEADERS, timeout=30)
        assert tools.status_code == 200
        process.stdin.close()
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.kill()
