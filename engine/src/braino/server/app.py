"""The routes of the local engine service.

Everything is under ``/api``. Every route except ``/api/health`` needs the service's token, in
an ``Authorization: Bearer <token>`` or ``X-BrainO-Token`` header; the WebSocket takes it as
its second subprotocol (``new WebSocket(url, ["braino", token])``), since browsers cannot set
headers on one. Requests must name the computer itself as their host, which stops a web page
from reaching the service through DNS rebinding.

Quick reads (tools, datasets, proposals, pipelines, runs) answer directly. Work that takes
time (running a tool, a pipeline, an import, a decision, the model setup) is a job: the
request returns the queued job, and its progress arrives over ``/api/events``.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from collections.abc import AsyncIterator, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.websockets import WebSocketDisconnect

from braino import __version__
from braino.server.jobs import JobConflictError, JobInfo, JobNotFoundError, JobRunner
from braino.server.store import BuiltinPipelineError, PipelineStore

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]")
WS_PROTOCOL = "braino"


# --- requests -----------------------------------------------------------------------------


class ToolJobRequest(BaseModel):
    tool: str = Field(description="Tool name, e.g. 'eeg.qc_report'")
    arguments: dict[str, Any] = Field(default_factory=dict)
    dataset: str | None = Field(default=None, description="BIDS dataset folder")


class PipelineJobRequest(BaseModel):
    dataset: str
    name: str | None = Field(default=None, description="A built-in or saved pipeline")
    spec: dict[str, Any] | None = Field(default=None, description="Or a pipeline spec")
    subjects: list[str] | None = None
    from_step: str | None = None
    auto_approve: bool = Field(
        default=False, description="Approve every proposal as made, in the user's name"
    )


class PipelineCheckRequest(BaseModel):
    spec: dict[str, Any]
    dataset: str | None = None
    subjects: list[str] | None = None


class SavePipelineRequest(BaseModel):
    spec: dict[str, Any]


class ImportJobRequest(BaseModel):
    dataset: str = Field(description="BIDS dataset to import into (created if missing)")
    key: str = Field(description="Subject key file; private, outside the dataset")
    jobs: list[dict[str, Any]] = Field(
        min_length=1, description="Recordings: source, subject_id, task, session, run, line_freq"
    )
    overwrite: bool = False


class DetectRequest(BaseModel):
    paths: list[str] = Field(min_length=1, description="Files or folders")


class DecisionRequest(BaseModel):
    dataset: str
    approve: bool
    edits: dict[str, Any] | None = Field(default=None, description="Fields to change first")
    note: str | None = None


class ModelSetupRequest(BaseModel):
    model: str | None = Field(default=None, description="Model ID; the recommended one if null")
    backend: str | None = None
    benchmark: bool = True


# --- the app ------------------------------------------------------------------------------


def create_app(
    *,
    token: str,
    runner: JobRunner | None = None,
    pipelines: PipelineStore | None = None,
    allow_origins: Sequence[str] = (),
    allowed_hosts: Sequence[str] = LOCAL_HOSTS,
) -> FastAPI:
    """The service. ``runner`` and ``pipelines`` default to this computer's; a runner made
    here is shut down (its jobs cancelled) when the app stops."""
    own_runner = runner is None
    jobs = runner or JobRunner()
    store = pipelines or PipelineStore()

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if own_runner:
            await asyncio.to_thread(jobs.shutdown)

    app = FastAPI(
        title="BrainO engine service",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.runner = jobs
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    if allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(allow_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def authorised(request: Request) -> None:
        supplied = request.headers.get("x-braino-token")
        header = request.headers.get("authorization", "")
        if supplied is None and header.lower().startswith("bearer "):
            supplied = header[7:].strip()
        if not supplied or not secrets.compare_digest(supplied, token):
            raise HTTPException(401, "a valid token is required")

    _errors(app)
    auth = [Depends(authorised)]

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    # --- the computer and the model -------------------------------------------------------

    @app.get("/api/system", dependencies=auth)
    def system() -> dict[str, Any]:
        from braino.runtime import installed_models, load_settings

        return {
            "version": __version__,
            "hardware": _hardware().model_dump(mode="json"),
            "jobs": {"max_parallel": jobs.max_parallel, "threads_per_job": jobs.threads},
            "model": load_settings().model_dump(mode="json"),
            "installed_models": [m.id for m in installed_models()],
        }

    # --- tools ----------------------------------------------------------------------------

    @app.get("/api/tools", dependencies=auth)
    def tools() -> list[dict[str, Any]]:
        from braino.tools import registry

        return [_tool_info(t) for t in registry]

    @app.get("/api/tools/{name}", dependencies=auth)
    def tool(name: str) -> dict[str, Any]:
        from braino.tools import registry

        return _tool_info(registry.get(name))

    # --- datasets and import --------------------------------------------------------------

    @app.get("/api/datasets/summary", dependencies=auth)
    def dataset_summary(
        path: str, max_recordings: int | None = Query(default=None, ge=0)
    ) -> dict[str, Any]:
        from braino.io.dataset import scan_dataset

        return scan_dataset(_folder(path), max_recordings=max_recordings).model_dump(mode="json")

    @app.post("/api/import/detect", dependencies=auth)
    def detect(request: DetectRequest) -> dict[str, Any]:
        return _detect(request.paths)

    @app.post("/api/jobs/import", dependencies=auth)
    def import_job(request: ImportJobRequest) -> JobInfo:
        from braino.io.batch import ImportJob

        for job in request.jobs:
            ImportJob.model_validate(job)
        dataset = str(Path(request.dataset).expanduser().resolve())
        n = len(request.jobs)
        return jobs.submit(
            "import",
            {**request.model_dump(), "dataset": dataset},
            title=f"Import {n} recording{'s' if n != 1 else ''}",
            dataset=dataset,
            key=f"import:{dataset}",
        )

    # --- running tools --------------------------------------------------------------------

    @app.post("/api/jobs/tool", dependencies=auth)
    def tool_job(request: ToolJobRequest) -> JobInfo:
        from braino.tools import registry

        found = registry.get(request.tool)
        found.input_model.model_validate(request.arguments)
        dataset = str(_folder(request.dataset)) if request.dataset else None
        subject = request.arguments.get("subject")
        title = found.name + (f" · sub-{subject}" if subject else "")
        payload = {"tool": found.name, "arguments": request.arguments, "dataset": dataset}
        return jobs.submit("tool", payload, title=title, dataset=dataset)

    # --- proposals and decisions ----------------------------------------------------------

    @app.get("/api/proposals", dependencies=auth)
    def proposals(dataset: str, status: str | None = None) -> list[dict[str, Any]]:
        from braino.tools import RunContext

        records = RunContext(dataset_root=_folder(dataset)).proposals.list()
        return [r.model_dump(mode="json") for r in records if status in (None, r.status)]

    @app.get("/api/proposals/{proposal_id}", dependencies=auth)
    def proposal(proposal_id: str, dataset: str) -> dict[str, Any]:
        from braino.tools import RunContext

        ctx = RunContext(dataset_root=_folder(dataset))
        return ctx.proposals.get(proposal_id).model_dump(mode="json")

    @app.post("/api/proposals/{proposal_id}/decide", dependencies=auth)
    def decide(proposal_id: str, request: DecisionRequest) -> JobInfo:
        from braino.tools import RunContext, current_user

        folder = _folder(request.dataset)
        record = RunContext(dataset_root=folder).proposals.get(proposal_id)
        if record.status != "pending":
            raise HTTPException(409, f"proposal {proposal_id} is already {record.status}")
        verb = "Approve" if request.approve else "Reject"
        payload = {
            **request.model_dump(),
            "dataset": str(folder),
            "proposal_id": proposal_id,
            "decided_by": current_user(),
        }
        return jobs.submit(
            "decision",
            payload,
            title=f"{verb} {record.tool} proposal",
            dataset=str(folder),
            key=f"proposal:{folder}:{proposal_id}",
        )

    # --- pipelines ------------------------------------------------------------------------

    @app.get("/api/pipelines", dependencies=auth)
    def pipeline_list() -> list[dict[str, Any]]:
        return store.list()

    @app.get("/api/pipelines/{name}", dependencies=auth)
    def pipeline_get(name: str) -> dict[str, Any]:
        source, spec = store.get(name)
        return {"source": source, "spec": spec.model_dump(mode="json")}

    @app.put("/api/pipelines/{name}", dependencies=auth)
    def pipeline_save(name: str, request: SavePipelineRequest) -> dict[str, Any]:
        spec = store.save(name, request.spec)
        return {"source": "user", "spec": spec.model_dump(mode="json")}

    @app.delete("/api/pipelines/{name}", dependencies=auth)
    def pipeline_delete(name: str) -> dict[str, Any]:
        store.delete(name)
        return {"deleted": name}

    @app.post("/api/pipelines/check", dependencies=auth)
    def pipeline_check(request: PipelineCheckRequest) -> dict[str, Any]:
        dataset = _folder(request.dataset) if request.dataset else None
        return _check(request.spec, dataset, request.subjects)

    @app.get("/api/pipelines/{name}/state", dependencies=auth)
    def pipeline_state(name: str, dataset: str) -> dict[str, Any]:
        _, spec = store.get(name)
        return _state(spec, _folder(dataset))

    @app.post("/api/jobs/pipeline", dependencies=auth)
    def pipeline_job(request: PipelineJobRequest) -> JobInfo:
        if (request.name is None) == (request.spec is None):
            raise HTTPException(422, "give either a pipeline name or a spec")
        spec_data = (
            store.get(request.name)[1].model_dump(mode="json")
            if request.name is not None
            else request.spec
        )
        assert spec_data is not None
        folder = _folder(request.dataset)
        check = _check(spec_data, folder, request.subjects)
        if not check["valid"]:
            message = "the pipeline check found problems; nothing was run"
            raise HTTPException(422, {"message": message, **check})
        name = spec_data["name"]
        payload = {**request.model_dump(), "spec": spec_data, "dataset": str(folder)}
        return jobs.submit(
            "pipeline",
            payload,
            title=f"Pipeline {name}",
            dataset=str(folder),
            key=f"pipeline:{folder}:{name}",
        )

    # --- the model ------------------------------------------------------------------------

    @app.post("/api/jobs/model-setup", dependencies=auth)
    def model_setup(request: ModelSetupRequest) -> JobInfo:
        return jobs.submit(
            "model_setup", request.model_dump(), title="Set up the AI model", key="model_setup"
        )

    # --- jobs -----------------------------------------------------------------------------

    @app.get("/api/jobs", dependencies=auth)
    def job_list() -> list[dict[str, Any]]:
        return [j.summary() for j in jobs.list()]

    @app.get("/api/jobs/{job_id}", dependencies=auth)
    def job_get(job_id: str) -> JobInfo:
        return jobs.get(job_id)

    @app.post("/api/jobs/{job_id}/cancel", dependencies=auth)
    def job_cancel(job_id: str) -> JobInfo:
        return jobs.cancel(job_id)

    # --- provenance -----------------------------------------------------------------------

    @app.get("/api/runs", dependencies=auth)
    def runs(dataset: str) -> list[str]:
        from braino.provenance.store import list_runs
        from braino.tools import RunContext

        return list_runs(RunContext(dataset_root=_folder(dataset)).derivatives_root)

    @app.get("/api/runs/{run_id}", dependencies=auth)
    def run(run_id: str, dataset: str) -> dict[str, Any]:
        from braino.provenance.store import load_run
        from braino.tools import RunContext

        root = RunContext(dataset_root=_folder(dataset)).derivatives_root
        return load_run(root, run_id).model_dump(mode="json")

    # --- events ---------------------------------------------------------------------------

    @app.websocket("/api/events")
    async def events(websocket: WebSocket) -> None:
        offered = [
            p.strip()
            for p in websocket.headers.get("sec-websocket-protocol", "").split(",")
            if p.strip()
        ]
        given = offered[1] if len(offered) == 2 and offered[0] == WS_PROTOCOL else ""
        if not given or not secrets.compare_digest(given, token):
            await websocket.close(code=1008)
            return
        await websocket.accept(subprotocol=WS_PROTOCOL)
        loop = asyncio.get_running_loop()
        inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def push(message: dict[str, Any]) -> None:  # called from the runner's threads
            loop.call_soon_threadsafe(inbox.put_nowait, message)

        unsubscribe = jobs.subscribe(push)
        try:
            await websocket.send_json({"type": "jobs", "jobs": [j.summary() for j in jobs.list()]})
            closed = asyncio.ensure_future(_until_closed(websocket))
            while not closed.done():
                getting = asyncio.ensure_future(inbox.get())
                await asyncio.wait({getting, closed}, return_when=asyncio.FIRST_COMPLETED)
                if getting.done():
                    await websocket.send_json(getting.result())
                else:
                    getting.cancel()
        except WebSocketDisconnect:
            pass
        finally:
            unsubscribe()

    return app


async def _until_closed(websocket: WebSocket) -> None:
    """Read (and ignore) what the client sends, until it disconnects."""
    with contextlib.suppress(WebSocketDisconnect):
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return


# --- helpers ------------------------------------------------------------------------------


def _errors(app: FastAPI) -> None:
    """Map the engine's exceptions to HTTP answers."""
    from braino.pipelines import SpecError
    from braino.provenance.proposals import ProposalNotFoundError
    from braino.provenance.store import RunNotFoundError
    from braino.tools import ToolNotFoundError

    def answer(status: int) -> Any:
        async def handler(_: Request, exc: Exception) -> JSONResponse:
            detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            return JSONResponse(status_code=status, content={"detail": str(detail)})

        return handler

    for exc_type, status in (
        (ToolNotFoundError, 404),
        (ProposalNotFoundError, 404),
        (RunNotFoundError, 404),
        (JobNotFoundError, 404),
        (FileNotFoundError, 404),
        (NotADirectoryError, 400),
        (JobConflictError, 409),
        (BuiltinPipelineError, 409),
        (SpecError, 422),
    ):
        app.add_exception_handler(exc_type, answer(status))

    async def invalid(_: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, ValidationError)
        errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    app.add_exception_handler(ValidationError, invalid)


def _folder(path: str) -> Path:
    folder = Path(path).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"no such folder: {folder}")
    return folder


@cache
def _hardware() -> Any:
    from braino.runtime import detect

    return detect()


def _tool_info(tool: Any) -> dict[str, Any]:
    fields = tool.input_model.model_fields
    return {
        "name": tool.name,
        "description": tool.description,
        "risk": tool.risk.value,
        "needs_dataset": tool.needs_context,
        "per_recording": "subject" in fields,
        "input_schema": tool.input_model.model_json_schema(),
        "output_schema": tool.output_model.model_json_schema(),
    }


def _check(
    spec_data: dict[str, Any], dataset: Path | None, subjects: list[str] | None
) -> dict[str, Any]:
    """The spec's own check, then the whole-plan check (against the data, with a dataset)."""
    from braino.pipelines import PipelineSpec, SpecError, check_spec
    from braino.pipelines.checks import check_plan

    try:
        spec = PipelineSpec.model_validate(spec_data)
        check_spec(spec)
    except (ValidationError, SpecError) as exc:
        return {"valid": False, "problems": str(exc).splitlines(), "findings": []}
    plan = check_plan(spec, dataset=dataset, subjects=subjects)
    return {
        "valid": plan.ok,
        "problems": [],
        "findings": [f.model_dump(mode="json") for f in plan.findings],
    }


def _state(spec: Any, dataset: Path) -> dict[str, Any]:
    """For each subject and step, whether it has a current result or waits for a decision."""
    from braino.pipelines import load_state
    from braino.provenance.proposals import ProposalNotFoundError
    from braino.tools import RunContext

    ctx = RunContext(dataset_root=dataset)
    state = load_state(ctx, spec)
    subjects: dict[str, dict[str, Any]] = {}
    for record in state.records.values():
        status = "done"
        if record.proposal_id and record.apply is None:
            try:
                status = ctx.proposals.get(record.proposal_id).status
            except ProposalNotFoundError:
                status = "pending"
            status = "waiting" if status == "pending" else status
        subjects.setdefault(record.subject, {})[record.step] = {
            "tool": record.tool,
            "status": status,
            "proposal_id": record.proposal_id,
            "finished_at": record.finished_at.isoformat(),
        }
    return {
        "name": spec.name,
        "steps": [s.id for s in spec.steps],
        "group_steps": [s.id for s in spec.group],
        "subjects": subjects,
    }


MAX_DETECT_FILES = 5000


def _detect(paths: list[str]) -> dict[str, Any]:
    """The recordings among files and folders: each recording once, with its format and
    files, and the other files with the reason they were left out."""
    from braino.io.formats import detect_format

    files: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"no such file or folder: {path}")
        if path.is_dir() and path.suffix.lower() != ".mff":
            files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            files.append(path)
        if len(files) > MAX_DETECT_FILES:
            raise HTTPException(422, f"more than {MAX_DETECT_FILES} files; choose fewer")

    recordings: dict[Path, dict[str, Any]] = {}
    claimed: set[Path] = set()
    failed: dict[Path, str] = {}
    for file in files:
        try:
            found = detect_format(file)
        except Exception as exc:  # unsupported, or a needed file is missing
            failed[file] = str(exc)
            continue
        claimed.update(Path(f).resolve() for f in found.files)
        recordings.setdefault(
            found.path,
            {
                "path": str(found.path),
                "format": found.format.id,
                "format_name": found.format.name,
                "files": [str(f) for f in found.files],
            },
        )
    skipped = [
        {"path": str(f), "reason": reason} for f, reason in failed.items() if f not in claimed
    ]
    return {"recordings": list(recordings.values()), "skipped": skipped}
