"""Import recordings into a dataset with a provenance record, one at a time or from a mapping
file.

Import is not an agent tool: it reads the original, identifiable files, and its inputs (file
paths, the person's real ID) must not reach the agent or the dataset. So the provenance record of
an import holds only what is safe: the subject label, task, session and run, the hashes of the
source files (recorded under neutral names), the files written, and the import report. Error
messages are scrubbed of the source paths and the person's ID before they are recorded.

Needs the ``[eeg]`` extra.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from braino.context import RunContext
from braino.io.deidentify import SubjectKey
from braino.io.formats import detect_format
from braino.io.importer import ImportReport, import_recording

IMPORT_RECORD_NAME = "dataset.import"

MAPPING_COLUMNS = ("source", "subject_id", "task", "session", "run", "line_freq")
"""Columns of a mapping file; ``source``, ``subject_id`` and ``task`` are required."""


class ImportJob(BaseModel):
    """One recording to import."""

    source: Path
    subject_id: str = Field(description="The person's original identifier; stays in the key")
    task: str
    session: str | None = None
    run: int | None = None
    line_freq: float | None = None

    @field_validator("session", "line_freq", "run", mode="before")
    @classmethod
    def _blank_is_none(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value


class ImportArguments(BaseModel):
    """What the provenance record of an import keeps about its inputs."""

    subject: str | None = Field(description="Subject label in the dataset")
    task: str
    session: str | None
    run: int | None


class ImportFailedError(RuntimeError):
    """Raised when an import fails; the message names no source path or original ID."""


class MappingError(ValueError):
    pass


def read_mapping(path: str | Path, *, dataset_root: str | Path) -> list[ImportJob]:
    """Read a mapping file: a CSV with the columns in :data:`MAPPING_COLUMNS`.

    Relative ``source`` paths are taken from the mapping file's folder. The file links real IDs
    to recordings, so like the subject key it must be outside the dataset.
    """
    mapping = Path(path).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    if mapping.is_relative_to(root):
        raise MappingError(f"the mapping file {mapping} must be kept outside the dataset {root}")
    with mapping.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or ())
        missing = {"source", "subject_id", "task"} - columns
        if missing:
            raise MappingError(f"{mapping.name} lacks the columns: {', '.join(sorted(missing))}")
        unknown = columns - set(MAPPING_COLUMNS)
        if unknown:
            raise MappingError(f"{mapping.name} has unknown columns: {', '.join(sorted(unknown))}")
        jobs = []
        for line, row in enumerate(reader, start=2):
            try:
                job = ImportJob.model_validate({k: v for k, v in row.items() if k})
            except ValueError as exc:
                raise MappingError(f"{mapping.name}, line {line}: {exc}") from None
            if not job.source.is_absolute():
                job = job.model_copy(update={"source": mapping.parent / job.source})
            jobs.append(job)
    return jobs


def import_job(
    ctx: RunContext, key: SubjectKey, job: ImportJob, *, overwrite: bool = False
) -> ImportReport:
    """Import ``job`` into the context's dataset and record it in the run's provenance.

    Raises :class:`ImportFailedError` (with a scrubbed message) if the import fails.
    """
    arguments = ImportArguments(subject=None, task=job.task, session=job.session, run=job.run)
    secrets = _secrets(job)
    with ctx.provenance.tool_call(IMPORT_RECORD_NAME, "compute", arguments) as call:
        try:
            for path in detect_format(job.source).files:
                ctx.record_input(path)
            report = import_recording(
                job.source,
                ctx.dataset,
                key=key,
                original_id=job.subject_id,
                task=job.task,
                session=job.session,
                run=job.run,
                line_freq=job.line_freq,
                overwrite=overwrite,
            )
        except Exception as exc:
            message = _scrub(f"{type(exc).__name__}: {exc}", secrets)
            raise ImportFailedError(message) from None
        call.arguments = arguments.model_copy(update={"subject": report.deidentification.subject})
        for name in report.conversion.files:
            ctx.record_output(ctx.dataset / name)
        call.result = report
    return report


def _secrets(job: ImportJob) -> list[str]:
    """Strings an error message about ``job`` must not carry."""
    source = job.source.expanduser()
    values = {job.subject_id, str(source), source.name, str(source.parent), source.stem}
    try:
        resolved = source.resolve()
        values |= {str(resolved), str(resolved.parent), resolved.as_posix()}
    except OSError:
        pass
    return sorted((v for v in values if v and len(v) >= 3), key=len, reverse=True)


def _scrub(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "<source>")
    return text
