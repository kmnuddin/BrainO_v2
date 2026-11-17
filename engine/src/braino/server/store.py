"""Pipelines the user builds in the app, kept next to the built-in ones.

Saved pipelines are YAML files in :func:`braino.paths.user_pipelines_dir`, in the same format
as ``braino pipeline run`` reads, so a pipeline built in the app also runs from the command
line. A saved pipeline cannot take a built-in one's name.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from braino.paths import user_pipelines_dir
from braino.pipelines import PipelineSpec, SpecError, builtin_specs, check_spec, load_spec
from braino.pipelines.spec import NAME_PATTERN

Source = Literal["builtin", "user", "ran"]
"""Built in, saved by the user, or only run on a dataset (from a template or the builder)."""


class BuiltinPipelineError(ValueError):
    """Raised when a change would replace or delete a built-in pipeline."""


class PipelineStore:
    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory

    @property
    def directory(self) -> Path:
        return self._directory or user_pipelines_dir()

    def list(self) -> list[dict[str, Any]]:
        """Built-in pipelines first, then the user's, each with its name, description, task
        and steps."""
        items = [self.item("builtin", load_spec(name)) for name in sorted(builtin_specs())]
        for path in sorted(self.directory.glob("*.yaml")) if self.directory.is_dir() else []:
            try:
                spec = load_spec(path)
            except (SpecError, OSError, yaml.YAMLError) as exc:
                items.append({"name": path.stem, "source": "user", "error": str(exc)})
                continue
            items.append(self.item("user", spec))
        return items

    def get(self, name: str) -> tuple[Source, PipelineSpec]:
        if name in builtin_specs():
            return "builtin", load_spec(name)
        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(f"no pipeline {name!r}")
        return "user", load_spec(path)

    def save(self, name: str, data: dict[str, Any]) -> PipelineSpec:
        """Check and save a pipeline under ``name`` (its spec's own name), replacing a saved
        one of that name."""
        if name in builtin_specs():
            raise BuiltinPipelineError(
                f"{name!r} is a built-in pipeline; save your version under another name"
            )
        if data.get("name") != name:
            raise SpecError(f"the spec's name ({data.get('name')!r}) must be {name!r}")
        try:
            spec = PipelineSpec.model_validate(data)
        except ValidationError as exc:
            raise SpecError(str(exc)) from None
        check_spec(spec)
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(
            spec.model_dump(mode="json", exclude_defaults=True),
            sort_keys=False,
            allow_unicode=True,
        )
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        return spec

    def delete(self, name: str) -> None:
        if name in builtin_specs():
            raise BuiltinPipelineError(f"{name!r} is a built-in pipeline and cannot be deleted")
        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(f"no pipeline {name!r}")
        path.unlink()

    def _path(self, name: str) -> Path:
        # The spec's name pattern allows only lower-case letters, digits, "-" and "_", so a
        # name cannot reach outside the folder.
        if not re.fullmatch(NAME_PATTERN, name):
            raise SpecError(f"invalid pipeline name {name!r}")
        return self.directory / f"{name}.yaml"

    @staticmethod
    def item(source: Source, spec: PipelineSpec) -> dict[str, Any]:
        return {
            "name": spec.name,
            "source": source,
            "description": spec.description,
            "task": spec.task,
            "steps": [{"id": s.id, "tool": s.tool} for s in spec.steps],
            "group": [{"id": s.id, "tool": s.tool} for s in spec.group],
        }
