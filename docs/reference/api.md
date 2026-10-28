# Python API

The main entry points. Every tool is also callable through the registry (see
[Tools](tools.md)): `registry.get("eeg.preprocess").run({...}, ctx)`.

## Running tools

::: braino.context.RunContext

::: braino.tools.registry.ToolRegistry

::: braino.tools.decisions.decide

## Pipelines

::: braino.pipelines.spec.PipelineSpec

::: braino.pipelines.spec.load_spec

::: braino.pipelines.runner.run_pipeline

::: braino.pipelines.runner.PipelineReport

## Provenance and export

::: braino.provenance.store.load_run

::: braino.provenance.export.render_script

::: braino.provenance.replay.Replay

## Local model runtime

::: braino.runtime.tiers.recommend

::: braino.runtime.manager.setup

::: braino.runtime.manager.start_server

::: braino.runtime.server.LlamaServer

## Agent evaluation

::: braino.agent.session.Session

::: braino.evals.runner.run_suite

::: braino.evals.tasks.EvalTask

## Importing

::: braino.io.importer.import_recording

::: braino.io.deidentify.SubjectKey

## Validation data

::: braino.datasets.erpcore.fetch

::: braino.validation.erpcore.run

::: braino.validation.erpcore.compare
