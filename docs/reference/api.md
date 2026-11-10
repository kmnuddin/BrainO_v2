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

::: braino.pipelines.checks.check_plan

::: braino.pipelines.templates.build

## Provenance and export

::: braino.provenance.store.load_run

::: braino.provenance.export.render_script

::: braino.provenance.replay.Replay

## Local model runtime

::: braino.runtime.tiers.recommend

::: braino.runtime.manager.setup

::: braino.runtime.manager.start_server

::: braino.runtime.server.LlamaServer

## LLM client

::: braino.llm.client.ChatClient

::: braino.llm.conversation.Conversation

::: braino.llm.steps.step_schema

::: braino.llm.select.select_tools

## Documentation search

::: braino.knowledge.search

::: braino.knowledge.index.DocsIndex

::: braino.knowledge.embed.ServerEmbedder

::: braino.knowledge.build.build_and_save

## Agent

::: braino.agent.loop.LLMAgent

::: braino.agent.console.ConsoleSession

::: braino.agent.session.Session

::: braino.agent.grounding.ungrounded_in_answer

## Agent evaluation

::: braino.evals.runner.run_suite

::: braino.evals.tasks.EvalTask

::: braino.training.episodes.Episode

::: braino.training.episodes.run_episode

::: braino.training.build.build

## Importing

::: braino.io.importer.import_recording

::: braino.io.deidentify.SubjectKey

## Validation data

::: braino.datasets.erpcore.fetch

::: braino.validation.erpcore.run

::: braino.validation.erpcore.compare
