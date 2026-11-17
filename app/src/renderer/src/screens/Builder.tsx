/** The pipeline builder: add steps from a menu (clean, measure, test), reorder and remove
 * them, edit each step's form, check that every step gets the data it needs, save and run. */

import { useEffect, useMemo, useRef, useState } from "react";
import { RiskPill, RunPanel, StepCard, useSuggestions } from "../components/PipelineParts";
import * as Icon from "../components/icons";
import { Loading } from "../components/ui";
import { api } from "../engine/api";
import type { CheckResult, PipelineSpec, PipelineStep, ToolInfo } from "../engine/types";
import { type StepGroup, emptySpec, moved, newStepId, problemsByStep, stepGroup } from "../lib/pipeline";
import { useApp, useTools } from "../state/app";
import { useDataset } from "../state/dataset";
import { SaveAs } from "./Pipelines";

/** The draft survives leaving the builder and coming back (until the app closes). */
let kept: PipelineSpec | null = null;

const GROUPS: { key: StepGroup; label: string; hint: string }[] = [
  { key: "clean", label: "Clean", hint: "filter, bad channels, ICA" },
  { key: "measure", label: "Measure", hint: "epochs, ERPs, spectra, time–frequency" },
  { key: "test", label: "Test", hint: "group statistics over subjects" },
];

export function Builder({ pipeline, draft }: { pipeline?: string; draft?: PipelineSpec }) {
  const info = useDataset();
  const tools = useTools();
  const [spec, setSpecState] = useState<PipelineSpec | null>(() => draft ?? (pipeline ? null : kept));
  const [open, setOpen] = useState<string | null>(null);
  const [check, setCheck] = useState<CheckResult | null>(null);
  const [checking, setChecking] = useState(false);
  const { go } = useApp();

  const setSpec = (next: PipelineSpec) => {
    kept = next;
    setSpecState(next);
  };

  // Start from a pipeline, the kept draft, or an empty pipeline for the dataset's task.
  useEffect(() => {
    if (draft) {
      kept = draft;
      return;
    }
    if (pipeline) {
      void api.pipeline(pipeline).then(({ spec: s }) => setSpec(s));
    } else if (!spec && info.summary) {
      setSpec(emptySpec(info.summary.tasks[0] ?? "task"));
    }
  }, [pipeline, draft, info.summary]);

  // Check as the user edits (half a second after the last change).
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (!spec || spec.steps.length === 0) {
      setCheck(null);
      return;
    }
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      setChecking(true);
      api
        .checkPipeline(spec, info.path)
        .then(setCheck, () => setCheck(null))
        .finally(() => setChecking(false));
    }, 500);
    return () => window.clearTimeout(timer.current);
  }, [spec, info.path]);

  const menu = useMemo(() => {
    const out: Record<StepGroup, ToolInfo[]> = { clean: [], measure: [], test: [] };
    for (const tool of tools?.values() ?? []) {
      const group = stepGroup(tool);
      if (group) out[group].push(tool);
    }
    return out;
  }, [tools]);
  const suggestions = useSuggestions(spec);

  if (!spec || !tools) return <Loading />;

  const problems = problemsByStep(check);
  const tasks = [...new Set([...(info.summary?.tasks ?? []), spec.task])];

  const add = (tool: ToolInfo) => {
    const step: PipelineStep = { id: newStepId(spec, tool.name), tool: tool.name, args: {} };
    if (tool.per_recording) setSpec({ ...spec, steps: [...spec.steps, step] });
    else setSpec({ ...spec, group: [...spec.group, step] });
    setOpen(step.id);
  };
  const update = (group: boolean, index: number, change: Partial<PipelineStep>) => {
    const key = group ? "group" : "steps";
    setSpec({ ...spec, [key]: spec[key].map((s, i) => (i === index ? { ...s, ...change } : s)) });
  };
  const rename = (group: boolean, index: number, id: string) => {
    const taken = [...spec.steps, ...spec.group].some((s, i) => s.id === id && !(i === index + (group ? spec.steps.length : 0)));
    if (taken) return;
    update(group, index, { id });
    setOpen(id);
  };
  const remove = (group: boolean, index: number) => {
    const key = group ? "group" : "steps";
    setSpec({ ...spec, [key]: spec[key].filter((_, i) => i !== index) });
  };
  const move = (group: boolean, index: number, by: number) => {
    const key = group ? "group" : "steps";
    setSpec({ ...spec, [key]: moved(spec[key], index, by) });
  };

  const card = (step: PipelineStep, index: number, group: boolean) => {
    const list = group ? spec.group : spec.steps;
    return (
      <StepCard
        key={`${group ? "g" : "s"}-${index}`}
        index={index}
        step={step}
        spec={spec}
        group={group}
        tool={tools.get(step.tool)}
        problems={problems.get(step.id)}
        open={open === step.id}
        onToggle={() => setOpen(open === step.id ? null : step.id)}
        onChange={(args) => update(group, index, { args })}
        onRename={(id) => rename(group, index, id)}
        suggestions={suggestions}
        actions={
          <>
            <button type="button" className="icon-btn" aria-label={`Move ${step.id} up`} disabled={index === 0} onClick={() => move(group, index, -1)}>
              <Icon.Up />
            </button>
            <button
              type="button"
              className="icon-btn"
              aria-label={`Move ${step.id} down`}
              disabled={index === list.length - 1}
              onClick={() => move(group, index, 1)}
            >
              <Icon.Down />
            </button>
            <button type="button" className="icon-btn" aria-label={`Remove ${step.id}`} onClick={() => remove(group, index)}>
              <Icon.Close />
            </button>
          </>
        }
      />
    );
  };

  const overall = problems.get("")?.messages ?? [];

  return (
    <main className="workspace">
      <div className="row">
        <h1>Pipeline builder</h1>
        <span className="spacer" />
        <button
          type="button"
          className="btn"
          onClick={() => {
            if (spec.steps.length === 0 || window.confirm("Start again with an empty pipeline?")) {
              setSpec(emptySpec(info.summary?.tasks[0] ?? spec.task));
              setOpen(null);
            }
          }}
        >
          New pipeline
        </button>
        <button type="button" className="btn" onClick={() => go({ name: "pipelines" })}>
          Open a template…
        </button>
      </div>

      <section className="card">
        <div className="form-grid">
          <label className="field">
            Name
            <input className="text mono" value={spec.name} onChange={(e) => setSpec({ ...spec, name: e.target.value.toLowerCase() })} />
          </label>
          <label className="field">
            Task
            <select className="text" value={spec.task} onChange={(e) => setSpec({ ...spec, task: e.target.value })}>
              {tasks.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>
          <label className="field" style={{ gridColumn: "1 / -1" }}>
            Description
            <input
              className="text"
              value={spec.description}
              placeholder="What the pipeline is for"
              onChange={(e) => setSpec({ ...spec, description: e.target.value })}
            />
          </label>
        </div>
      </section>

      <section className="stack" style={{ gap: 8 }}>
        <div className="row">
          <h2>Steps for each subject</h2>
          <span className="small muted">run in this order on every subject's recording</span>
          <span className="spacer" />
          {checking ? <span className="tiny muted">Checking…</span> : check ? <CheckSummary check={check} /> : null}
        </div>
        {spec.steps.length === 0 ? <div className="card compact muted">No steps yet: add one from the menu below.</div> : null}
        {spec.steps.map((s, i) => card(s, i, false))}
        {spec.group.length ? (
          <>
            <div className="row" style={{ paddingTop: 8 }}>
              <h2>Over all subjects</h2>
              <span className="small muted">run once, after every subject's steps</span>
            </div>
            {spec.group.map((s, i) => card(s, i, true))}
          </>
        ) : null}
        {overall.map((m) => (
          <div key={m} className={`notice ${m.startsWith("Warning") ? "warn" : "error"}`}>
            {m}
          </div>
        ))}
      </section>

      <section className="card compact">
        <h2>Add a step</h2>
        <div className="grid cols-3">
          {GROUPS.map((g) => (
            <div key={g.key} className="stack" style={{ gap: 6 }}>
              <div>
                <span style={{ fontWeight: 600 }}>{g.label}</span> <span className="tiny muted">{g.hint}</span>
              </div>
              {menu[g.key].map((tool) => (
                <button key={tool.name} type="button" className="btn small" style={{ justifyContent: "flex-start" }} onClick={() => add(tool)} title={tool.description}>
                  <Icon.Plus /> <span className="mono">{tool.name}</span> <RiskPill tool={tool} />
                </button>
              ))}
            </div>
          ))}
        </div>
      </section>

      {spec.steps.length > 0 ? (
        <section className="card compact">
          <SaveAs spec={spec} suggested={spec.name} overwrite onSaved={(name) => {
            info.reload();
            go({ name: "pipelines", pipeline: name });
          }} />
        </section>
      ) : null}

      {spec.steps.length > 0 ? <RunPanel spec={spec} name={null} check={check} onChecked={setCheck} /> : null}
    </main>
  );
}

function CheckSummary({ check }: { check: CheckResult }) {
  const errors = check.problems.length + check.findings.filter((f) => f.severity === "error").length;
  const warnings = check.findings.filter((f) => f.severity === "warning").length;
  if (errors) return <span className="pill danger">{errors} to fix</span>;
  if (warnings) return <span className="pill warn">Checked · {warnings} warning{warnings > 1 ? "s" : ""}</span>;
  return <span className="pill ok">Checked: every step gets its data</span>;
}
