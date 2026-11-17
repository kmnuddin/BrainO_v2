/** What the pipeline screens share: a step with its form, and the panel that checks and runs a
 * pipeline on the open dataset. */

import { type ReactNode, useMemo, useState } from "react";
import { ApiError, api } from "../engine/api";
import type { CheckResult, Job, PipelineSpec, PipelineStep, ToolInfo } from "../engine/types";
import {
  GROUP_RUNNER_FIELDS,
  RUNNER_FIELDS,
  STEP_ID_PATTERN,
  type StepProblems,
  inputOf,
  problemsByStep,
} from "../lib/pipeline";
import { plural } from "../lib/format";
import { useApp, useJobs } from "../state/app";
import { useDataset } from "../state/dataset";
import { isActive, samePath } from "../state/jobs";
import { SchemaForm } from "./SchemaForm";
import * as Icon from "./icons";
import { ErrorNotice, JobLine, Pill } from "./ui";

export function RiskPill({ tool }: { tool: ToolInfo | undefined }) {
  if (!tool) return <Pill tone="danger">Unknown tool</Pill>;
  if (tool.risk === "decision") return <Pill tone="warn">You decide</Pill>;
  return null;
}

/** One step: its number, id and tool, folding open to its form. */
export function StepCard({
  index,
  step,
  spec,
  group,
  tool,
  problems,
  open,
  onToggle,
  onChange,
  readOnly = false,
  actions,
  suggestions,
  onRename,
}: {
  index: number;
  step: PipelineStep;
  spec: PipelineSpec;
  group: boolean;
  tool: ToolInfo | undefined;
  problems: StepProblems | undefined;
  open: boolean;
  onToggle: () => void;
  onChange?: (args: Record<string, unknown>) => void;
  readOnly?: boolean;
  actions?: ReactNode;
  suggestions?: Record<string, string[]>;
  /** Rename the step (the builder). */
  onRename?: (id: string) => void;
}) {
  const input = inputOf(spec, step, group);
  const bad = !!problems && (problems.messages.some((m) => !m.startsWith("Warning")) || Object.keys(problems.fields).length > 0);
  const set = Object.keys(step.args).length;
  return (
    <div className={`step-card${open ? " selected" : ""}${bad ? " invalid" : ""}`}>
      <div className="row" style={{ gap: 0, flexWrap: "nowrap" }}>
        <button type="button" className="head" onClick={onToggle} aria-expanded={open}>
          <span className="step-num">{group ? "G" : index + 1}</span>
          <span style={{ fontWeight: 600 }} className="mono">
            {step.id}
          </span>
          <span className="small muted">{tool ? tool.name : step.tool}</span>
          <RiskPill tool={tool} />
          {set ? <span className="tiny muted">{plural(set, "setting")} changed</span> : null}
          <span className="spacer" />
          {bad ? <Pill tone="danger">Needs a fix</Pill> : null}
        </button>
        {actions ? <div className="row" style={{ gap: 2, paddingRight: 8, flexWrap: "nowrap" }}>{actions}</div> : null}
      </div>
      {open ? (
        <div className="body-inner stack">
          {tool ? <p className="small muted">{tool.description}</p> : null}
          {onRename ? (
            <label className="row small" style={{ gap: 8 }}>
              <span className="muted">Step name</span>
              <RenameInput id={step.id} onRename={onRename} />
            </label>
          ) : null}
          <div className="small">
            {group ? "Reads every subject's " : "Uses "}
            {input.step ? (
              <>
                {group ? "results of " : "the output of "}
                <span className="mono">{input.step.id}</span> ({input.step.tool})
              </>
            ) : input.needs ? (
              <span style={{ color: "var(--danger)" }}>
                {input.needs}'s output, but there is no {input.needs} step before it
              </span>
            ) : (
              "the recording itself"
            )}
          </div>
          {problems?.messages.map((m) => (
            <div key={m} className={`notice ${m.startsWith("Warning") ? "warn" : "error"}`}>
              {m}
            </div>
          ))}
          {tool ? (
            <SchemaForm
              schema={tool.input_schema}
              value={step.args}
              hidden={group ? GROUP_RUNNER_FIELDS : RUNNER_FIELDS}
              problems={problems?.fields}
              readOnly={readOnly || !onChange}
              suggestions={suggestions}
              onChange={(args) => onChange?.(args)}
            />
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

/** A step's name, applied when it is valid and unique (checked by the caller). */
function RenameInput({ id, onRename }: { id: string; onRename: (id: string) => void }) {
  const [text, setText] = useState(id);
  const valid = STEP_ID_PATTERN.test(text);
  return (
    <input
      className={`text mono${valid ? "" : " invalid"}`}
      value={text}
      aria-label="Step name"
      onChange={(e) => {
        const value = e.target.value.toLowerCase();
        setText(value);
        if (STEP_ID_PATTERN.test(value)) onRename(value);
      }}
    />
  );
}

/** Words the forms can offer: the conditions a spec defines, and the dataset's events. */
export function useSuggestions(spec: PipelineSpec | null): Record<string, string[]> {
  const { summary } = useDataset();
  return useMemo(() => {
    const events = new Set<string>();
    for (const r of summary?.recordings ?? []) for (const e of Object.keys(r.events)) events.add(e);
    const conditions = new Set<string>();
    for (const s of spec?.steps ?? []) {
      const c = s.args.conditions;
      if (c && typeof c === "object" && !Array.isArray(c)) for (const k of Object.keys(c)) conditions.add(k);
    }
    return { events: [...events].sort(), conditions: [...conditions] };
  }, [summary, spec]);
}

/** The steps of a spec, each with its form. */
export function StepList({
  spec,
  tools,
  check,
  onChange,
  readOnly,
}: {
  spec: PipelineSpec;
  tools: Map<string, ToolInfo> | undefined;
  check: CheckResult | null;
  onChange?: (spec: PipelineSpec) => void;
  readOnly?: boolean;
}) {
  const [open, setOpen] = useState<string | null>(null);
  const problems = problemsByStep(check);
  const suggestions = useSuggestions(spec);
  const setArgs = (group: boolean, index: number, args: Record<string, unknown>) => {
    const key = group ? "group" : "steps";
    onChange?.({ ...spec, [key]: spec[key].map((s, i) => (i === index ? { ...s, args } : s)) });
  };
  return (
    <div className="stack" style={{ gap: 8 }}>
      {spec.steps.map((step, i) => (
        <StepCard
          key={step.id}
          index={i}
          step={step}
          spec={spec}
          group={false}
          tool={tools?.get(step.tool)}
          problems={problems.get(step.id)}
          open={open === step.id}
          onToggle={() => setOpen(open === step.id ? null : step.id)}
          onChange={onChange ? (args) => setArgs(false, i, args) : undefined}
          readOnly={readOnly}
          suggestions={suggestions}
        />
      ))}
      {spec.group.length ? <div className="section-label" style={{ padding: "8px 0 0" }}>Over all subjects</div> : null}
      {spec.group.map((step, i) => (
        <StepCard
          key={step.id}
          index={i}
          step={step}
          spec={spec}
          group
          tool={tools?.get(step.tool)}
          problems={problems.get(step.id)}
          open={open === step.id}
          onToggle={() => setOpen(open === step.id ? null : step.id)}
          onChange={onChange ? (args) => setArgs(true, i, args) : undefined}
          readOnly={readOnly}
          suggestions={suggestions}
        />
      ))}
    </div>
  );
}

/** Check and run a pipeline on the open dataset: all subjects or chosen ones. ``name`` runs a
 * saved or built-in pipeline as it is; otherwise ``spec`` is sent. */
export function RunPanel({
  spec,
  name,
  check,
  onChecked,
}: {
  spec: PipelineSpec;
  name: string | null;
  check: CheckResult | null;
  onChecked: (check: CheckResult | null) => void;
}) {
  const { go } = useApp();
  const info = useDataset();
  const jobs = useJobs();
  const [chooseSubjects, setChooseSubjects] = useState(false);
  const [chosen, setChosen] = useState<Set<string>>(new Set());
  const [autoApprove, setAutoApprove] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const available = useMemo(
    () => [...new Set((info.summary?.recordings ?? []).filter((r) => r.task === spec.task).map((r) => r.subject))].sort(),
    [info.summary, spec.task],
  );
  const selected = spec.subjects ?? available.filter((s) => !spec.exclude_subjects.includes(s));
  const subjects = chooseSubjects ? selected.filter((s) => chosen.has(s)) : null;
  const state = info.pipelines.find((p) => p.item.name === spec.name)?.state;
  const withResults = selected.filter((s) => {
    const done = state?.subjects[s] ?? {};
    return spec.steps.length > 0 && spec.steps.every((step) => done[step.id]?.status === "done");
  }).length;
  const running = jobs.find(
    (j) => j.kind === "pipeline" && isActive(j) && samePath(j.dataset, info.path) && j.title === `Pipeline ${spec.name}`,
  );
  const job: Job | undefined = jobs.find((j) => j.id === jobId) ?? running;
  const taskMissing = info.summary && !info.summary.tasks.includes(spec.task);

  const runCheck = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await api.checkPipeline(spec, info.path, subjects);
      onChecked(result);
      return result;
    } catch (e) {
      setError(e);
      return null;
    } finally {
      setBusy(false);
    }
  };

  const run = async () => {
    const result = await runCheck();
    if (!result?.valid) return;
    setBusy(true);
    try {
      const started = await api.runPipeline({
        dataset: info.path,
        ...(name ? { name } : { spec }),
        subjects,
        auto_approve: autoApprove,
      });
      setJobId(started.id);
    } catch (e) {
      if (e instanceof ApiError && e.status === 422 && e.detail && typeof e.detail === "object") {
        onChecked(e.detail as CheckResult);
      }
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const errors = check ? check.problems.length + check.findings.filter((f) => f.severity === "error").length : 0;
  const warnings = check ? check.findings.filter((f) => f.severity === "warning").length : 0;

  return (
    <section className="card">
      <h2>Run on {info.summary ? "this dataset" : "…"}</h2>
      {taskMissing ? (
        <div className="notice warn">
          This pipeline analyses task <b>{spec.task}</b>, which this dataset does not have (it has{" "}
          {info.summary?.tasks.join(", ") || "no task"}).
        </div>
      ) : null}
      <div className="row">
        <label className="row small" style={{ gap: 6 }}>
          <input type="radio" checked={!chooseSubjects} onChange={() => setChooseSubjects(false)} />
          All {plural(selected.length, "subject")}
        </label>
        <label className="row small" style={{ gap: 6 }}>
          <input
            type="radio"
            checked={chooseSubjects}
            onChange={() => {
              setChooseSubjects(true);
              if (chosen.size === 0) setChosen(new Set(selected));
            }}
          />
          Chosen subjects
        </label>
      </div>
      {chooseSubjects ? (
        <div className="chips">
          {selected.map((s) => (
            <button
              key={s}
              type="button"
              className="chip mono"
              aria-pressed={chosen.has(s)}
              onClick={() => {
                const next = new Set(chosen);
                if (next.has(s)) next.delete(s);
                else next.add(s);
                setChosen(next);
              }}
            >
              sub-{s}
            </button>
          ))}
        </div>
      ) : null}
      {withResults > 0 ? (
        <p className="small muted">
          {plural(withResults, "subject")} already {withResults === 1 ? "has" : "have"} results for every step: steps
          whose settings and data have not changed are reused, not run again.
        </p>
      ) : null}
      <label className="check">
        <input type="checkbox" checked={autoApprove} onChange={(e) => setAutoApprove(e.target.checked)} />
        Approve every proposal as it is made (in your name), instead of deciding each one
      </label>
      {check ? (
        <div className={`notice ${errors ? "error" : warnings ? "warn" : "ok"}`}>
          {errors
            ? `The check found ${plural(errors, "problem")}: see the steps marked “Needs a fix”.`
            : warnings
              ? `The check passed with ${plural(warnings, "warning")}.`
              : "The check passed: every step gets the data it needs."}
          {check.problems.length + check.findings.length > 0 ? (
            <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
              {[...check.problems, ...check.findings.map((f) => `${f.step ? `${f.step}: ` : ""}${f.message}`)].map((m) => (
                <li key={m}>{m}</li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
      <ErrorNotice error={error} />
      {job ? <JobLine job={job} /> : null}
      <div className="row">
        <button type="button" className="btn" disabled={busy} onClick={() => void runCheck()}>
          Check
        </button>
        <span className="spacer" />
        {job && !isActive(job) && job.status === "done" ? (
          <button type="button" className="btn" onClick={() => go({ name: "overview" })}>
            See the subjects <Icon.ChevronRight />
          </button>
        ) : null}
        <button
          type="button"
          className="btn primary"
          disabled={busy || !!running || taskMissing || (chooseSubjects && chosen.size === 0)}
          onClick={() => void run()}
        >
          {running ? "Running…" : `Run on ${plural(subjects?.length ?? selected.length, "subject")}`}
        </button>
      </div>
    </section>
  );
}
