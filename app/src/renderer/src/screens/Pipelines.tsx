/** Pipelines: templates filled in from a form, and the built-in and saved pipelines with each
 * step's form. Any of them runs on the open dataset, or opens in the builder to be adapted. */

import { useEffect, useState } from "react";
import { RunPanel, StepList } from "../components/PipelineParts";
import { SchemaForm } from "../components/SchemaForm";
import { ErrorNotice, Loading, Pill } from "../components/ui";
import { api } from "../engine/api";
import type { CheckResult, PipelineListItem, PipelineSpec, TemplateInfo } from "../engine/types";
import { plural } from "../lib/format";
import { NAME_PATTERN, sameSpec } from "../lib/pipeline";
import { useAsync } from "../lib/useAsync";
import { useApp, useTools } from "../state/app";
import { useDataset } from "../state/dataset";

type Selection = { kind: "template"; name: string } | { kind: "pipeline"; name: string };

export function Pipelines({ initial }: { initial?: string }) {
  const info = useDataset();
  const list = useAsync(() => Promise.all([api.templates(), api.pipelines()]), []);
  const [selected, setSelected] = useState<Selection | null>(initial ? { kind: "pipeline", name: initial } : null);
  const [templates, pipelines] = list.data ?? [[], []];
  const tasks = new Set(info.summary?.tasks ?? []);

  // Choose the first pipeline for the dataset's task, else the first template.
  useEffect(() => {
    if (selected || !list.data) return;
    const fitting = pipelines.find((p) => p.task && tasks.has(p.task));
    setSelected(fitting ? { kind: "pipeline", name: fitting.name } : templates[0] ? { kind: "template", name: templates[0].name } : null);
  }, [list.data]);

  if (list.error) return <main className="workspace"><ErrorNotice error={list.error} /></main>;
  if (!list.data) return <Loading what="Reading the pipelines" />;

  const isSel = (kind: Selection["kind"], name: string) => selected?.kind === kind && selected.name === name;

  return (
    <main className="workspace">
      <div className="row">
        <h1>Pipelines</h1>
        <span className="spacer" />
      </div>
      <div className="split">
        <nav className="card compact" aria-label="Pipelines" style={{ gap: 4 }}>
          <div className="section-label" style={{ paddingLeft: 12 }}>
            New from a template
          </div>
          {templates.map((t) => (
            <button
              key={t.name}
              type="button"
              className="list-item"
              aria-current={isSel("template", t.name)}
              onClick={() => setSelected({ kind: "template", name: t.name })}
            >
              <span style={{ fontWeight: 600 }}>{titleOf(t.name)}</span>
              <span className="tiny muted">{t.description}</span>
            </button>
          ))}
          <div className="section-label" style={{ paddingLeft: 12, paddingTop: 12 }}>
            Built in and yours
          </div>
          {pipelines.map((p) => (
            <PipelineItem
              key={p.name}
              item={p}
              fits={!!p.task && tasks.has(p.task)}
              current={isSel("pipeline", p.name)}
              onClick={() => setSelected({ kind: "pipeline", name: p.name })}
            />
          ))}
        </nav>
        <div className="stack" style={{ minWidth: 0 }}>
          {selected?.kind === "template" ? (
            <TemplateView
              key={selected.name}
              template={templates.find((t) => t.name === selected.name)!}
              onSaved={(name) => {
                list.reload();
                info.reload();
                setSelected({ kind: "pipeline", name });
              }}
            />
          ) : selected ? (
            <PipelineView
              key={selected.name}
              name={selected.name}
              onSaved={(name) => {
                list.reload();
                info.reload();
                setSelected({ kind: "pipeline", name });
              }}
              onDeleted={() => {
                list.reload();
                info.reload();
                setSelected(null);
              }}
            />
          ) : (
            <div className="card muted">Choose a pipeline or a template.</div>
          )}
        </div>
      </div>
    </main>
  );
}

function titleOf(name: string): string {
  const text = name.replace(/-/g, " ").replace(/\berp\b/gi, "ERP");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function PipelineItem({
  item,
  fits,
  current,
  onClick,
}: {
  item: PipelineListItem;
  fits: boolean;
  current: boolean;
  onClick: () => void;
}) {
  return (
    <button type="button" className="list-item" aria-current={current} onClick={onClick}>
      <span className="row" style={{ gap: 6 }}>
        <span style={{ fontWeight: 600 }} className="mono">
          {item.name}
        </span>
        {item.source === "user" ? <Pill tone="info">Yours</Pill> : null}
        {item.error ? <Pill tone="danger">Broken</Pill> : null}
      </span>
      <span className="tiny muted">
        {item.error
          ? "Cannot be read"
          : `task ${item.task} · ${plural((item.steps?.length ?? 0) + (item.group?.length ?? 0), "step")}${fits ? "" : " · not this dataset's task"}`}
      </span>
    </button>
  );
}

function PipelineView({
  name,
  onSaved,
  onDeleted,
}: {
  name: string;
  onSaved: (name: string) => void;
  onDeleted: () => void;
}) {
  const { go } = useApp();
  const tools = useTools();
  const loaded = useAsync(() => api.pipeline(name), [name]);
  const [draft, setDraft] = useState<PipelineSpec | null>(null);
  const [check, setCheck] = useState<CheckResult | null>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    if (loaded.data) setDraft(structuredClone(loaded.data.spec));
  }, [loaded.data]);

  if (loaded.error) return <ErrorNotice error={loaded.error} />;
  if (!loaded.data || !draft) return <Loading />;
  const original = loaded.data.spec;
  const changed = !sameSpec(original, draft);
  const mine = loaded.data.source === "user";

  const remove = async () => {
    if (!window.confirm(`Delete your pipeline ${name}? Its results in datasets are kept.`)) return;
    try {
      await api.deletePipeline(name);
      onDeleted();
    } catch (e) {
      setError(e);
    }
  };

  return (
    <>
      <section className="card">
        <div className="row">
          <h2 className="mono">{original.name}</h2>
          {mine ? <Pill tone="info">Yours</Pill> : <Pill tone="todo">Built in</Pill>}
          <span className="spacer" />
          <button type="button" className="btn small" onClick={() => go({ name: "builder", draft })}>
            Open in the builder
          </button>
          {mine ? (
            <button type="button" className="btn small danger" onClick={() => void remove()}>
              Delete
            </button>
          ) : null}
        </div>
        {original.description ? <p className="muted">{original.description}</p> : null}
        <p className="small muted">
          Task <b>{original.task}</b>
          {original.session ? `, session ${original.session}` : ""} ·{" "}
          {original.subjects ? plural(original.subjects.length, "chosen subject") : "every subject"}
          {original.exclude_subjects.length ? `, leaving out ${original.exclude_subjects.join(", ")}` : ""}. Open a step to
          see or change its settings.
        </p>
        <ErrorNotice error={error} />
        {changed ? (
          <div className="notice warn row">
            <span>
              You changed settings. Run it as changed, or save it as your own pipeline
              {mine ? "" : " (built-in pipelines stay as they are)"}.
            </span>
            <span className="spacer" />
            <button type="button" className="btn small" onClick={() => setDraft(structuredClone(original))}>
              Undo the changes
            </button>
          </div>
        ) : null}
        <StepList
          spec={draft}
          tools={tools}
          check={check}
          onChange={(s) => {
            setDraft(s);
            setCheck(null);
          }}
        />
        {changed ? <SaveAs spec={draft} suggested={mine ? name : `${name}-mine`} overwrite={mine} onSaved={onSaved} /> : null}
      </section>
      <RunPanel spec={draft} name={changed ? null : name} check={check} onChecked={setCheck} />
    </>
  );
}

function TemplateView({ template, onSaved }: { template: TemplateInfo; onSaved: (name: string) => void }) {
  const { go } = useApp();
  const info = useDataset();
  const tools = useTools();
  const [settings, setSettings] = useState<Record<string, unknown>>({});
  const [spec, setSpec] = useState<PipelineSpec | null>(null);
  const [check, setCheck] = useState<CheckResult | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  const events = [...new Set((info.summary?.recordings ?? []).flatMap((r) => Object.keys(r.events)))].sort();

  const build = async () => {
    setBusy(true);
    setError(null);
    setCheck(null);
    try {
      setSpec(await api.buildTemplate(template.name, settings, info.path));
    } catch (e) {
      setSpec(null);
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <section className="card">
        <h2>{titleOf(template.name)}</h2>
        <p className="muted">{template.description}.</p>
        {events.length ? (
          <p className="small muted">
            Events in this dataset: <span className="mono">{events.slice(0, 16).join(", ")}</span>
            {events.length > 16 ? ", …" : ""}
          </p>
        ) : null}
        <SchemaForm
          schema={template.settings_schema}
          value={settings}
          suggestions={{ events }}
          onChange={(v) => {
            setSettings(v);
            setSpec(null);
          }}
        />
        <ErrorNotice error={error} />
        <div className="row">
          <span className="spacer" />
          <button type="button" className="btn primary" disabled={busy} onClick={() => void build()}>
            Make the pipeline
          </button>
        </div>
      </section>
      {spec ? (
        <>
          <section className="card">
            <div className="row">
              <h2>The pipeline</h2>
              <span className="small muted">
                task {spec.task} · {plural(spec.subjects?.length ?? 0, "subject")}
              </span>
              <span className="spacer" />
              <button type="button" className="btn small" onClick={() => go({ name: "builder", draft: spec })}>
                Open in the builder
              </button>
            </div>
            <StepList spec={spec} tools={tools} check={check} onChange={setSpec} />
            <SaveAs spec={spec} suggested={spec.name} overwrite={false} onSaved={onSaved} />
          </section>
          <RunPanel spec={spec} name={null} check={check} onChecked={setCheck} />
        </>
      ) : null}
    </>
  );
}

/** Save a spec as the user's pipeline under a name. */
export function SaveAs({
  spec,
  suggested,
  overwrite,
  onSaved,
}: {
  spec: PipelineSpec;
  suggested: string;
  overwrite: boolean;
  onSaved: (name: string) => void;
}) {
  const [name, setName] = useState(suggested);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const valid = NAME_PATTERN.test(name);
  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      const saved = await api.savePipeline({ ...spec, name });
      onSaved(saved.spec.name);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="row" style={{ gap: 8 }}>
        <label className="row small" style={{ gap: 8 }}>
          <span className="muted">Save as</span>
          <input
            className={`text mono${valid ? "" : " invalid"}`}
            value={name}
            onChange={(e) => setName(e.target.value.toLowerCase())}
            aria-label="Pipeline name"
          />
        </label>
        <button type="button" className="btn" disabled={!valid || busy} onClick={() => void save()}>
          {overwrite && name === suggested ? "Save" : "Save as my pipeline"}
        </button>
      </div>
      {!valid ? <div className="tiny muted">Lower-case letters, digits, “-” and “_”, starting with a letter or digit.</div> : null}
      <ErrorNotice error={error} />
    </div>
  );
}
